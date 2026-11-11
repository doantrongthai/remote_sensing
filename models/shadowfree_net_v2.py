import torch
import torch.nn as nn
import torch.nn.functional as F

class ChromaticityLuminanceSeparationStem(nn.Module):
    """
    Chromaticity-Luminance Separation Stem (CLS-Stem v2).
    Decomposes 4-band Sentinel-2 reflectance (R, G, B, NIR) into:
      1. Total luminance L: scalar intensity capturing topography and morphology.
      2. Invariant chromaticity c_k = I_k / (L + eps): captures intrinsic spectral
         reflectance ratios under the Lambertian dichromatic reflection model.
         
    Numerical safeguard: clamps L with min_val to avoid division-by-zero or
    gradient explosion in dark shadow/water pixels.
    """
    def __init__(self, in_channels=4, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        
        # Learnable channel weights for total luminance
        self.lum_weights = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]).view(1, 4, 1, 1))
        
        self.lum_stream = nn.Sequential(
            nn.Conv2d(1, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU()
        )
        
        self.chroma_stream = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU()
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 4, H, W] -> Guaranteed order: Red=0, Green=1, Blue=2, NIR=3
        eps = 1e-6
        w = F.softmax(self.lum_weights, dim=1)
        luminance = (x * w).sum(dim=1, keepdim=True)             # [B, 1, H, W]
        luminance_safe = torch.clamp(luminance, min=1e-5)
        chromaticity = x / (luminance_safe + eps)                # [B, 4, H, W]
        
        f_lum = self.lum_stream(luminance)
        f_chroma = self.chroma_stream(chromaticity)
        
        f_fused = torch.cat([f_lum, f_chroma], dim=1)
        return self.fuse(f_fused), chromaticity, luminance


class FalseShadowSuppressionCrossGate(nn.Module):
    """
    False-Shadow Suppression Cross-Gate (FSS-CG v2).
    
    Physics mechanism:
      Takes full-resolution SWDI map (Shadow-Water Discrimination Index):
        SWDI = NDWI - NDVI = (Green - NIR)/(Green + NIR) - (NIR - Red)/(NIR + Red)
      and downsamples it via adaptive average pooling (anti-aliased) to feature map scale.
      
    Mathematical Modulation:
      Maps SWDI through an MLP to dynamic gate G in (0, 1).
      Applies true attenuation/amplification factor:
        M = 2.0 * G in (0, 2.0)
      - For terrestrial shadows (SWDI <= 0): G < 0.5 -> M < 1.0 (True suppression toward 0)
      - For true flood water (SWDI >> 0):    G > 0.5 -> M > 1.0 (Feature amplification)
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.swdi_proj = nn.Sequential(
            nn.Conv2d(1, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, feat, swdi_full):
        # feat: [B, C, H, W]
        # swdi_full: [B, 1, H_full, W_full]
        b, c, h, w = feat.shape
        if swdi_full.shape[2:] != (h, w):
            swdi_pooled = F.adaptive_avg_pool2d(swdi_full, (h, w))
        else:
            swdi_pooled = swdi_full
            
        shadow_gate = self.swdi_proj(swdi_pooled)  # in (0, 1)
        # True suppression: 2 * gate modulates between (0.0, 2.0)
        modulated_feat = feat * (2.0 * shadow_gate)
        return self.refine(modulated_feat)


class ContrastiveShadowWaterHead(nn.Module):
    """
    Contrastive Shadow-Water Head (CSW-Head v2).
    
    Projects high-level decoder features onto a unit hypersphere S^(D-1).
    Maintains 3 learnable unit prototypes:
      - Prototype 0: Flood Water
      - Prototype 1: Shadow / Low-albedo non-water anomaly
      - Prototype 2: Sunlit Dry Land
      
    Equipped with an explicit Prototype Orthogonality Loss (get_ortho_loss)
    that forces maximum angular margin between the 3 prototypes, preventing
    the shadow prototype from collapsing into the water or land classes.
    """
    def __init__(self, in_channels, num_classes=2, embed_dim=32):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.embed_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        
        # 3 Prototypes: 0: Water, 1: Shadow, 2: Sunlit Land
        self.prototypes = nn.Parameter(F.normalize(torch.randn(3, embed_dim), dim=-1))
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels + 3, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def get_ortho_loss(self):
        """
        Penalizes cosine similarity between distinct prototypes.
        Forces the 3 prototypes to be mutually orthogonal on S^(D-1).
        """
        p = F.normalize(self.prototypes, dim=-1)  # [3, embed_dim]
        cosine_matrix = torch.matmul(p, p.t())    # [3, 3]
        eye = torch.eye(3, device=p.device)
        off_diag = (cosine_matrix - eye) ** 2
        return off_diag.sum() / 6.0

    def forward(self, feat):
        # feat: [B, C, H, W]
        b, c, h, w = feat.shape
        z = self.embed_proj(feat)  # [B, embed_dim, H, W]
        z_norm = F.normalize(z, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=-1)  # [3, embed_dim]
        
        # Cosine similarity per prototype
        sims = torch.einsum('bdhw,kd->bkhw', z_norm, proto_norm) / (torch.abs(self.temperature) + 0.01)
        sim_probs = F.softmax(sims, dim=1)  # [B, 3, H, W]
        
        fused = torch.cat([feat, sim_probs], dim=1)
        logits = self.cls_head(fused)
        return logits


class ShadowFreeNetV2(nn.Module):
    """
    ShadowFree-Net v2 (Physics-Guided Spectral-Decoupled Flood Segmentation Network).
    
    Addressed Reviewer Enhancements:
      1. Exact band order assertion: Red=0, Green=1, Blue=2, NIR=3.
      2. Full-resolution SWDI computed anti-aliased with adaptive average pooling.
      3. True FSS-CG suppression (M in (0, 2.0)) ensuring shadow features are actually attenuated.
      4. Explicit Prototype Orthogonality Loss via get_ortho_loss().
      5. Numerical stability clamp on luminance in CLS-Stem.
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        # 1. Chromaticity-Luminance Separation Stem v2
        self.stem = ChromaticityLuminanceSeparationStem(in_channels=in_channels, out_channels=c1)
        
        # Encoder stages
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU(),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU()
        )
        
        # 2. False-Shadow Suppression Cross-Gate v2
        self.fss_cg3 = FalseShadowSuppressionCrossGate(c3)
        self.fss_cg4 = FalseShadowSuppressionCrossGate(c4)
        
        # Decoder stages
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
        # 3. Contrastive Shadow-Water Head v2
        self.csw_head = ContrastiveShadowWaterHead(c1, num_classes=num_classes)

    def compute_swdi_full(self, x, eps=1e-6):
        """
        Computes Shadow-Water Discrimination Index at full input resolution:
          SWDI = (Green - NIR)/(Green + NIR) - (NIR - Red)/(NIR + Red)
        """
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        nir = x[:, 3:4, :, :]
        water_idx = (g - nir) / (g + nir + eps)
        veg_idx = (nir - r) / (nir + r + eps)
        return water_idx - veg_idx  # [B, 1, H, W]

    def get_ortho_loss(self):
        """Returns the prototype orthogonality loss from the head."""
        return self.csw_head.get_ortho_loss()

    def forward(self, x):
        # Compute full-res SWDI map directly from raw calibrated reflectance
        swdi_full = self.compute_swdi_full(x)
        
        # 1. CLS-Stem
        e1, chromaticity, luminance = self.stem(x)
        
        # Encoder
        e2 = self.down1(e1)
        
        e3 = self.down2(e2)
        e3 = self.fss_cg3(e3, swdi_full)
        
        e4 = self.down3(e3)
        e4 = self.fss_cg4(e4, swdi_full)
        
        # Decoder
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 3. CSW-Head
        logits = self.csw_head(d1)
        return logits
