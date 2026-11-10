import torch
import torch.nn as nn
import torch.nn.functional as F

class ChromaticityLuminanceSeparationStem(nn.Module):
    """
    Chromaticity-Luminance Separation Stem (CLS-Stem).
    Decomposes 4-band Sentinel-2 reflectance into total luminance L and
    invariant chromaticity coordinates c_k = I_k / (L + eps).
    Since shadows drastically reduce luminance while largely preserving chromaticity ratios,
    decoupling them allows invariant surface reflectance extraction.
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
            nn.Conv2d(4, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU()
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 4, H, W] -> Red=0, Green=1, Blue=2, NIR=3
        eps = 1e-6
        # Normalized weights
        w = F.softmax(self.lum_weights, dim=1)
        luminance = (x * w).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        chromaticity = x / (luminance + eps)          # [B, 4, H, W]
        
        f_lum = self.lum_stream(luminance)
        f_chroma = self.chroma_stream(chromaticity)
        
        f_fused = torch.cat([f_lum, f_chroma], dim=1)
        return self.fuse(f_fused), chromaticity, luminance


class FalseShadowSuppressionCrossGate(nn.Module):
    """
    False-Shadow Suppression Cross-Gate (FSS-CG).
    Uses the Shadow-Water Discrimination Index (SWDI):
      SWDI = (c_Green - c_NIR) / (c_Green + c_NIR) - (c_NIR - c_Red) / (c_NIR + c_Red)
    to differentiate deep flood water (strong NIR absorption) from terrestrial cloud/mountain
    shadows (which retain vegetation NIR red-edge scattering).
    Suppresses false-shadow activation in feature maps.
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

    def forward(self, feat, chromaticity):
        # feat: [B, C, H, W]
        # chromaticity: [B, 4, H_c, W_c]
        b, c, h, w = feat.shape
        eps = 1e-6
        if chromaticity.shape[2:] != (h, w):
            chromaticity = F.interpolate(chromaticity, size=(h, w), mode='bilinear', align_corners=False)
            
        c_red = chromaticity[:, 0:1, :, :]
        c_green = chromaticity[:, 1:2, :, :]
        c_nir = chromaticity[:, 3:4, :, :]
        
        # SWDI calculation
        water_spec = (c_green - c_nir) / (c_green + c_nir + eps)
        veg_edge = (c_nir - c_red) / (c_nir + c_red + eps)
        swdi = water_spec - veg_edge  # High for water, low/negative for terrestrial shadow
        
        shadow_gate = self.swdi_proj(swdi)
        gated_feat = feat * shadow_gate
        return self.refine(feat + gated_feat)


class ContrastiveShadowWaterHead(nn.Module):
    """
    Contrastive Shadow-Water Head (CSW-Head).
    Projects features onto a normalized unit hypersphere and discriminates water
    against shadow and dry land using learnable prototypes (Water vs Shadow vs Land).
    Maximizes prototype margin to prevent false flood alarms in shadowed regions.
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
        
        # 3 Prototypes: 0: Water, 1: Cloud/Mountain Shadow, 2: Sunlit Dry Land
        self.prototypes = nn.Parameter(F.normalize(torch.randn(3, embed_dim), dim=-1))
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels + 3, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, feat):
        # feat: [B, C, H, W]
        b, c, h, w = feat.shape
        z = self.embed_proj(feat)  # [B, embed_dim, H, W]
        z_norm = F.normalize(z, dim=1)
        
        proto_norm = F.normalize(self.prototypes, dim=-1)  # [3, embed_dim]
        # Cosine similarity per prototype
        # z_norm: [B, D, H, W], proto_norm: [3, D]
        sims = torch.einsum('bdhw,kd->bkhw', z_norm, proto_norm) / (torch.abs(self.temperature) + 0.01)
        sim_probs = F.softmax(sims, dim=1)  # [B, 3, H, W]
        
        fused = torch.cat([feat, sim_probs], dim=1)
        logits = self.cls_head(fused)
        return logits


class ShadowFreeNet(nn.Module):
    """
    ShadowFree-Net: Chromaticity-Luminance Separation and Shadow-Suppressed Flood Segmentation.
    
    Trio Modules:
    1. CLS-Stem: Chromaticity-Luminance Separation Stem
    2. FSS-CG: False-Shadow Suppression Cross-Gate
    3. CSW-Head: Contrastive Shadow-Water Head
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        # 1. Chromaticity-Luminance Separation Stem
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
        
        # 2. False-Shadow Suppression Cross-Gate
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
        
        # 3. Contrastive Shadow-Water Head
        self.csw_head = ContrastiveShadowWaterHead(c1, num_classes=num_classes)

    def forward(self, x):
        # 1. CLS-Stem
        e1, chromaticity, luminance = self.stem(x)
        
        # Encoder
        e2 = self.down1(e1)
        
        e3 = self.down2(e2)
        e3 = self.fss_cg3(e3, chromaticity)
        
        e4 = self.down3(e3)
        e4 = self.fss_cg4(e4, chromaticity)
        
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
