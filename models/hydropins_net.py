import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiablePriorGenerator(nn.Module):
    """
    Differentiable NDWI-AWEI Prior Generator (DPG).
    Computes remote sensing physical water indices (NDWI, AWEIsh, NDVI, MNDWI_est)
    and combines them with learnable convolutional projections to generate
    a calibrated water probability prior, certainty map, and rich prior features.
    """
    def __init__(self, in_channels=4, prior_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.prior_channels = prior_channels
        
        # 4 indices + 4 raw bands = 8 channels
        self.proj = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, prior_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prior_channels),
            nn.GELU()
        )
        
        # Generates soft water prior [B, 1, H, W] and spatial certainty [B, 1, H, W]
        self.prior_head = nn.Sequential(
            nn.Conv2d(prior_channels, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1)
        )

    def forward(self, x):
        # x is [B, 4, H, W]
        # Sentinel-2 channels: 0: Red (B4), 1: Green (B3), 2: Blue (B2), 3: NIR (B8)
        eps = 1e-6
        red = x[:, 0:1, :, :]
        green = x[:, 1:2, :, :]
        blue = x[:, 2:3, :, :]
        nir = x[:, 3:4, :, :]
        
        # McFeeters NDWI: (Green - NIR) / (Green + NIR)
        ndwi = (green - nir) / (green + nir + eps)
        
        # Automated Water Extraction Index with shadow elimination (AWEIsh)
        # Blue + 2.5 * Green - 1.5 * (NIR + Red) - 0.25 * SWIR (here SWIR approx with NIR/Red)
        awei = blue + 2.5 * green - 1.5 * (nir + red)
        
        # Normalized Difference Vegetation Index (NDVI)
        ndvi = (nir - red) / (nir + red + eps)
        
        # Green-Red Normalized Ratio (MNDWI proxy in VIS spectrum)
        gr_diff = (green - red) / (green + red + eps)
        
        indices = torch.cat([ndwi, awei, ndvi, gr_diff], dim=1)
        combined = torch.cat([x, indices], dim=1)  # [B, 8, H, W]
        
        prior_feats = self.proj(combined)  # [B, prior_channels, H, W]
        prior_logits = self.prior_head(prior_feats)
        
        soft_prior = torch.sigmoid(prior_logits[:, 0:1, :, :])
        certainty = torch.sigmoid(prior_logits[:, 1:2, :, :])
        
        return soft_prior, certainty, prior_feats


class SpectralGuidedCrossAttention(nn.Module):
    """
    Spectral-Guided Cross-Attention (SG-CA).
    Cross-attends deep visual features with downsampled spectral prior features.
    The certainty map modulates the attention scores, allowing strong spectral
    priors to guide ambiguous visual regions while visual context corrects noise.
    """
    def __init__(self, visual_channels, prior_channels, reduced_channels=None):
        super().__init__()
        if reduced_channels is None:
            reduced_channels = max(16, visual_channels // 2)
            
        self.reduced_channels = reduced_channels
        self.q_conv = nn.Conv2d(visual_channels, reduced_channels, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(prior_channels, reduced_channels, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(prior_channels, visual_channels, kernel_size=1, bias=False)
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(visual_channels + 1, visual_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(visual_channels),
            nn.Sigmoid()
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(visual_channels, visual_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(visual_channels),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_vis, f_prior, certainty):
        # f_vis: [B, C_v, H, W]
        # f_prior: [B, C_p, H_p, W_p]
        # certainty: [B, 1, H_c, W_c]
        b, c, h, w = f_vis.shape
        
        # Align prior resolution to f_vis if needed
        if f_prior.shape[2:] != (h, w):
            f_prior = F.interpolate(f_prior, size=(h, w), mode='bilinear', align_corners=False)
        if certainty.shape[2:] != (h, w):
            certainty = F.interpolate(certainty, size=(h, w), mode='bilinear', align_corners=False)
            
        q = self.q_conv(f_vis)    # [B, C_r, H, W]
        k = self.k_conv(f_prior)  # [B, C_r, H, W]
        v = self.v_conv(f_prior)  # [B, C_v, H, W]
        
        # Spatial channel-wise modulation
        sim = torch.sigmoid(q * k) * certainty  # [B, C_r, H, W]
        guided_v = v * F.interpolate(sim.mean(dim=1, keepdim=True), size=(h, w), mode='bilinear', align_corners=False)
        
        # Modulate with spatial gate
        gate = self.spatial_gate(torch.cat([f_vis, certainty], dim=1))
        f_att = gate * guided_v
        
        out = self.out_conv(f_vis + self.gamma * f_att)
        return out


class HydrostaticBoundaryConsistencyHead(nn.Module):
    """
    Hydrostatic Boundary Consistency Head (HBC-Head).
    Imposes hydrostatic boundary consistency by computing spatial gradient fields
    on multi-scale features and prior masks, enforcing edge sharpness and physical
    water-land level consistency before generating the final segmentation logits.
    """
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        # Sobel kernels for horizontal and vertical spatial derivatives
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        self.boundary_pred = nn.Conv2d(16, 1, kernel_size=1)
        
        self.final_cls = nn.Sequential(
            nn.Conv2d(in_channels + 16 + 1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, feat, soft_prior):
        # feat: [B, C, H, W]
        # soft_prior: [B, 1, H, W]
        if soft_prior.shape[2:] != feat.shape[2:]:
            soft_prior = F.interpolate(soft_prior, size=feat.shape[2:], mode='bilinear', align_corners=False)
            
        # Compute gradient of soft prior
        gx = F.conv2d(soft_prior, self.sobel_x, padding=1)
        gy = F.conv2d(soft_prior, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        
        # Edge-consistent representation
        edge_input = torch.cat([feat, soft_prior, grad_mag], dim=1)
        edge_feats = self.edge_conv(edge_input)
        boundary_map = torch.sigmoid(self.boundary_pred(edge_feats))
        
        # Combine deep features, edge features, and boundary guidance
        fused = torch.cat([feat, edge_feats, boundary_map], dim=1)
        logits = self.final_cls(fused)
        return logits


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.res(x)


class HydroPINSNet(nn.Module):
    """
    HydroPINS-Net: Hydrostatic Prior-Informed Neural Segmentation Network for Flood Mapping.
    
    Trio Modules:
    1. DPG: Differentiable NDWI-AWEI Prior Generator
    2. SG-CA: Spectral-Guided Cross-Attention
    3. HBC-Head: Hydrostatic Boundary Consistency Head
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 1. Differentiable NDWI-AWEI Prior Generator
        prior_ch = 16
        self.dpg = DifferentiablePriorGenerator(in_channels=in_channels, prior_channels=prior_ch)
        
        # Visual Encoder Stages
        # Initial stem combines 4 bands + 16 prior channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        self.stem = ConvBlock(in_channels + prior_ch, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c1, c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c2, c3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c3, c4))
        
        # 2. Spectral-Guided Cross-Attention Modules
        self.sg_ca_stage3 = SpectralGuidedCrossAttention(c3, prior_ch)
        self.sg_ca_stage4 = SpectralGuidedCrossAttention(c4, prior_ch)
        
        # Bottleneck
        self.bottleneck = ConvBlock(c4, c4)
        
        # Decoder Stages
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)
        
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)
        
        # 3. Hydrostatic Boundary Consistency Head
        self.hbc_head = HydrostaticBoundaryConsistencyHead(c1, num_classes=num_classes)

    def forward(self, x):
        # Input: [B, 4, H, W]
        # 1. Prior Generation
        soft_prior, certainty, prior_feats = self.dpg(x)
        
        # Stem
        x_stem = torch.cat([x, prior_feats], dim=1)
        e1 = self.stem(x_stem)     # [B, c1, H, W]
        
        e2 = self.down1(e1)        # [B, c2, H/2, W/2]
        e3 = self.down2(e2)        # [B, c3, H/4, W/4]
        e3 = self.sg_ca_stage3(e3, prior_feats, certainty)
        
        e4 = self.down3(e3)        # [B, c4, H/8, W/8]
        e4 = self.sg_ca_stage4(e4, prior_feats, certainty)
        
        # Bottleneck
        b = self.bottleneck(e4)    # [B, c4, H/8, W/8]
        
        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 3. Hydrostatic Boundary Consistency Head
        logits = self.hbc_head(d1, soft_prior)
        return logits
