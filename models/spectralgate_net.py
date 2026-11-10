import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectanceRatioEmbeddingStem(nn.Module):
    """
    Reflectance Ratio Embedding Stem (RRE-Stem).
    Computes illumination-invariant physical spectral ratios:
      Green/NIR, Green/Red, Blue/NIR, NDWI, NDVI, NDBI_proxy
    and projects both raw reflectance and ratio signatures into a unified
    high-dimensional spectral embedding space.
    """
    def __init__(self, in_channels=4, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        # 4 raw bands + 6 ratio indices = 10 spectral channels
        num_spectral_ch = 10
        
        self.raw_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU()
        )
        
        self.ratio_proj = nn.Sequential(
            nn.Conv2d(6, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU()
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 4, H, W] -> Red=0, Green=1, Blue=2, NIR=3
        eps = 1e-6
        red = x[:, 0:1, :, :]
        green = x[:, 1:2, :, :]
        blue = x[:, 2:3, :, :]
        nir = x[:, 3:4, :, :]
        
        r_g_nir = green / (nir + eps)
        r_g_r = green / (red + eps)
        r_b_nir = blue / (nir + eps)
        ndwi = (green - nir) / (green + nir + eps)
        ndvi = (nir - red) / (nir + red + eps)
        ndbi_proxy = (red - green) / (red + green + eps)
        
        ratios = torch.cat([r_g_nir, r_g_r, r_b_nir, ndwi, ndvi, ndbi_proxy], dim=1)
        
        f_raw = self.raw_proj(x)
        f_ratio = self.ratio_proj(ratios)
        
        f_fused = torch.cat([f_raw, f_ratio], dim=1)
        return self.fuse(f_fused), ratios


class ModalityConfidenceSE(nn.Module):
    """
    Modality Confidence Squeeze-Excitation (MC-SE).
    Quantifies channel-wise confidence under environmental noise (haze, cloud shadows)
    using both first-order mean and second-order spatial variance statistics.
    Dynamically modulates channel weights to prioritize high-confidence features.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        inter_ch = max(8, channels // reduction)
        
        # Takes both spatial mean and spatial variance (2 * channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, inter_ch, bias=False),
            nn.GELU(),
            nn.Linear(inter_ch, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        mean = x.mean(dim=(2, 3))  # [B, C]
        var = ((x - mean.view(b, c, 1, 1)) ** 2).mean(dim=(2, 3))  # [B, C]
        
        stats = torch.cat([mean, var], dim=1)  # [B, 2*C]
        weights = self.mlp(stats).view(b, c, 1, 1)
        
        return x * weights


class SpectralSpatialBilateralDecoder(nn.Module):
    """
    Spectral-Spatial Bilateral Decoder (SSB-Dec).
    Applies bilateral filtering in deep feature space: features are smoothed
    based on spatial proximity and spectral range similarity, strictly preventing
    cross-boundary blurring between water and dry soil.
    """
    def __init__(self, encoder_channels, num_classes=2):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        
        # Bilateral kernel range projection
        self.range_proj = nn.Conv2d(c1, c1 // 2, kernel_size=1)
        
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
        
        # Deep Bilateral Refinement
        self.bilateral_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
        self.sigma_r = nn.Parameter(torch.tensor(1.0))
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(c1, num_classes, kernel_size=1)
        )

    def forward(self, e1, e2, e3, e4):
        # Stage 3
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        # Stage 2
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        # Stage 1
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Bilateral range similarity modulation
        range_feat = self.range_proj(d1)
        spatial_filtered = self.bilateral_conv(d1)
        
        # Similarity weight: exp(- |feat - filtered|^2 / (2 * sigma_r^2))
        diff = torch.norm(d1 - spatial_filtered, dim=1, keepdim=True)
        bilateral_weight = torch.exp(- (diff ** 2) / (2.0 * (torch.abs(self.sigma_r) + 0.1) ** 2))
        
        refined = d1 * bilateral_weight + spatial_filtered * (1.0 - bilateral_weight)
        
        logits = self.cls_head(refined)
        return logits


class SpectralGateNet(nn.Module):
    """
    SpectralGate-Net: Reflectance-Ratio Embedding and Modality Confidence Gated Network.
    
    Trio Modules:
    1. RRE-Stem: Reflectance Ratio Embedding Stem
    2. MC-SE: Modality Confidence Squeeze-Excitation
    3. SSB-Dec: Spectral-Spatial Bilateral Decoder
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        # 1. Reflectance Ratio Embedding Stem
        self.stem = ReflectanceRatioEmbeddingStem(in_channels=in_channels, out_channels=c1)
        
        # Encoder stages with 2. MC-SE blocks
        self.mc_se1 = ModalityConfidenceSE(c1)
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        self.mc_se2 = ModalityConfidenceSE(c2)
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        self.mc_se3 = ModalityConfidenceSE(c3)
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU(),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU()
        )
        self.mc_se4 = ModalityConfidenceSE(c4)
        
        # 3. Spectral-Spatial Bilateral Decoder
        self.decoder = SpectralSpatialBilateralDecoder(
            encoder_channels=[c1, c2, c3, c4],
            num_classes=num_classes
        )

    def forward(self, x):
        # 1. RRE-Stem
        e1, ratios = self.stem(x)  # [B, c1, H, W]
        e1 = self.mc_se1(e1)
        
        # Encoder
        e2 = self.down1(e1)
        e2 = self.mc_se2(e2)
        
        e3 = self.down2(e2)
        e3 = self.mc_se3(e3)
        
        e4 = self.down3(e3)
        e4 = self.mc_se4(e4)
        
        # 3. SSB-Dec
        logits = self.decoder(e1, e2, e3, e4)
        return logits
