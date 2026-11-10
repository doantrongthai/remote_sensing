import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletBasisConv2d(nn.Module):
    """
    Wavelet-Basis Kolmogorov-Arnold Conv2d (Conv-WaveKAN) layer.
    Replaces static MLP / convolutional non-linearities with a combination of
    a base SiLU projection and learnable continuous Morlet wavelet basis expansions:
      y = Conv(SiLU(x)) + sum_k c_k * cos(w_k * (x - mu_k) / sigma_k) * exp(- (x - mu_k)^2 / (2 * sigma_k^2))
    Achieves non-linear representation capacity while remaining numerically stable.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_wavelets=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_wavelets = num_wavelets
        
        # Base linear convolution
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.base_bn = nn.BatchNorm2d(out_channels)
        
        # Wavelet parameters: learnable frequencies w, shifts mu, scales sigma
        self.freqs = nn.Parameter(torch.linspace(0.5, 3.0, num_wavelets).view(1, num_wavelets, 1, 1, 1))
        self.shifts = nn.Parameter(torch.zeros(1, num_wavelets, 1, 1, 1))
        self.scales = nn.Parameter(torch.ones(1, num_wavelets, 1, 1, 1))
        
        # Wavelet mixing convolution
        self.wavelet_conv = nn.Conv2d(in_channels * num_wavelets, out_channels, kernel_size=1, bias=False)
        self.wavelet_bn = nn.BatchNorm2d(out_channels)
        
        self.out_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, C, H, W]
        base = self.base_bn(self.base_conv(self.act(x)))
        
        # Compute wavelet basis responses
        # Expand x: [B, 1, C, H, W]
        x_exp = x.unsqueeze(1)
        sigma = torch.abs(self.scales) + 0.1
        norm_x = (x_exp - self.shifts) / sigma
        
        # Morlet wavelet: cos(w * norm_x) * exp(-norm_x^2 / 2)
        wavelet_vals = torch.cos(self.freqs * norm_x) * torch.exp(-0.5 * (norm_x ** 2))
        
        # Reshape to [B, K * C, H, W]
        b, k, c, h, w = wavelet_vals.shape
        wavelet_flat = wavelet_vals.view(b, k * c, h, w)
        wave_proj = self.wavelet_bn(self.wavelet_conv(wavelet_flat))
        
        return self.act(self.out_bn(base + wave_proj))


class WaveletBasisKANFeatureExtractor(nn.Module):
    """
    Wavelet-Basis KAN Feature Extractor (WB-KAN-FE).
    Combines Conv-WaveKAN residual blocks across encoder levels to model
    high-degree non-linear multispectral interactions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kan1 = WaveletBasisConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.kan2 = WaveletBasisConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.res(x)
        out = self.kan1(x)
        out = self.kan2(out)
        return out + residual


class MorletSplineCrossModalGating(nn.Module):
    """
    Morlet-Spline Cross-Modal Gating (MS-CMG).
    Computes a non-linear spectral water transition gate from NDWI and NIR bands
    via Morlet wavelet-spline expansion, adaptively gating spatial feature channels.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Wavelet projection for spectral guidance
        self.spec_proj = nn.Sequential(
            nn.Conv2d(2, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.w_freq = nn.Parameter(torch.ones(1, channels, 1, 1) * 2.0)
        self.w_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x, raw_s2):
        # x: [B, C, H, W]
        # raw_s2: [B, 4, H, W] -> Red=0, Green=1, Blue=2, NIR=3
        b, c, h, w = x.shape
        eps = 1e-6
        if raw_s2.shape[2:] != (h, w):
            raw_s2 = F.interpolate(raw_s2, size=(h, w), mode='bilinear', align_corners=False)
            
        green = raw_s2[:, 1:2, :, :]
        nir = raw_s2[:, 3:4, :, :]
        ndwi = (green - nir) / (green + nir + eps)
        ratio = green / (nir + 0.1)
        
        spec_in = torch.cat([ndwi, ratio], dim=1)
        spec_feat = self.spec_proj(spec_in)
        
        # Morlet-spline transform
        s = torch.abs(self.w_scale) + 0.1
        norm_spec = spec_feat / s
        morlet_gate = torch.cos(self.w_freq * norm_spec) * torch.exp(-0.5 * (norm_spec ** 2))
        
        combined = torch.cat([x, morlet_gate], dim=1)
        gate = self.gate_conv(combined)
        return x * gate


class MultiResolutionKANBoundaryHead(nn.Module):
    """
    Multi-Resolution KAN Boundary Head (MR-KAN-BH).
    Extracts multi-resolution edge representations using WaveKAN layers,
    forming a high-precision boundary certainty manifold for final logits.
    """
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.kan_edge = WaveletBasisConv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_cls = nn.Sequential(
            nn.Conv2d(in_channels + (in_channels // 2) + 1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        edge_feat = self.kan_edge(x)
        b_map = self.boundary_gate(edge_feat)
        fused = torch.cat([x, edge_feat, b_map], dim=1)
        logits = self.final_cls(fused)
        return logits


class WaveKANHydroNet(nn.Module):
    """
    WaveKAN-HydroNet: Wavelet-Basis Kolmogorov-Arnold Network for Hydrological Flood Segmentation.
    
    Trio Modules:
    1. WB-KAN-FE: Wavelet-Basis KAN Feature Extractor
    2. MS-CMG: Morlet-Spline Cross-Modal Gating
    3. MR-KAN-BH: Multi-Resolution KAN Boundary Head
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        # 1. WB-KAN Feature Extractor stages
        self.stem = WaveletBasisKANFeatureExtractor(in_channels, c1)
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            WaveletBasisKANFeatureExtractor(c1, c2)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            WaveletBasisKANFeatureExtractor(c2, c3)
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            WaveletBasisKANFeatureExtractor(c3, c4)
        )
        
        # 2. Morlet-Spline Cross-Modal Gating at intermediate stages
        self.ms_cmg2 = MorletSplineCrossModalGating(c2)
        self.ms_cmg3 = MorletSplineCrossModalGating(c3)
        
        # Decoder stages
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU()
        )
        
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        
        # 3. Multi-Resolution KAN Boundary Head
        self.mr_kan_bh = MultiResolutionKANBoundaryHead(c1, num_classes=num_classes)

    def forward(self, x):
        # 1. WB-KAN Stem
        e1 = self.stem(x)         # [B, c1, H, W]
        
        # Encoder
        e2 = self.down1(e1)        # [B, c2, H/2, W/2]
        e2 = self.ms_cmg2(e2, x)
        
        e3 = self.down2(e2)        # [B, c3, H/4, W/4]
        e3 = self.ms_cmg3(e3, x)
        
        e4 = self.down3(e3)        # [B, c4, H/8, W/8]
        
        # Decoder
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 3. Multi-Resolution KAN Boundary Head
        logits = self.mr_kan_bh(d1)
        return logits
