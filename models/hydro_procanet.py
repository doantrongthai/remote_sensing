import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Hydro-ProCANet: Hydro-Spectral Progressive Cross-Attention Network for Flood Segmentation.

Incorporates a Learned Differentiable NDWI (Normalized Difference Water Index) spectral prior.
Water bodies exhibit distinct physical absorption in the NIR spectrum contrasted against Green
reflectance. Hydro-ProCANet formulates a learnable, differentiable hydro-spectral prior directly
from the raw physical bands and injects multi-scale hydrological guidance into the ProCA blocks.
"""


class ConvBNReLU(nn.Module):
    """Standard 2D Convolution + BatchNorm + ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive 3x3 Conv-BN-ReLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=1),
            ConvBNReLU(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class LearnedHydroSpectralPrior(nn.Module):
    """
    Learned Differentiable NDWI (Normalized Difference Water Index) Module.

    Calculates a parameterized, differentiable water spectral response:
        GNDWI = (w_g * Green - w_n * NIR + b) / (|w_g * Green| + |w_n * NIR| + eps)
    Followed by a calibrated sigmoid confidence estimator with learnable threshold theta:
        P_water = sigmoid((GNDWI - theta) / tau)
    Then projects the hydrological water probability into multi-scale hydrological features.
    """
    def __init__(self, out_channels=32):
        super().__init__()
        # Learnable spectral coefficients initialized to standard McFeeters NDWI weights
        self.w_green = nn.Parameter(torch.tensor([1.0]))
        self.w_nir = nn.Parameter(torch.tensor([1.0]))
        self.bias = nn.Parameter(torch.tensor([0.0]))
        self.theta = nn.Parameter(torch.tensor([0.0]))  # Standard NDWI threshold for water is 0.0
        self.tau = nn.Parameter(torch.tensor([0.2]))    # Smooth temperature scale

        # Feature projection for hydrological guidance
        self.prior_proj = nn.Sequential(
            nn.Conv2d(2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, green, nir):
        """
        Args:
            green: Green band tensor [B, 1, H, W]
            nir: NIR band tensor [B, 1, H, W]
        Returns:
            hydro_feat: Multi-channel hydrological prior features [B, out_channels, H, W]
        """
        wg = torch.clamp(self.w_green, min=0.01)
        wn = torch.clamp(self.w_nir, min=0.01)
        tau = torch.clamp(torch.abs(self.tau), min=0.05)

        num = (wg * green - wn * nir) + self.bias
        denom = torch.abs(wg * green) + torch.abs(wn * nir) + 1e-6
        gndwi = num / denom

        p_water = torch.sigmoid((gndwi - self.theta) / tau)

        # Concatenate raw normalized GNDWI index with continuous water confidence probability
        hydro_input = torch.cat([gndwi, p_water], dim=1)
        return self.prior_proj(hydro_input)


class HydroSpectralProCA(nn.Module):
    """
    Hydro-Spectral Progressive Cross-Attention Module.

    Conditions cross-modal attention on the physical hydro-spectral prior:
    1) Standard self-attention maps a_R and a_N compute intra-modal focus.
    2) A spatial hydrological gate S_hydro is extracted from the learned NDWI prior.
    3) Cross-modal exchange is dynamically emphasized in regions with hydrological ambiguity:
       a_{N->R} = a_{N->R} * (1.0 + S_hydro)
       a_{R->N} = a_{R->N} * (1.0 + S_hydro)
    4) Fused features are enriched with direct hydrological prior projections.
    """
    def __init__(self, channels, hydro_ch=32):
        super().__init__()
        self.channels = channels

        # Stage 1: Self-attention
        self.self_conv_r = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.self_conv_n = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Stage 2: Cross-attention
        self.cross_conv_r2n = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.cross_conv_n2r = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Hydro gate and projection
        self.hydro_gate = nn.Sequential(
            nn.Conv2d(hydro_ch, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.hydro_proj = nn.Sequential(
            nn.Conv2d(hydro_ch, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Stage 3: Fusion refinement
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_r, x_n, hydro_feat):
        """
        Args:
            x_r: RGB+NIR feature map [B, C, H, W]
            x_n: NIR feature map [B, C, H, W]
            hydro_feat: Hydro-spectral prior features [B, hydro_ch, H, W]
        Returns:
            fused: Hydro-spectral conditioned composite feature map [B, C, H, W]
        """
        if hydro_feat.shape[-2:] != x_r.shape[-2:]:
            hydro_feat = F.interpolate(hydro_feat, size=x_r.shape[-2:], mode='bilinear', align_corners=False)

        # 1. Self-attention
        a_r = self.self_conv_r(x_r)
        a_n = self.self_conv_n(x_n)
        x_r_self = x_r * a_r
        x_n_self = x_n * a_n

        # 2. Hydro gate
        s_hydro = self.hydro_gate(hydro_feat)

        # 3. Hydro-conditioned cross-attention
        a_r2n = self.cross_conv_r2n(x_r_self) * (1.0 + s_hydro)
        a_n2r = self.cross_conv_n2r(x_n_self) * (1.0 + s_hydro)

        x_r_cross = x_r_self * a_n2r
        x_n_cross = x_n_self * a_r2n

        # 4. Composite fusion with hydro prior embedding
        hydro_res = self.hydro_proj(hydro_feat)
        fused = x_r_cross + x_n_cross + hydro_res
        return self.fuse_refine(fused)


class DecoderBlock(nn.Module):
    """Upsampling followed by feature concatenation and double convolution."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_ch // 2) + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HydroProCANet(nn.Module):
    """
    Hydro-ProCANet: Hydro-Spectral Progressive Cross-Attention Network.

    Pairs dual-stream RGB+NIR and NIR encoders with a Learned Differentiable NDWI
    Spectral Prior Branch and HydroSpectralProCA modules at all stages.

    Args:
        in_channels (int): Multispectral input channels (default: 4 for Sentinel-2 B4, B3, B2, B8).
        num_classes (int): Segmentation output classes (default: 2 for flood water vs land).
        base_channels (int): Initial encoder stage channel width (default: 32).
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels
        hydro_ch = 32

        self.s1_ch = in_channels
        self.s2_ch = 1 if in_channels >= 4 else max(1, in_channels // 2)

        # Learned differentiable hydro-spectral prior
        self.hydro_prior = LearnedHydroSpectralPrior(out_channels=hydro_ch)

        # Stream 1: RGB + NIR Encoder
        self.enc1_conv1 = DoubleConv(self.s1_ch, c)
        self.pool1_r = nn.MaxPool2d(2)
        self.enc1_conv2 = DoubleConv(c, c * 2)
        self.pool2_r = nn.MaxPool2d(2)
        self.enc1_conv3 = DoubleConv(c * 2, c * 4)
        self.pool3_r = nn.MaxPool2d(2)
        self.enc1_conv4 = DoubleConv(c * 4, c * 8)
        self.pool4_r = nn.MaxPool2d(2)

        # Stream 2: NIR Encoder
        self.enc2_conv1 = DoubleConv(self.s2_ch, c)
        self.pool1_n = nn.MaxPool2d(2)
        self.enc2_conv2 = DoubleConv(c, c * 2)
        self.pool2_n = nn.MaxPool2d(2)
        self.enc2_conv3 = DoubleConv(c * 2, c * 4)
        self.pool3_n = nn.MaxPool2d(2)
        self.enc2_conv4 = DoubleConv(c * 4, c * 8)
        self.pool4_n = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck_r = DoubleConv(c * 8, c * 16)
        self.bottleneck_n = DoubleConv(c * 8, c * 16)

        # Multi-scale HydroSpectralProCA blocks
        self.proca1 = HydroSpectralProCA(c, hydro_ch=hydro_ch)
        self.proca2 = HydroSpectralProCA(c * 2, hydro_ch=hydro_ch)
        self.proca3 = HydroSpectralProCA(c * 4, hydro_ch=hydro_ch)
        self.proca4 = HydroSpectralProCA(c * 8, hydro_ch=hydro_ch)
        self.proca_bottleneck = HydroSpectralProCA(c * 16, hydro_ch=hydro_ch)

        # Decoder path
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        # Head
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def _split_modalities(self, x):
        """Split input into RGB+NIR, NIR, and Green bands."""
        if self.in_channels >= 4:
            x_rgb_nir = x[:, :4, :, :]
            x_nir = x[:, 3:4, :, :]
            x_green = x[:, 1:2, :, :]
        elif self.in_channels == 3:
            x_rgb_nir = x
            x_green = x[:, 1:2, :, :]
            x_nir = x[:, 2:3, :, :]
        else:
            x_rgb_nir = x
            x_green = x[:, 0:1, :, :]
            x_nir = x[:, -1:, :, :]
        return x_rgb_nir, x_nir, x_green

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Multispectral image tensor [B, in_channels, H, W]
        Returns:
            Tensor: Raw logits [B, num_classes, H, W]
        """
        x_r, x_n, x_g = self._split_modalities(x)

        # Extract differentiable NDWI hydro-spectral prior
        hydro_feat = self.hydro_prior(x_g, x_n)

        # Stage 1
        x1_r = self.enc1_conv1(x_r)
        x1_n = self.enc2_conv1(x_n)
        skip1 = self.proca1(x1_r, x1_n, hydro_feat)

        # Stage 2
        x2_r = self.enc1_conv2(self.pool1_r(x1_r))
        x2_n = self.enc2_conv2(self.pool1_n(x1_n))
        skip2 = self.proca2(x2_r, x2_n, hydro_feat)

        # Stage 3
        x3_r = self.enc1_conv3(self.pool2_r(x2_r))
        x3_n = self.enc2_conv3(self.pool2_n(x2_n))
        skip3 = self.proca3(x3_r, x3_n, hydro_feat)

        # Stage 4
        x4_r = self.enc1_conv4(self.pool3_r(x3_r))
        x4_n = self.enc2_conv4(self.pool3_n(x3_n))
        skip4 = self.proca4(x4_r, x4_n, hydro_feat)

        # Bottleneck
        b_r = self.bottleneck_r(self.pool4_r(x4_r))
        b_n = self.bottleneck_n(self.pool4_n(x4_n))
        b_fused = self.proca_bottleneck(b_r, b_n, hydro_feat)

        # Decoder
        d4 = self.dec4(b_fused, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        return self.head(d1)
