import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Strip-ProCANet: Strip Progressive Cross-Attention Network for Flood Segmentation.

In remote sensing flood analysis, water bodies predominantly form elongated, anisotropic
structures (rivers, canals, flooded highway corridors, dykes, and coastal reaches). Standard
square convolution kernels exhibit an isotropic bias that can fragment narrow linear waterways.
Strip-ProCANet introduces anisotropic StripProCA blocks equipped with orthogonal (1xK and Kx1)
strip cross-convolutions to capture continuous long-range contextual relationships along
cardinal hydrological axes.
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


class AnisotropicStripConv(nn.Module):
    """
    Anisotropic Strip Convolution Branch.
    Extracts elongated horizontal (1xK) and vertical (Kx1) spatial context,
    then combines them via a 1x1 projection followed by BatchNorm and Sigmoid.
    """
    def __init__(self, channels, kernel_size=11):
        super().__init__()
        pad = kernel_size // 2

        # Horizontal strip convolution: aggregates across columns
        self.conv_h = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, pad), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # Vertical strip convolution: aggregates across rows
        self.conv_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # Anisotropic fusion to cross-attention mask
        self.combine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat_h = self.conv_h(x)
        feat_v = self.conv_v(x)
        # Element-wise addition of directional context before sigmoid gating
        return self.combine(feat_h + feat_v)


class StripProCA(nn.Module):
    """
    Strip Progressive Cross-Attention Module.

    Replaces isotropic square cross-attention convolutions with orthogonal strip cross-convolutions:
    1) Self-attention computes initial intra-modality importance maps via 3x3 convs.
    2) Anisotropic Cross-Attention computes horizontal (1xK) and vertical (Kx1) directional
       attention masks to preserve continuous linear flood channels.
    3) Modulates opposite modalities with anisotropic masks:
       X~_R = X^_R * a_{N->R}^{strip}
       X~_N = X^_N * a_{R->N}^{strip}
    4) Combines cross-attended features via residual fusion.
    """
    def __init__(self, channels, strip_kernel=11):
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

        # Stage 2: Anisotropic Strip Cross-Attention
        self.strip_r2n = AnisotropicStripConv(channels, kernel_size=strip_kernel)
        self.strip_n2r = AnisotropicStripConv(channels, kernel_size=strip_kernel)

        # Stage 3: Fusion refinement
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_r, x_n):
        """
        Args:
            x_r: RGB+NIR feature map [B, C, H, W]
            x_n: NIR feature map [B, C, H, W]
        Returns:
            fused: Anisotropically cross-attended composite feature map [B, C, H, W]
        """
        # 1. Self-attention
        a_r = self.self_conv_r(x_r)
        a_n = self.self_conv_n(x_n)
        x_r_self = x_r * a_r
        x_n_self = x_n * a_n

        # 2. Anisotropic Strip Cross-Attention
        a_r2n = self.strip_r2n(x_r_self)
        a_n2r = self.strip_n2r(x_n_self)

        x_r_cross = x_r_self * a_n2r
        x_n_cross = x_n_self * a_r2n

        # 3. Composite Fusion
        fused = x_r_cross + x_n_cross
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


class StripProCANet(nn.Module):
    """
    Strip-ProCANet: Strip Progressive Cross-Attention Network for Flood Segmentation.

    Dual-stream architecture with Anisotropic Strip Progressive Cross-Attention
    modules designed to preserve elongated river networks and drainage corridors.

    Args:
        in_channels (int): Input multispectral channels (default: 4 for Sentinel-2 B4, B3, B2, B8).
        num_classes (int): Segmentation target classes (default: 2 for flood water vs land).
        base_channels (int): Initial channel width (default: 32).
        strip_kernel (int): Anisotropic strip convolution kernel size (default: 11).
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32, strip_kernel=11):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels

        self.s1_ch = in_channels
        self.s2_ch = 1 if in_channels >= 4 else max(1, in_channels // 2)

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

        # Multi-scale StripProCA blocks
        self.proca1 = StripProCA(c, strip_kernel=strip_kernel)
        self.proca2 = StripProCA(c * 2, strip_kernel=strip_kernel)
        self.proca3 = StripProCA(c * 4, strip_kernel=strip_kernel)
        self.proca4 = StripProCA(c * 8, strip_kernel=strip_kernel)
        self.proca_bottleneck = StripProCA(c * 16, strip_kernel=strip_kernel)

        # Decoder path
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        # Head
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def _split_modalities(self, x):
        """Extract RGB+NIR and NIR bands."""
        if self.in_channels >= 4:
            x_rgb_nir = x[:, :4, :, :]
            x_nir = x[:, 3:4, :, :]
        elif self.in_channels == 3:
            x_rgb_nir = x
            x_nir = x[:, 2:3, :, :]
        else:
            x_rgb_nir = x
            x_nir = x[:, -1:, :, :]
        return x_rgb_nir, x_nir

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Multispectral image tensor [B, in_channels, H, W]
        Returns:
            Tensor: Raw logits [B, num_classes, H, W]
        """
        x_r, x_n = self._split_modalities(x)

        # Stage 1
        x1_r = self.enc1_conv1(x_r)
        x1_n = self.enc2_conv1(x_n)
        skip1 = self.proca1(x1_r, x1_n)

        # Stage 2
        x2_r = self.enc1_conv2(self.pool1_r(x1_r))
        x2_n = self.enc2_conv2(self.pool1_n(x1_n))
        skip2 = self.proca2(x2_r, x2_n)

        # Stage 3
        x3_r = self.enc1_conv3(self.pool2_r(x2_r))
        x3_n = self.enc2_conv3(self.pool2_n(x2_n))
        skip3 = self.proca3(x3_r, x3_n)

        # Stage 4
        x4_r = self.enc1_conv4(self.pool3_r(x3_r))
        x4_n = self.enc2_conv4(self.pool3_n(x3_n))
        skip4 = self.proca4(x4_r, x4_n)

        # Bottleneck
        b_r = self.bottleneck_r(self.pool4_r(x4_r))
        b_n = self.bottleneck_n(self.pool4_n(x4_n))
        b_fused = self.proca_bottleneck(b_r, b_n)

        # Decoder path
        d4 = self.dec4(b_fused, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        return self.head(d1)
