import torch
import torch.nn as nn
import torch.nn.functional as F

"""
BMG-ProCANet: Bidirectional Gated Progressive Cross-Attention Network for Flood Segmentation.

Addresses modality corruption and uncertainty in remote sensing (e.g. cloud shadows, haze,
or specular glint in visible RGB vs moist soil ambiguity in NIR). Standard cross-attention
can inadvertently propagate noise from an occluded modality into a clean modality.
BMG-ProCANet introduces dynamic Confidence Estimators (combining spatial and channel gating)
to regulate bidirectional cross-attention: modalities only transfer features when the source
is confident and the target has complementary uncertainty.
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


class ModalityConfidenceEstimator(nn.Module):
    """
    Joint Spatial and Channel Confidence Estimator for Remote Sensing Streams.
    Computes a continuous reliability confidence map C in [0, 1] for an input modality:
      - Channel confidence: Global average pooling + MLP evaluates overall spectral validity.
      - Spatial confidence: 1x1 conv evaluates local pixelwise sharpness and certainty.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_ch = max(8, channels // reduction)

        # Global channel confidence (Squeeze-and-Excitation branch)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=1, bias=True)
        )

        # Local spatial confidence branch
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        c_gate = self.channel_gate(x)
        s_gate = self.spatial_gate(x)
        # Combined confidence bounded between 0 and 1 via sigmoid
        confidence = torch.sigmoid(c_gate + s_gate)
        return confidence


class BidirectionalGatedProCA(nn.Module):
    """
    Bidirectional Gated Progressive Cross-Attention Module.

    1) Self-attention computes intra-modality importance:
       X^_R = X_R * a_R,  X^_N = X_N * a_N
    2) ModalityConfidenceEstimators compute reliability masks C_R and C_N in [0, 1].
    3) Directional asymmetric cross-transfer gates:
       Gate_{R->N} = C_R * (1.0 - C_N * 0.5)
       Gate_{N->R} = C_N * (1.0 - C_R * 0.5)
    4) Modulated cross-attention prevents corrupted regions from contaminating clean signals:
       X~_R = X^_R * (1.0 + Gate_{N->R} * a_{N->R})
       X~_N = X^_N * (1.0 + Gate_{R->N} * a_{R->N})
    5) Confidence-weighted composite fusion:
       X~_{fused} = (C_R * X~_R) + (C_N * X~_N), followed by refinement convolution.
    """
    def __init__(self, channels):
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

        # Stage 2: Cross-attention convolutions
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

        # Dynamic Modality Confidence Estimators
        self.conf_r = ModalityConfidenceEstimator(channels)
        self.conf_n = ModalityConfidenceEstimator(channels)

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
            fused: Confidence-gated cross-attended feature map [B, C, H, W]
        """
        # 1. Self-attention
        a_r = self.self_conv_r(x_r)
        a_n = self.self_conv_n(x_n)
        x_r_self = x_r * a_r
        x_n_self = x_n * a_n

        # 2. Modality confidence estimation
        c_r = self.conf_r(x_r_self)
        c_n = self.conf_n(x_n_self)

        # 3. Asymmetric cross-gating
        gate_r2n = c_r * (1.0 - c_n * 0.5)
        gate_n2r = c_n * (1.0 - c_r * 0.5)

        # 4. Cross-attention modulated by confidence transfer gates
        a_r2n = self.cross_conv_r2n(x_r_self)
        a_n2r = self.cross_conv_n2r(x_n_self)

        x_r_cross = x_r_self * (1.0 + gate_n2r * a_n2r)
        x_n_cross = x_n_self * (1.0 + gate_r2n * a_r2n)

        # 5. Confidence-weighted fusion
        fused = (c_r * x_r_cross) + (c_n * x_n_cross)
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


class BMGProCANet(nn.Module):
    """
    BMG-ProCANet: Bidirectional Gated Progressive Cross-Attention Network.

    Features dual-stream encoders with Bidirectional Gated Progressive Cross-Attention
    modules that evaluate modality confidence to protect against cloud/shadow or glint artifacts.

    Args:
        in_channels (int): Multispectral input channels (default: 4 for Sentinel-2 B4, B3, B2, B8).
        num_classes (int): Segmentation output classes (default: 2 for flood water vs land).
        base_channels (int): Channel capacity at initial stage (default: 32).
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
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

        # Multi-scale Bidirectional Gated ProCA blocks
        self.proca1 = BidirectionalGatedProCA(c)
        self.proca2 = BidirectionalGatedProCA(c * 2)
        self.proca3 = BidirectionalGatedProCA(c * 4)
        self.proca4 = BidirectionalGatedProCA(c * 8)
        self.proca_bottleneck = BidirectionalGatedProCA(c * 16)

        # Decoder path
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        # Segmentation Head
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
