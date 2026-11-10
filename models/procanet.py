import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Progressive Cross-Attention Network (ProCANet) for Remote Sensing Flood Detection
Reference:
    "Progressive Cross Attention Network for Flood Segmentation using Multispectral Satellite Imagery"
    Vicky Feliren, Fithrothul Khikmah, Irfan Dwiki Bhaswara, Bahrul I. Nasution,
    Alex M. Lechner, Muhamad Risqi U. Saputra.
    IEEE Geoscience and Remote Sensing Letters (GRSL), vol. 22, 2025.
    arXiv: 2501.11923
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


class ProCABlock(nn.Module):
    """
    Progressive Cross-Attention (ProCA) Block (Feliren et al., IEEE GRSL 2025).

    The module executes three sequential stages:
    1) Self-Attention Stage:
       Computes modality-specific attention maps a_R and a_N via 3x3 conv + Sigmoid,
       producing self-attended features X^_R = X_R * a_R and X^_N = X_N * a_N.
    2) Cross-Attention Stage:
       Computes cross-modal attention masks a_{R->N} and a_{N->R} from self-attended
       features, then modulates each stream with the opposite modality's mask:
       X~_R = X^_R * a_{N->R}
       X~_N = X^_N * a_{R->N}
    3) Cross-Attended Fusion:
       Merges the cross-attended features via element-wise addition:
       X~_{R+N} = X~_R + X~_N, followed by a refinement convolution.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Stage 1: Self-attention convolutions
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

        # Stage 3: Fusion refinement
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_r, x_n):
        """
        Args:
            x_r: Feature map from RGB+NIR encoder stream [B, C, H, W]
            x_n: Feature map from NIR-only encoder stream [B, C, H, W]
        Returns:
            fused: Cross-attended composite feature map [B, C, H, W]
        """
        # 1. Self-attention stage
        a_r = self.self_conv_r(x_r)
        a_n = self.self_conv_n(x_n)
        x_r_self = x_r * a_r
        x_n_self = x_n * a_n

        # 2. Cross-attention stage
        a_r2n = self.cross_conv_r2n(x_r_self)
        a_n2r = self.cross_conv_n2r(x_n_self)
        x_r_cross = x_r_self * a_n2r
        x_n_cross = x_n_self * a_r2n

        # 3. Cross Attended Fusion
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


class ProCANet(nn.Module):
    """
    ProCANet: Progressive Cross-Attention Network for Flood Segmentation.

    Dual-stream encoder architecture:
      - Stream 1: RGB + NIR bands (capturing spatial, textural, and spectral context).
      - Stream 2: NIR band only (capturing strong water-absorption signatures).
      - Multi-scale ProCA blocks placed at each encoder stage and bottleneck to
        progressively exchange cross-modal representations before UNet decoding.

    Args:
        in_channels (int): Total input channels (default: 4 for Sentinel-2 B4, B3, B2, B8).
        num_classes (int): Number of target classes (default: 2 for flood water vs land).
        base_channels (int): Number of base filters in the initial stage (default: 32).
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels

        # Modality channel assignments
        # Stream 1 receives all 4 bands (RGB + NIR) as verified optimal in Table IV of the paper.
        # Stream 2 receives the single NIR band (Band 8, index 3).
        self.s1_ch = in_channels
        self.s2_ch = 1 if in_channels >= 4 else max(1, in_channels // 2)

        # ------------------------------------------------------------------
        # Stream 1: RGB + NIR Encoder
        # ------------------------------------------------------------------
        self.enc1_conv1 = DoubleConv(self.s1_ch, c)
        self.pool1_r = nn.MaxPool2d(2)
        self.enc1_conv2 = DoubleConv(c, c * 2)
        self.pool2_r = nn.MaxPool2d(2)
        self.enc1_conv3 = DoubleConv(c * 2, c * 4)
        self.pool3_r = nn.MaxPool2d(2)
        self.enc1_conv4 = DoubleConv(c * 4, c * 8)
        self.pool4_r = nn.MaxPool2d(2)

        # ------------------------------------------------------------------
        # Stream 2: NIR Encoder
        # ------------------------------------------------------------------
        self.enc2_conv1 = DoubleConv(self.s2_ch, c)
        self.pool1_n = nn.MaxPool2d(2)
        self.enc2_conv2 = DoubleConv(c, c * 2)
        self.pool2_n = nn.MaxPool2d(2)
        self.enc2_conv3 = DoubleConv(c * 2, c * 4)
        self.pool3_n = nn.MaxPool2d(2)
        self.enc2_conv4 = DoubleConv(c * 4, c * 8)
        self.pool4_n = nn.MaxPool2d(2)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.bottleneck_r = DoubleConv(c * 8, c * 16)
        self.bottleneck_n = DoubleConv(c * 8, c * 16)

        # ------------------------------------------------------------------
        # Progressive Cross-Attention (ProCA) Blocks at all scales
        # ------------------------------------------------------------------
        self.proca1 = ProCABlock(c)
        self.proca2 = ProCABlock(c * 2)
        self.proca3 = ProCABlock(c * 4)
        self.proca4 = ProCABlock(c * 8)
        self.proca_bottleneck = ProCABlock(c * 16)

        # ------------------------------------------------------------------
        # Decoder Path
        # ------------------------------------------------------------------
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        # Final Classification Head
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def _split_modalities(self, x):
        """Split input tensor into RGB+NIR stream and NIR stream."""
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

        # Decoder path with ProCA fused skip connections
        d4 = self.dec4(b_fused, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        logits = self.head(d1)
        return logits
