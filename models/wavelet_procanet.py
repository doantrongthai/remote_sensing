import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Wavelet-ProCANet: Wavelet Progressive Cross-Attention Network for Flood Segmentation.

Multispectral flood imagery naturally encompasses low spatial frequency components
(broad inundated flood basins, uniform open water bodies) and high spatial frequency components
(delicate waterlines, riverbanks, urban drainage corridors). Wavelet-ProCANet decomposes
features into low-frequency (LL) and high-frequency (LH, HL, HH) subbands using a 2D Haar
Discrete Wavelet Transform (DWT), applies specialized cross-attention in each frequency subband,
and perfectly synthesizes multi-modal representations via Inverse DWT (IDWT).
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


class HaarDWT2D(nn.Module):
    """
    2D Haar Discrete Wavelet Transform.
    Decomposes an input [B, C, H, W] into 4 subbands:
      - LL: Low-frequency approximation
      - LH: Horizontal high frequencies
      - HL: Vertical high frequencies
      - HH: Diagonal high frequencies
    All subbands have spatial shape [B, C, H/2, W/2].
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Ensure dimensions are even
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode='replicate')

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """
    2D Inverse Haar Discrete Wavelet Transform.
    Reconstructs original spatial tensor [B, C, H, W] from 4 subbands [B, C, H/2, W/2].
    """
    def __init__(self):
        super().__init__()

    def forward(self, ll, lh, hl, hh):
        B, C, H_half, W_half = ll.shape
        r00 = (ll - lh - hl + hh) * 0.5
        r01 = (ll - lh + hl - hh) * 0.5
        r10 = (ll + lh - hl - hh) * 0.5
        r11 = (ll + lh + hl + hh) * 0.5

        out = torch.zeros(B, C, H_half * 2, W_half * 2, dtype=ll.dtype, device=ll.device)
        out[:, :, 0::2, 0::2] = r00
        out[:, :, 0::2, 1::2] = r01
        out[:, :, 1::2, 0::2] = r10
        out[:, :, 1::2, 1::2] = r11
        return out


class WaveletProCA(nn.Module):
    """
    Wavelet Progressive Cross-Attention Module.

    Processes multispectral features simultaneously in the spatial and frequency domains:
    1) Decomposes RGB and NIR streams into low-frequency (LL) and high-frequency (HF = [LH, HL, HH]) subbands.
    2) Low-frequency Cross-Attention:
       Fosters global contextual exchange of water presence vs surrounding dry land.
    3) High-frequency Cross-Attention:
       Sharply aligns waterline transitions and fine boundary contours.
    4) Haar IDWT reconstructs the frequency-attended feature map, which is merged with
       spatial self-attended cross-attention features.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        # Spatial Self-Attention
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

        # Spatial Cross-Attention
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

        # Low-frequency Cross-Attention (channels -> channels)
        self.cross_ll_r2n = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.cross_ll_n2r = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # High-frequency Subband Cross-Attention (LH, HL, HH processed with groups=3 to preserve directional independence)
        self.cross_hf_r2n = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.Sigmoid()
        )
        self.cross_hf_n2r = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.Sigmoid()
        )

        # Combined Fusion Refinement (Spatial fused + Wavelet reconstructed fused)
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_r, x_n):
        """
        Args:
            x_r: RGB+NIR feature map [B, C, H, W]
            x_n: NIR feature map [B, C, H, W]
        Returns:
            fused: Wavelet-enhanced cross-attended feature map [B, C, H, W]
        """
        # 1. Spatial Self-Attention
        a_r = self.self_conv_r(x_r)
        a_n = self.self_conv_n(x_n)
        x_r_self = x_r * a_r
        x_n_self = x_n * a_n

        # 2. Spatial Cross-Attention
        a_r2n = self.cross_conv_r2n(x_r_self)
        a_n2r = self.cross_conv_n2r(x_n_self)
        spatial_fused = (x_r_self * a_n2r) + (x_n_self * a_r2n)

        # 3. Wavelet 2D Haar Decomposition
        ll_r, lh_r, hl_r, hh_r = self.dwt(x_r)
        ll_n, lh_n, hl_n, hh_n = self.dwt(x_n)

        hf_r = torch.cat([lh_r, hl_r, hh_r], dim=1)
        hf_n = torch.cat([lh_n, hl_n, hh_n], dim=1)

        # 4. Low-Frequency Subband Cross-Attention
        a_ll_r2n = self.cross_ll_r2n(ll_r)
        a_ll_n2r = self.cross_ll_n2r(ll_n)
        ll_fused = (ll_r * a_ll_n2r) + (ll_n * a_ll_r2n)

        # 5. High-Frequency Subband Cross-Attention
        a_hf_r2n = self.cross_hf_r2n(hf_r)
        a_hf_n2r = self.cross_hf_n2r(hf_n)
        hf_fused = (hf_r * a_hf_n2r) + (hf_n * a_hf_r2n)

        c = self.channels
        lh_fused = hf_fused[:, 0:c, :, :]
        hl_fused = hf_fused[:, c:c*2, :, :]
        hh_fused = hf_fused[:, c*2:c*3, :, :]

        # 6. Wavelet Reconstruction (IDWT)
        wavelet_fused = self.idwt(ll_fused, lh_fused, hl_fused, hh_fused)

        # Align dimensions if pad was added
        if wavelet_fused.shape[-2:] != spatial_fused.shape[-2:]:
            wavelet_fused = wavelet_fused[:, :, :spatial_fused.shape[-2], :spatial_fused.shape[-1]]

        # 7. Joint Spatial-Frequency Fusion
        combined = torch.cat([spatial_fused, wavelet_fused], dim=1)
        return self.fuse_refine(combined)


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


class WaveletProCANet(nn.Module):
    """
    Wavelet-ProCANet: Wavelet Progressive Cross-Attention Network for Flood Segmentation.

    Integrates Haar 2D DWT low/high frequency cross-attention blocks into the progressive
    fusion hierarchy between RGB+NIR and NIR encoder streams.

    Args:
        in_channels (int): Input multispectral channels (default: 4 for Sentinel-2 B4, B3, B2, B8).
        num_classes (int): Target segmentation classes (default: 2 for flood water vs land).
        base_channels (int): Base channel depth (default: 32).
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

        # Wavelet-ProCA blocks at all stages
        self.proca1 = WaveletProCA(c)
        self.proca2 = WaveletProCA(c * 2)
        self.proca3 = WaveletProCA(c * 4)
        self.proca4 = WaveletProCA(c * 8)
        self.proca_bottleneck = WaveletProCA(c * 16)

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
