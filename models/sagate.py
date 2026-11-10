"""
SA-Gate: Bi-Directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate (ECCV 2020)
Adapted for Sentinel-2 Multimodal Remote Sensing (RGB + NIR).

Original Paper: "Bi-directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate for RGB-D Semantic Segmentation"
Reference: https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch

Key Components:
- Dual-stream convolutional encoder (RGB stream + NIR stream)
- Separation-and-Aggregation Gate (SA-Gate):
  - Feature Separation (FS): Unimodal spatial and channel recalibration to filter noise
  - Feature Aggregation (FA): Bi-directional cross-modality gating and feature propagation
- Multi-scale Feature Pyramid Decoder for dense semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard 3x3 Conv-BN-ReLU residual block.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.relu(out)


class SeparationAggregationGate(nn.Module):
    """
    Separation-and-Aggregation Gate (SA-Gate).
    1. Feature Separation (FS): Recalibrates individual modality features across channel
       and spatial dimensions to eliminate unimodal noise / artifacts.
    2. Feature Aggregation (FA): Bi-directionally gates and propagates complementary cues
       between RGB and NIR streams.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(8, channels // reduction)

        # Feature Separation: RGB spatial and channel recalibration
        self.fs_c_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fs_s_rgb = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Feature Separation: NIR spatial and channel recalibration
        self.fs_c_nir = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fs_s_nir = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Feature Aggregation: Bi-directional propagation gates
        self.gate_nir_to_rgb = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.gate_rgb_to_nir = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Transformation for propagated features
        self.trans_nir = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.trans_rgb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_rgb, f_nir):
        # 1. Feature Separation (FS)
        w_c_rgb = self.fs_c_rgb(f_rgb)
        w_s_rgb = self.fs_s_rgb(f_rgb)
        f_rgb_sep = f_rgb * w_c_rgb * w_s_rgb

        w_c_nir = self.fs_c_nir(f_nir)
        w_s_nir = self.fs_s_nir(f_nir)
        f_nir_sep = f_nir * w_c_nir * w_s_nir

        # 2. Feature Aggregation (FA)
        g_nir_to_rgb = self.gate_nir_to_rgb(torch.cat([f_rgb, f_nir_sep], dim=1))
        g_rgb_to_nir = self.gate_rgb_to_nir(torch.cat([f_nir, f_rgb_sep], dim=1))

        # Bi-directional feature update
        f_rgb_out = f_rgb + g_nir_to_rgb * self.trans_nir(f_nir_sep)
        f_nir_out = f_nir + g_rgb_to_nir * self.trans_rgb(f_rgb_sep)

        return f_rgb_out, f_nir_out


class SAGate(nn.Module):
    """
    SA-Gate: Bi-Directional Cross-Modality Feature Propagation with Separation-and-Aggregation Gate.
    Adapted for 4-channel input: Channels 0,1,2 = RGB, Channel 3 = NIR.
    """
    def __init__(
        self,
        in_channels=4,
        num_classes=2,
        base_channels=32
    ):
        super().__init__()
        self.num_classes = num_classes
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # --- RGB Encoder Stream ---
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ConvBlock(c1, c1)
        )
        self.rgb_stage2 = ConvBlock(c1, c2, stride=2)
        self.rgb_stage3 = ConvBlock(c2, c3, stride=2)
        self.rgb_stage4 = ConvBlock(c3, c4, stride=2)

        # --- NIR Encoder Stream ---
        self.nir_stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ConvBlock(c1, c1)
        )
        self.nir_stage2 = ConvBlock(c1, c2, stride=2)
        self.nir_stage3 = ConvBlock(c2, c3, stride=2)
        self.nir_stage4 = ConvBlock(c3, c4, stride=2)

        # --- Separation-and-Aggregation Gates at Each Stage ---
        self.sagate1 = SeparationAggregationGate(c1)
        self.sagate2 = SeparationAggregationGate(c2)
        self.sagate3 = SeparationAggregationGate(c3)
        self.sagate4 = SeparationAggregationGate(c4)

        # --- Fused Feature Projections ---
        decoder_dim = 128
        self.proj4 = nn.Sequential(
            nn.Conv2d(c4 * 2, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(c3 * 2, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(c2 * 2, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(c1 * 2, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 4, H, W]
               x[:, :3] = RGB (B4, B3, B2)
               x[:, 3:4] = NIR (B8)
        Returns:
            logits: Tensor of shape [B, num_classes, H, W]
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        rgb = x[:, :3, :, :]
        nir = x[:, 3:4, :, :]

        # --- Stage 1 ---
        r1 = self.rgb_stem(rgb)
        n1 = self.nir_stem(nir)
        r1, n1 = self.sagate1(r1, n1)

        # --- Stage 2 ---
        r2 = self.rgb_stage2(r1)
        n2 = self.nir_stage2(n1)
        r2, n2 = self.sagate2(r2, n2)

        # --- Stage 3 ---
        r3 = self.rgb_stage3(r2)
        n3 = self.nir_stage3(n2)
        r3, n3 = self.sagate3(r3, n3)

        # --- Stage 4 ---
        r4 = self.rgb_stage4(r3)
        n4 = self.nir_stage4(n3)
        r4, n4 = self.sagate4(r4, n4)

        # --- Decoder ---
        p1 = self.proj1(torch.cat([r1, n1], dim=1))
        h1, w1 = p1.shape[2], p1.shape[3]

        p2 = F.interpolate(self.proj2(torch.cat([r2, n2], dim=1)), size=(h1, w1), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.proj3(torch.cat([r3, n3], dim=1)), size=(h1, w1), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.proj4(torch.cat([r4, n4], dim=1)), size=(h1, w1), mode='bilinear', align_corners=False)

        fused = self.fuse_all(torch.cat([p1, p2, p3, p4], dim=1))
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return logits
