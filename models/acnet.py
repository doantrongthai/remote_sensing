"""
ACNet: Attention Based Network to Exploit Complementary Features (IEEE TIP 2019)
Adapted for Sentinel-2 Multimodal Remote Sensing (RGB + NIR).

Original Paper: "ACNet: Attention Based Network to Exploit Complementary Features for RGBD Semantic Segmentation"
Reference: https://github.com/anheidelonghu/ACNet, https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch

Key Components:
- Three-branch architecture: RGB branch, NIR branch, and Fused branch
- Attention Complementary Module (ACM): Computes channel-wise attention weights
  to selectively combine complementary features from unimodal branches into the fused branch
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


class AttentionComplementaryModule(nn.Module):
    """
    Attention Complementary Module (ACM).
    Computes cross-modal channel attention weights by aggregating global spatial contexts
    from both modalities and applying element-wise channel reweighting.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels * 2, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels * 2, bias=False),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, nir_feat):
        b, c, h, w = rgb_feat.shape

        # Global average pooling descriptors
        rgb_gap = self.gap(rgb_feat).view(b, c)
        nir_gap = self.gap(nir_feat).view(b, c)

        # Concatenate and compute cross-modal channel weights
        combined = torch.cat([rgb_gap, nir_gap], dim=1)  # [B, 2*C]
        weights = self.fc(combined).view(b, 2, c, 1, 1)  # [B, 2, C, 1, 1]

        w_rgb = weights[:, 0, :, :, :]
        w_nir = weights[:, 1, :, :, :]

        # Attention-weighted complementary fusion
        fused = w_rgb * rgb_feat + w_nir * nir_feat
        return self.fusion_conv(fused)


class ACNet(nn.Module):
    """
    ACNet: Attention Based Network to Exploit Complementary Features.
    Adapted for 4-channel input: Channels 0,1,2 = RGB, Channel 3 = NIR.
    Maintains three distinct parallel branches: RGB, NIR, and Fused.
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

        # --- 1. RGB Branch ---
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ConvBlock(c1, c1)
        )
        self.rgb_stage2 = ConvBlock(c1, c2, stride=2)
        self.rgb_stage3 = ConvBlock(c2, c3, stride=2)
        self.rgb_stage4 = ConvBlock(c3, c4, stride=2)

        # --- 2. NIR Branch ---
        self.nir_stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ConvBlock(c1, c1)
        )
        self.nir_stage2 = ConvBlock(c1, c2, stride=2)
        self.nir_stage3 = ConvBlock(c2, c3, stride=2)
        self.nir_stage4 = ConvBlock(c3, c4, stride=2)

        # --- Attention Complementary Modules (ACM) ---
        self.acm1 = AttentionComplementaryModule(c1)
        self.acm2 = AttentionComplementaryModule(c2)
        self.acm3 = AttentionComplementaryModule(c3)
        self.acm4 = AttentionComplementaryModule(c4)

        # --- 3. Fused Branch ---
        self.fuse_stage2 = ConvBlock(c1, c2, stride=2)
        self.fuse_stage3 = ConvBlock(c2, c3, stride=2)
        self.fuse_stage4 = ConvBlock(c3, c4, stride=2)

        # --- Multi-scale Decoder ---
        decoder_dim = 128
        self.proj4 = nn.Sequential(
            nn.Conv2d(c4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(c3, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(c2, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(c1, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        self.fuse_head = nn.Sequential(
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
        f1 = self.acm1(r1, n1)

        # --- Stage 2 ---
        r2 = self.rgb_stage2(r1)
        n2 = self.nir_stage2(n1)
        comp2 = self.acm2(r2, n2)
        f2 = self.fuse_stage2(f1) + comp2

        # --- Stage 3 ---
        r3 = self.rgb_stage3(r2)
        n3 = self.nir_stage3(n2)
        comp3 = self.acm3(r3, n3)
        f3 = self.fuse_stage3(f2) + comp3

        # --- Stage 4 ---
        r4 = self.rgb_stage4(r3)
        n4 = self.nir_stage4(n3)
        comp4 = self.acm4(r4, n4)
        f4 = self.fuse_stage4(f3) + comp4

        # --- Decoder on Fused Branch ---
        p1 = self.proj1(f1)
        h1, w1 = p1.shape[2], p1.shape[3]

        p2 = F.interpolate(self.proj2(f2), size=(h1, w1), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.proj3(f3), size=(h1, w1), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.proj4(f4), size=(h1, w1), mode='bilinear', align_corners=False)

        fused = self.fuse_head(torch.cat([p1, p2, p3, p4], dim=1))
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return logits
