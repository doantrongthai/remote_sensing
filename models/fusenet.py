"""
FuseNet: Incorporating Depth into Semantic Segmentation (ACCV 2016)
Adapted for Sentinel-2 Multimodal Remote Sensing (RGB + NIR).

Original Paper: "FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-Based CNN Architecture"
Reference: https://github.com/MehmetAygun/fusenet-pytorch, https://github.com/mshakeri/FuseNet-PyTorch

Key Components:
- Dual-branch encoder: RGB branch and NIR branch
- Element-wise additive fusion: NIR feature maps are injected additively into the RGB branch
  at each encoder stage
- Symmetric decoder with transposed convolutions and multi-scale skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two sequential 3x3 convolutions with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class FuseNet(nn.Module):
    """
    FuseNet: Dual-branch additive fusion network (ACCV 2016).
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
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # --- RGB Encoder Stream ---
        self.rgb_conv1 = DoubleConv(3, c1)
        self.rgb_conv2 = DoubleConv(c1, c2)
        self.rgb_conv3 = DoubleConv(c2, c3)
        self.rgb_conv4 = DoubleConv(c3, c4)
        self.rgb_conv5 = DoubleConv(c4, c5)

        # --- NIR Encoder Stream ---
        self.nir_conv1 = DoubleConv(1, c1)
        self.nir_conv2 = DoubleConv(c1, c2)
        self.nir_conv3 = DoubleConv(c2, c3)
        self.nir_conv4 = DoubleConv(c3, c4)
        self.nir_conv5 = DoubleConv(c4, c5)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Decoder Stream ---
        self.up5 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec_conv4 = DoubleConv(c4 * 2, c4)

        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec_conv3 = DoubleConv(c3 * 2, c3)

        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec_conv2 = DoubleConv(c2 * 2, c2)

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec_conv1 = DoubleConv(c1 * 2, c1)

        self.classifier = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(c1, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, 4, H, W]
               x[:, :3] = RGB (B4, B3, B2)
               x[:, 3:4] = NIR (B8)
        Returns:
            logits: Tensor of shape [B, num_classes, H, W]
        """
        rgb = x[:, :3, :, :]
        nir = x[:, 3:4, :, :]

        # --- Stage 1 ---
        r1 = self.rgb_conv1(rgb)
        n1 = self.nir_conv1(nir)
        r1_fused = r1 + n1  # Additive fusion
        r1_pool = self.pool(r1_fused)
        n1_pool = self.pool(n1)

        # --- Stage 2 ---
        r2 = self.rgb_conv2(r1_pool)
        n2 = self.nir_conv2(n1_pool)
        r2_fused = r2 + n2  # Additive fusion
        r2_pool = self.pool(r2_fused)
        n2_pool = self.pool(n2)

        # --- Stage 3 ---
        r3 = self.rgb_conv3(r2_pool)
        n3 = self.nir_conv3(n2_pool)
        r3_fused = r3 + n3  # Additive fusion
        r3_pool = self.pool(r3_fused)
        n3_pool = self.pool(n3)

        # --- Stage 4 ---
        r4 = self.rgb_conv4(r3_pool)
        n4 = self.nir_conv4(n3_pool)
        r4_fused = r4 + n4  # Additive fusion
        r4_pool = self.pool(r4_fused)
        n4_pool = self.pool(n4)

        # --- Stage 5 ---
        r5 = self.rgb_conv5(r4_pool)
        n5 = self.nir_conv5(n4_pool)
        r5_fused = r5 + n5  # Additive fusion

        # --- Decoder ---
        d4 = self.up5(r5_fused)
        d4 = self.dec_conv4(torch.cat([r4_fused, d4], dim=1))

        d3 = self.up4(d4)
        d3 = self.dec_conv3(torch.cat([r3_fused, d3], dim=1))

        d2 = self.up3(d3)
        d2 = self.dec_conv2(torch.cat([r2_fused, d2], dim=1))

        d1 = self.up2(d2)
        d1 = self.dec_conv1(torch.cat([r1_fused, d1], dim=1))

        logits = self.classifier(d1)
        return logits
