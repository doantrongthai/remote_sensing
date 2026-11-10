"""
RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation (IEEE RAL 2019)
Adapted for Sentinel-2 Multimodal Remote Sensing (RGB + NIR).

Original Paper: "RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes"
Reference: https://github.com/yuxiangsun/RTFNet

Key Components:
- Dual-stream ResNet encoder (RGB stream + NIR stream)
- Multi-level feature fusion via element-wise addition across encoder stages
- Up-fusion decoder composed of Upception blocks (UpceptionA for feature refinement,
  UpceptionB for transposed-convolution upsampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Standard ResNet basic residual block.
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


class UpceptionA(nn.Module):
    """
    Upception Block A: Preserves spatial resolution while refining features.
    Inception-style dual branches with residual shortcut.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch1(x) + self.branch2(x))


class UpceptionB(nn.Module):
    """
    Upception Block B: Upsamples spatial resolution by 2x using transposed convolutions.
    Multi-pathway architecture inspired by Inception.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2

        # Path 1: Conv 1x1 -> Conv 3x3 -> ConvTranspose2d 2x2
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Path 2: Direct ConvTranspose2d 2x2
        self.path2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut: ConvTranspose2d 2x2
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.path1(x) + self.path2(x) + self.shortcut(x))


class RTFNet(nn.Module):
    """
    RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation (IEEE RAL 2019).
    Adapted for 4-channel input: Channels 0,1,2 = RGB, Channel 3 = NIR.
    Dual-stream ResNet encoder + Upception-based Up-fusion decoder.
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
        self.rgb_conv1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.rgb_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rgb_layer1 = nn.Sequential(ResBlock(c1, c1), ResBlock(c1, c1))
        self.rgb_layer2 = nn.Sequential(ResBlock(c1, c2, stride=2), ResBlock(c2, c2))
        self.rgb_layer3 = nn.Sequential(ResBlock(c2, c3, stride=2), ResBlock(c3, c3))
        self.rgb_layer4 = nn.Sequential(ResBlock(c3, c4, stride=2), ResBlock(c4, c4))

        # --- NIR Encoder Stream ---
        self.nir_conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.nir_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.nir_layer1 = nn.Sequential(ResBlock(c1, c1), ResBlock(c1, c1))
        self.nir_layer2 = nn.Sequential(ResBlock(c1, c2, stride=2), ResBlock(c2, c2))
        self.nir_layer3 = nn.Sequential(ResBlock(c2, c3, stride=2), ResBlock(c3, c3))
        self.nir_layer4 = nn.Sequential(ResBlock(c3, c4, stride=2), ResBlock(c4, c4))

        # --- Up-fusion Decoder (Upception Stages) ---
        # Level 4 to 3: c4 -> c3 (upsample 2x)
        self.up4 = UpceptionB(c4, c3)
        self.refine3 = UpceptionA(c3, c3)

        # Level 3 to 2: c3 -> c2 (upsample 2x)
        self.up3 = UpceptionB(c3, c2)
        self.refine2 = UpceptionA(c2, c2)

        # Level 2 to 1: c2 -> c1 (upsample 2x)
        self.up2 = UpceptionB(c2, c1)
        self.refine1 = UpceptionA(c1, c1)

        # Level 1 to 0: c1 -> c1 (upsample 2x)
        self.up1 = UpceptionB(c1, c1)
        self.refine0 = UpceptionA(c1, c1)

        # Final 2x upsample to original resolution
        self.up0 = UpceptionB(c1, c1)
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
        orig_h, orig_w = x.shape[2], x.shape[3]

        rgb = x[:, :3, :, :]
        nir = x[:, 3:4, :, :]

        # --- RGB Encoder Forward ---
        r0 = self.rgb_conv1(rgb)       # 1/2
        r_pool = self.rgb_maxpool(r0)  # 1/4
        r1 = self.rgb_layer1(r_pool)   # 1/4
        r2 = self.rgb_layer2(r1)       # 1/8
        r3 = self.rgb_layer3(r2)       # 1/16
        r4 = self.rgb_layer4(r3)       # 1/32

        # --- NIR Encoder Forward ---
        n0 = self.nir_conv1(nir)       # 1/2
        n_pool = self.nir_maxpool(n0)  # 1/4
        n1 = self.nir_layer1(n_pool)   # 1/4
        n2 = self.nir_layer2(n1)       # 1/8
        n3 = self.nir_layer3(n2)       # 1/16
        n4 = self.nir_layer4(n3)       # 1/32

        # --- Multi-level Additive Fusion ---
        f0 = r0 + n0                   # 1/2, c1
        f1 = r1 + n1                   # 1/4, c1
        f2 = r2 + n2                   # 1/8, c2
        f3 = r3 + n3                   # 1/16, c3
        f4 = r4 + n4                   # 1/32, c4

        # --- Up-fusion Decoder Forward ---
        d3 = self.up4(f4)              # 1/16, c3
        d3 = self.refine3(d3 + f3)

        d2 = self.up3(d3)              # 1/8, c2
        d2 = self.refine2(d2 + f2)

        d1 = self.up2(d2)              # 1/4, c1
        d1 = self.refine1(d1 + f1)

        d0 = self.up1(d1)              # 1/2, c1
        d0 = self.refine0(d0 + f0)

        out = self.up0(d0)             # 1/1, c1
        logits = self.classifier(out)

        if logits.shape[2:] != (orig_h, orig_w):
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return logits
