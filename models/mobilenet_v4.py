"""
MobileNetV4: Five to Ten Times Faster (ECCV 2024)
Paper: https://arxiv.org/abs/2404.10518

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Features Universal Inverted Bottleneck (UIB) blocks unifying Inverted Bottleneck (IB),
ConvNeXt-style depthwise convolutions, Extra-Depthwise (Extra-DW) stages, and FFN blocks
for ultra-efficient low-latency spatial and channel representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalInvertedBottleneck(nn.Module):
    """
    Universal Inverted Bottleneck (UIB) block:
    Unifies Inverted Bottlenecks (IB), ConvNeXt blocks, Extra-DW, and FFNs.
    Structure:
      [Optional Start-DW Conv] -> Expand 1x1 Conv -> [Optional Middle-DW Conv] -> Project 1x1 Conv.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: float = 4.0,
        start_dw_kernel: int = 0,
        middle_dw_kernel: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []

        # 1. Start Depthwise Conv (if kernel > 0)
        if start_dw_kernel > 0:
            layers.extend([
                nn.Conv2d(
                    in_ch,
                    in_ch,
                    kernel_size=start_dw_kernel,
                    stride=stride,
                    padding=start_dw_kernel // 2,
                    groups=in_ch,
                    bias=False,
                ),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
            ])
            mid_stride = 1
        else:
            mid_stride = stride

        # 2. 1x1 Pointwise Expansion Conv
        hidden_ch = int(round(in_ch * expand_ratio))
        layers.extend([
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        ])

        # 3. Middle Depthwise Conv (if kernel > 0)
        if middle_dw_kernel > 0:
            layers.extend([
                nn.Conv2d(
                    hidden_ch,
                    hidden_ch,
                    kernel_size=middle_dw_kernel,
                    stride=mid_stride,
                    padding=middle_dw_kernel // 2,
                    groups=hidden_ch,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
            ])

        # 4. 1x1 Pointwise Projection Conv
        layers.extend([
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res:
            return x + self.block(x)
        return self.block(x)


class MobileNetV4(nn.Module):
    """
    MobileNetV4 semantic segmentation network for remote sensing imagery.
    Employs Universal Inverted Bottleneck (UIB) with Extra-DW blocks across multi-scale
    encoder and decoder paths.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        channels: list = [32, 64, 128, 256],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem: Conv 3x3 with stride 2 -> [B, C0, H/2, W/2]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1 (Extra-DW UIB): H/2, W/2, channels[0]
        self.stage1 = nn.Sequential(
            UniversalInvertedBottleneck(channels[0], channels[0], expand_ratio=2.0, start_dw_kernel=3, middle_dw_kernel=3),
            UniversalInvertedBottleneck(channels[0], channels[0], expand_ratio=2.0, start_dw_kernel=3, middle_dw_kernel=3),
        )

        # Stage 2 (Extra-DW UIB with stride 2): H/4, W/4, channels[1]
        self.stage2 = nn.Sequential(
            UniversalInvertedBottleneck(channels[0], channels[1], expand_ratio=4.0, start_dw_kernel=3, middle_dw_kernel=3, stride=2),
            UniversalInvertedBottleneck(channels[1], channels[1], expand_ratio=4.0, start_dw_kernel=3, middle_dw_kernel=3),
        )

        # Stage 3 (Extra-DW UIB with stride 2): H/8, W/8, channels[2]
        self.stage3 = nn.Sequential(
            UniversalInvertedBottleneck(channels[1], channels[2], expand_ratio=4.0, start_dw_kernel=3, middle_dw_kernel=3, stride=2),
            UniversalInvertedBottleneck(channels[2], channels[2], expand_ratio=4.0, start_dw_kernel=3, middle_dw_kernel=3),
        )

        # Bottleneck (Extra-DW UIB with stride 2): H/16, W/16, channels[3]
        self.bottleneck = nn.Sequential(
            UniversalInvertedBottleneck(channels[2], channels[3], expand_ratio=4.0, start_dw_kernel=3, middle_dw_kernel=3, stride=2),
            UniversalInvertedBottleneck(channels[3], channels[3], expand_ratio=4.0, start_dw_kernel=3, middle_dw_kernel=3),
        )

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            UniversalInvertedBottleneck(channels[2] * 2, channels[2], expand_ratio=2.0, start_dw_kernel=3, middle_dw_kernel=3),
        )

        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            UniversalInvertedBottleneck(channels[1] * 2, channels[1], expand_ratio=2.0, start_dw_kernel=3, middle_dw_kernel=3),
        )

        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            UniversalInvertedBottleneck(channels[0] * 2, channels[0], expand_ratio=2.0, start_dw_kernel=3, middle_dw_kernel=3),
        )

        # Final 2x upsampling to full resolution
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x_stem = self.stem(x)          # [B, 32, H/2, W/2]
        x1 = self.stage1(x_stem)       # [B, 32, H/2, W/2]
        x2 = self.stage2(x1)           # [B, 64, H/4, W/4]
        x3 = self.stage3(x2)           # [B, 128, H/8, W/8]
        b = self.bottleneck(x3)        # [B, 256, H/16, W/16]

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([x3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([x2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([x1, d1], dim=1))

        d0 = self.up0(d1)
        logits = self.head(d0)
        return logits
