"""
EGNet: Edge Guidance Network for Salient Object and RS Segmentation (ICCV 2019)
Paper: https://arxiv.org/abs/1908.08297
Official Repo: https://github.com/JXingZhao/EGNet

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Features dual-stream architecture: a dedicated Salient Edge Guidance stream that models
high-frequency water-land coastlines and levee boundaries, coupled with a progressive
salient region segmentation decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    """Residual convolutional block with two 3x3 convs and identity/projection shortcut."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class EdgeGuidanceModule(nn.Module):
    """
    Edge Guidance Module (EGM):
    Fuses multi-scale edge information by combining higher-level semantic edge context
    with lower-level fine-grained spatial boundaries.
    """
    def __init__(self, high_ch: int, low_ch: int, out_ch: int):
        super().__init__()
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        h_up = F.interpolate(self.conv_high(high_feat), size=low_feat.shape[-2:], mode="bilinear", align_corners=False)
        l_conv = self.conv_low(low_feat)
        return self.fuse(torch.cat([h_up, l_conv], dim=1))


class EdgeRegionFusion(nn.Module):
    """
    Fuses the explicit salient edge feature map into the region segmentation decoder stream.
    """
    def __init__(self, region_ch: int, edge_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(region_ch + edge_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, region_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        if region_feat.shape[-2:] != edge_feat.shape[-2:]:
            edge_feat = F.interpolate(edge_feat, size=region_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([region_feat, edge_feat], dim=1))


class EGNet(nn.Module):
    """
    EGNet: Edge Guidance Network with salient edge stream and region stream.
    Outputs binary flood segmentation logits [B, 2, H, W].
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        channels: list = [32, 64, 128, 256, 512],
        edge_ch: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder stages
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )  # [B, 32, H, W]

        self.stage1 = ResConvBlock(channels[0], channels[1], stride=2)  # [B, 64, H/2, W/2]
        self.stage2 = ResConvBlock(channels[1], channels[2], stride=2)  # [B, 128, H/4, W/4]
        self.stage3 = ResConvBlock(channels[2], channels[3], stride=2)  # [B, 256, H/8, W/8]
        self.stage4 = ResConvBlock(channels[3], channels[4], stride=2)  # [B, 512, H/16, W/16]

        # Salient Edge Stream (EGMs)
        self.edge_proj4 = nn.Sequential(
            nn.Conv2d(channels[4], edge_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(edge_ch),
            nn.ReLU(inplace=True),
        )
        self.egm3 = EdgeGuidanceModule(edge_ch, channels[3], edge_ch)
        self.egm2 = EdgeGuidanceModule(edge_ch, channels[2], edge_ch)
        self.egm1 = EdgeGuidanceModule(edge_ch, channels[1], edge_ch)
        self.egm0 = EdgeGuidanceModule(edge_ch, channels[0], edge_ch)

        # Auxiliary edge supervision head
        self.edge_head = nn.Conv2d(edge_ch, 1, kernel_size=1)

        # Region Decoder Stream with Edge Guidance Fusion
        self.up4_3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse_reg3 = EdgeRegionFusion(channels[4] + channels[3], edge_ch, channels[3])

        self.up3_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse_reg2 = EdgeRegionFusion(channels[3] + channels[2], edge_ch, channels[2])

        self.up2_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse_reg1 = EdgeRegionFusion(channels[2] + channels[1], edge_ch, channels[1])

        self.up1_0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse_reg0 = EdgeRegionFusion(channels[1] + channels[0], edge_ch, channels[0])

        # Final segmentation head
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_edge: bool = False):
        orig_size = x.shape[-2:]

        # 1. Multi-scale feature extraction
        f0 = self.stem(x)        # [B, 32, H, W]
        f1 = self.stage1(f0)     # [B, 64, H/2, W/2]
        f2 = self.stage2(f1)     # [B, 128, H/4, W/4]
        f3 = self.stage3(f2)     # [B, 256, H/8, W/8]
        f4 = self.stage4(f3)     # [B, 512, H/16, W/16]

        # 2. Edge Stream: progressive edge guidance
        e4 = self.edge_proj4(f4)
        e3 = self.egm3(e4, f3)   # at H/8
        e2 = self.egm2(e3, f2)   # at H/4
        e1 = self.egm1(e2, f1)   # at H/2
        e0 = self.egm0(e1, f0)   # at H

        # 3. Salient Region Stream guided by edge features
        # Stage 4 to 3
        r3 = self.up4_3(f4)
        r3 = self.fuse_reg3(torch.cat([r3, f3], dim=1), e3)

        # Stage 3 to 2
        r2 = self.up3_2(r3)
        r2 = self.fuse_reg2(torch.cat([r2, f2], dim=1), e2)

        # Stage 2 to 1
        r1 = self.up2_1(r2)
        r1 = self.fuse_reg1(torch.cat([r1, f1], dim=1), e1)

        # Stage 1 to 0
        r0 = self.up1_0(r1)
        r0 = self.fuse_reg0(torch.cat([r0, f0], dim=1), e0)

        # 4. Final segmentation logits
        seg_logits = self.seg_head(r0)

        if return_edge:
            edge_logits = self.edge_head(e0)
            return seg_logits, edge_logits

        return seg_logits
