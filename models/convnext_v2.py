"""
ConvNeXt-V2: Co-designing and Scaling ConvNets with Masked Autoencoders (CVPR 2023)
Paper: https://arxiv.org/abs/2301.00808
Official Repo: https://github.com/facebookresearch/ConvNeXt-V2

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Features 7x7 depthwise convolutions, Global Response Normalization (GRN) to enhance
channel contrast and eliminate feature collapse, inverted bottleneck MLP layers,
and multi-scale skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """
    Global Response Normalization (GRN) layer:
    Calculates L2-norm across spatial dimensions and normalizes channel responses
    with learnable affine parameters gamma and beta.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, H, W, C]
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt-V2 Block:
    7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv (expand 4x) -> GELU -> GRN -> 1x1 Conv (contract) -> Residual.
    """
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """
    ConvNeXt-V2 semantic segmentation network for remote sensing imagery.
    Integrates ConvNeXt-V2 blocks with Global Response Normalization (GRN) across
    multi-scale encoder and decoder stages.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        depths: list = [2, 2, 2, 2],
        dims: list = [32, 64, 128, 256],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem at full resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
            ConvNeXtV2Block(dims[0]),
        )

        # Stage 1: down to H/2, W/2
        self.down1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
        )
        self.stage1 = nn.Sequential(*[ConvNeXtV2Block(dims[1]) for _ in range(depths[0])])

        # Stage 2: down to H/4, W/4
        self.down2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[2]),
        )
        self.stage2 = nn.Sequential(*[ConvNeXtV2Block(dims[2]) for _ in range(depths[1])])

        # Stage 3: down to H/8, W/8
        self.down3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[3]),
        )
        self.stage3 = nn.Sequential(*[ConvNeXtV2Block(dims[3]) for _ in range(depths[2])])

        # Bottleneck: down to H/16, W/16
        self.down4 = nn.Sequential(
            nn.Conv2d(dims[3], dims[3] * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[3] * 2),
        )
        self.bottleneck = nn.Sequential(*[ConvNeXtV2Block(dims[3] * 2) for _ in range(depths[3])])

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(dims[3] * 2, dims[3], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(dims[3] * 2, dims[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[3]),
            ConvNeXtV2Block(dims[3]),
        )

        self.up2 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[2]),
            ConvNeXtV2Block(dims[2]),
        )

        self.up1 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            ConvNeXtV2Block(dims[1]),
        )

        self.up0 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            ConvNeXtV2Block(dims[0]),
        )

        self.head = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.stem(x)              # [B, 32, 256, 256]
        x1 = self.stage1(self.down1(x0))  # [B, 64, 128, 128]
        x2 = self.stage2(self.down2(x1))  # [B, 128, 64, 64]
        x3 = self.stage3(self.down3(x2))  # [B, 256, 32, 32]
        b = self.bottleneck(self.down4(x3))  # [B, 512, 16, 16]

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([x3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([x2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([x1, d1], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([x0, d0], dim=1))

        logits = self.head(d0)
        return logits
