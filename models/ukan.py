"""
U-KAN: U-KAN Makes Strong Backbone for Medical and RS Segmentation (AAAI 2025)
Paper: https://arxiv.org/abs/2406.02918
Official Repo: https://github.com/CUHK-AIM-Group/U-KAN

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Features Tokenized KAN blocks with learnable B-spline activation functions and depthwise
convolutions for spatial-frequency feature learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear layer parameterized with learnable
    B-spline activation functions alongside a linear base projection.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize uniform grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Base linear weight & spline coefficients
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))
        self.base_activation = base_activation()
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order) - 0.5) * 0.1
            self.spline_weight.copy_(noise * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Cox-de Boor B-splines of order `spline_order`.
        Input: [..., in_features]
        Output: [..., in_features, grid_size + spline_order]
        """
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[..., :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[..., 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear branch
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Learnable B-spline branch
        bases = self.b_splines(x)
        spline_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline_output = torch.einsum("...ik, oik -> ...o", bases, spline_weight)

        return base_output + spline_output


class KANBlock(nn.Module):
    """
    Tokenized KAN Block:
    Combines depthwise convolution for local spatial mixing with KANLinear layers
    for adaptive non-linear channel transformations.
    """
    def __init__(self, dim: int, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.kan1 = KANLinear(dim, dim * 2)
        self.kan2 = KANLinear(dim * 2, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual branch 1: Depthwise spatial conv
        res = x
        x = self.dwconv(self.norm1(x))
        x = x + res

        # Residual branch 2: Tokenized KAN channel transformation
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_norm = self.norm2(x_flat)
        x_kan = self.kan1(x_norm)
        x_kan = self.kan2(x_kan)
        x_kan = self.dropout(x_kan)
        x_kan = x_kan.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return x + x_kan


class ConvBlock(nn.Module):
    """Two 3x3 Conv-BN-ReLU layers for shallow spatial representations."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UKAN(nn.Module):
    """
    U-KAN: Kolmogorov-Arnold U-Net architecture for semantic segmentation.
    Integrates learnable B-spline KAN blocks into encoder stages, bottleneck, and
    decoder skip paths.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        channels: list = [32, 64, 128, 256, 512],
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem & Stage 0: Standard convolutional double conv
        self.inc = ConvBlock(in_channels, channels[0])

        # Encoder stages: Pooling + KANBlock
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[0], channels[1]),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[1], channels[2]),
            KANBlock(channels[2], drop=drop_rate),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[2], channels[3]),
            KANBlock(channels[3], drop=drop_rate),
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[3], channels[4]),
            KANBlock(channels[4], drop=drop_rate),
        )

        # Decoder stages with Upsample + Skip Concatenation + KAN / Conv
        self.up3 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.dec_kan3 = nn.Sequential(
            ConvBlock(channels[3] * 2, channels[3]),
            KANBlock(channels[3], drop=drop_rate),
        )

        self.up2 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec_kan2 = nn.Sequential(
            ConvBlock(channels[2] * 2, channels[2]),
            KANBlock(channels[2], drop=drop_rate),
        )

        self.up1 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec_conv1 = ConvBlock(channels[1] * 2, channels[1])

        self.up0 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec_conv0 = ConvBlock(channels[0] * 2, channels[0])

        # Final segmentation head
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.inc(x)             # [B, C0, H, W]
        x1 = self.down1(x0)          # [B, C1, H/2, W/2]
        x2 = self.down2(x1)          # [B, C2, H/4, W/4]
        x3 = self.down3(x2)          # [B, C3, H/8, W/8]
        b = self.bottleneck(x3)      # [B, C4, H/16, W/16]

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec_kan3(torch.cat([x3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec_kan2(torch.cat([x2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec_conv1(torch.cat([x1, d1], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec_conv0(torch.cat([x0, d0], dim=1))

        logits = self.classifier(d0)
        return logits
