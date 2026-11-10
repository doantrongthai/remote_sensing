"""
TransUNet: Transformers Make Strong Encoders for Medical and RS Segmentation (arXiv 2021)
Paper: https://arxiv.org/abs/2102.04306
Official Repo: https://github.com/Beckschen/TransUNet

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Features a hybrid CNN (ResNet) encoder for detailed low-level feature extraction,
a Vision Transformer (ViT) bottleneck for global contextual modeling, and a
Cascaded Upsampler (CUP) decoder with multi-scale skip connections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    """Residual block with two 3x3 convolutions and projection shortcut."""
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


class ViTBlock(nn.Module):
    """Transformer Encoder Block with Pre-LN, MHSA, and MLP."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """
    Cascaded Upsampler (CUP) Decoder Block:
    Upsample -> Concat Skip -> Conv-BN-ReLU -> Conv-BN-ReLU.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    """
    TransUNet: Hybrid ResNet + ViT + Cascaded Upsampler for Semantic Segmentation.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        resnet_channels: list = [64, 128, 256, 512],
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_dim: int = 2048,
        grid_size: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        # 1. ResNet CNN Backbone
        # Initial stem: 7x7 conv with stride 2 -> [B, 64, H/2, W/2]
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, resnet_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(resnet_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> [B, 64, H/4, W/4]

        # Stage 1: [B, 128, H/4, W/4]
        self.layer1 = nn.Sequential(
            ResNetBlock(resnet_channels[0], resnet_channels[1], stride=1),
            ResNetBlock(resnet_channels[1], resnet_channels[1], stride=1),
        )

        # Stage 2: [B, 256, H/8, W/8]
        self.layer2 = nn.Sequential(
            ResNetBlock(resnet_channels[1], resnet_channels[2], stride=2),
            ResNetBlock(resnet_channels[2], resnet_channels[2], stride=1),
        )

        # Stage 3: [B, 512, H/16, W/16]
        self.layer3 = nn.Sequential(
            ResNetBlock(resnet_channels[2], resnet_channels[3], stride=2),
            ResNetBlock(resnet_channels[3], resnet_channels[3], stride=1),
        )

        # 2. Vision Transformer Bottleneck
        self.proj_in = nn.Conv2d(resnet_channels[3], embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.vit_blocks = nn.ModuleList([
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
            for _ in range(num_layers)
        ])
        self.norm_vit = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Sequential(
            nn.Conv2d(embed_dim, resnet_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(resnet_channels[3]),
            nn.ReLU(inplace=True),
        )

        # 3. Cascaded Upsampler (CUP) Decoder
        # Dec 3: from H/16 to H/8, concat stage 2 (256 channels)
        self.dec3 = DecoderBlock(resnet_channels[3], resnet_channels[2], 256)

        # Dec 2: from H/8 to H/4, concat stage 1 (128 channels)
        self.dec2 = DecoderBlock(256, resnet_channels[1], 128)

        # Dec 1: from H/4 to H/2, concat stem (64 channels)
        self.dec1 = DecoderBlock(128, resnet_channels[0], 64)

        # Dec 0: from H/2 to H (full resolution)
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Final prediction head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # CNN Feature Extraction
        skip0 = self.stem_conv(x)      # [B, 64, H/2, W/2]
        p = self.stem_pool(skip0)      # [B, 64, H/4, W/4]
        skip1 = self.layer1(p)         # [B, 128, H/4, W/4]
        skip2 = self.layer2(skip1)     # [B, 256, H/8, W/8]
        feat3 = self.layer3(skip2)     # [B, 512, H/16, W/16]

        # ViT Bottleneck
        tokens = self.proj_in(feat3)   # [B, embed_dim, H/16, W/16]
        _, _, th, tw = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        # Add position embeddings (interpolated if input resolution varies)
        if tokens.shape[1] == self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed
        else:
            pos = self.pos_embed.permute(0, 2, 1).view(1, self.embed_dim, self.grid_size, self.grid_size)
            pos = F.interpolate(pos, size=(th, tw), mode="bilinear", align_corners=False)
            tokens = tokens + pos.flatten(2).transpose(1, 2)

        for blk in self.vit_blocks:
            tokens = blk(tokens)
        tokens = self.norm_vit(tokens)

        # Reshape back to 2D spatial map
        vit_feat = tokens.transpose(1, 2).view(b, self.embed_dim, th, tw)
        bottleneck = self.proj_out(vit_feat)  # [B, 512, H/16, W/16]

        # Cascaded Upsampler Decoder
        d3 = self.dec3(bottleneck, skip2)  # [B, 256, H/8, W/8]
        d2 = self.dec2(d3, skip1)          # [B, 128, H/4, W/4]
        d1 = self.dec1(d2, skip0)          # [B, 64, H/2, W/2]
        d0 = self.dec0(d1)                 # [B, 32, H, W]

        logits = self.head(d0)
        return logits
