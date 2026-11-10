"""
Swin UNETR: Swin Transformers for Semantic Segmentation (WACV 2022)
Paper: https://arxiv.org/abs/2201.01266
Official Repo: https://github.com/Project-MONAI/research-contributions

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Combines a hierarchical Swin Transformer encoder for extracting multi-scale contextual
representations with a convolutional residual (UnetResBlock) decoder for boundary
reconstruction and high-resolution mask generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """Local window multi-head self-attention with relative position bias."""
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(b_ // num_win, num_win, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class SwinBlock(nn.Module):
    """Swin Transformer block with W-MSA / SW-MSA and MLP."""
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, l, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, h, w, 1), device=x.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(b, h * w, c)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class UnetResBlock(nn.Module):
    """Residual Convolutional block for CNN decoder in Swin UNETR."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class UnetUpBlock(nn.Module):
    """Upsampling + concatenation with skip + UnetResBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = UnetResBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class SwinUNETR(nn.Module):
    """
    Swin UNETR: Swin Transformer encoder + multi-scale skip connections + CNN decoder.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        feature_size: int = 48,
        depths: list = [2, 2, 2, 2],
        num_heads: list = [3, 6, 12, 24],
        window_size: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.window_size = window_size

        # Scale 0: Stem at full resolution H, W (CNN skip)
        self.encoder0 = UnetResBlock(in_channels, feature_size)

        # Scale 1: 2x downsampling stem -> H/2, W/2
        self.stem1 = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.ModuleList([
            SwinBlock(feature_size, num_heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[0])
        ])

        # Scale 2: 2x downsample -> H/4, W/4, channels 2F
        self.down1_2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.ModuleList([
            SwinBlock(feature_size * 2, num_heads[1], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[1])
        ])

        # Scale 3: 2x downsample -> H/8, W/8, channels 4F
        self.down2_3 = nn.Sequential(
            nn.Conv2d(feature_size * 2, feature_size * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.ModuleList([
            SwinBlock(feature_size * 4, num_heads[2], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[2])
        ])

        # Scale 4: 2x downsample -> H/16, W/16, channels 8F (Bottleneck)
        self.down3_4 = nn.Sequential(
            nn.Conv2d(feature_size * 4, feature_size * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.ModuleList([
            SwinBlock(feature_size * 8, num_heads[3], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[3])
        ])

        # CNN Decoder
        self.decoder4 = UnetUpBlock(feature_size * 8, feature_size * 4, feature_size * 4)
        self.decoder3 = UnetUpBlock(feature_size * 4, feature_size * 2, feature_size * 2)
        self.decoder2 = UnetUpBlock(feature_size * 2, feature_size, feature_size)
        self.decoder1 = UnetUpBlock(feature_size, feature_size, feature_size)

        self.head = nn.Conv2d(feature_size, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # Scale 0 (CNN skip at full resolution)
        s0 = self.encoder0(x)  # [B, F, H, W]

        # Scale 1: H/2, W/2
        x1 = self.stem1(x)
        _, _, h1, w1 = x1.shape
        t1 = x1.flatten(2).transpose(1, 2)
        for blk in self.stage1:
            t1 = blk(t1, h1, w1)
        s1 = t1.transpose(1, 2).view(b, self.feature_size, h1, w1)

        # Scale 2: H/4, W/4
        x2 = self.down1_2(s1)
        _, _, h2, w2 = x2.shape
        t2 = x2.flatten(2).transpose(1, 2)
        for blk in self.stage2:
            t2 = blk(t2, h2, w2)
        s2 = t2.transpose(1, 2).view(b, self.feature_size * 2, h2, w2)

        # Scale 3: H/8, W/8
        x3 = self.down2_3(s2)
        _, _, h3, w3 = x3.shape
        t3 = x3.flatten(2).transpose(1, 2)
        for blk in self.stage3:
            t3 = blk(t3, h3, w3)
        s3 = t3.transpose(1, 2).view(b, self.feature_size * 4, h3, w3)

        # Scale 4: H/16, W/16 (Bottleneck)
        x4 = self.down3_4(s3)
        _, _, h4, w4 = x4.shape
        t4 = x4.flatten(2).transpose(1, 2)
        for blk in self.stage4:
            t4 = blk(t4, h4, w4)
        b4 = t4.transpose(1, 2).view(b, self.feature_size * 8, h4, w4)

        # CNN Decoder
        d3 = self.decoder4(b4, s3)  # [B, 4F, H/8, W/8]
        d2 = self.decoder3(d3, s2)  # [B, 2F, H/4, W/4]
        d1 = self.decoder2(d2, s1)  # [B, F, H/2, W/2]
        d0 = self.decoder1(d1, s0)  # [B, F, H, W]

        logits = self.head(d0)
        return logits
