"""
Swin-Unet: Unet-like Pure Transformer for Semantic Segmentation (ECCV-W 2022)
Paper: https://arxiv.org/abs/2105.05537
Official Repo: https://github.com/HuCaoFighting/Swin-Unet

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Implements shifted window multi-head self-attention (W-MSA & SW-MSA) encoder stages,
patch merging downsamplers, and symmetric patch expanding decoders with skip connections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partitions [B, H, W, C] feature maps into non-overlapping [num_windows * B, window_size, window_size, C]
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """
    Reverses window partitions back to [B, H, W, C]
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA) with relative position bias.
    Supports regular and shifted window self-attention.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Coordinate grid for relative position indices
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
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


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block: alternating W-MSA and SW-MSA with LayerNorm and MLP.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, l, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Cyclic shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # Calculate attention mask
            img_mask = torch.zeros((1, h, w, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
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

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(b, h * w, c)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer: downsamples spatial resolution by 2x and doubles channel dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
        b, l, c = x.shape
        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(b, -1, 4 * c)

        x = self.norm(x)
        x = self.reduction(x)
        return x, h // 2, w // 2


class PatchExpanding(nn.Module):
    """
    Patch Expanding Layer: upsamples spatial resolution by 2x and halves channel dimension.
    Uses linear expansion to 2C, then 2x2 spatial rearrangement.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
        b, l, c = x.shape
        x = self.expand(x)  # [B, H*W, 2C]
        x = x.view(b, h, w, 2 * c)

        # 2x2 spatial expansion: 2C -> C//2 channels at 2H, 2W
        c_out = c // 2
        x = x.view(b, h, w, 2, 2, c_out).permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h * 2, w * 2, c_out)
        x = x.view(b, -1, c_out)
        x = self.norm(x)
        return x, h * 2, w * 2


class FinalPatchExpanding(nn.Module):
    """
    Final Patch Expanding Layer: 4x spatial upsampling to restore full image resolution.
    """
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
        b, l, c = x.shape
        x = self.expand(x)
        out_c = x.shape[-1] // 16
        x = x.view(b, h, w, 4, 4, out_c).permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h * 4, w * 4, out_c)
        x = self.norm(x)
        return x, h * 4, w * 4


class SwinUnet(nn.Module):
    """
    Swin-Unet: Unet-like Pure Transformer for Semantic Segmentation.
    Encoder: Swin Transformer blocks with Patch Merging.
    Decoder: Swin Transformer blocks with Patch Expanding and skip connections.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        embed_dim: int = 48,
        depths: list = [2, 2, 2, 2],
        num_heads: list = [3, 6, 12, 24],
        window_size: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.window_size = window_size

        # Patch partition & linear projection: 4x downsampling stem
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4),
        )
        self.norm_stem = nn.LayerNorm(embed_dim)

        # Encoder stages
        dims = [embed_dim * (2 ** i) for i in range(4)]  # [C, 2C, 4C, 8C]

        # Stage 1
        self.enc_blocks1 = nn.ModuleList([
            SwinTransformerBlock(dims[0], num_heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[0])
        ])
        self.down1 = PatchMerging(dims[0])

        # Stage 2
        self.enc_blocks2 = nn.ModuleList([
            SwinTransformerBlock(dims[1], num_heads[1], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[1])
        ])
        self.down2 = PatchMerging(dims[1])

        # Stage 3
        self.enc_blocks3 = nn.ModuleList([
            SwinTransformerBlock(dims[2], num_heads[2], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[2])
        ])
        self.down3 = PatchMerging(dims[2])

        # Stage 4 / Bottleneck
        self.bottleneck_blocks = nn.ModuleList([
            SwinTransformerBlock(dims[3], num_heads[3], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[3])
        ])

        # Decoder stages
        # Up 3
        self.up3 = PatchExpanding(dims[3])
        self.fuse3 = nn.Linear(dims[2] + dims[2], dims[2])
        self.dec_blocks3 = nn.ModuleList([
            SwinTransformerBlock(dims[2], num_heads[2], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[2])
        ])

        # Up 2
        self.up2 = PatchExpanding(dims[2])
        self.fuse2 = nn.Linear(dims[1] + dims[1], dims[1])
        self.dec_blocks2 = nn.ModuleList([
            SwinTransformerBlock(dims[1], num_heads[1], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[1])
        ])

        # Up 1
        self.up1 = PatchExpanding(dims[1])
        self.fuse1 = nn.Linear(dims[0] + dims[0], dims[0])
        self.dec_blocks1 = nn.ModuleList([
            SwinTransformerBlock(dims[0], num_heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[0])
        ])

        # Final patch expanding to full image resolution (4x upsample)
        self.final_up = FinalPatchExpanding(dims[0], embed_dim // 2)
        self.head = nn.Linear(embed_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, orig_h, orig_w = x.shape

        # Patch embedding: [B, C, H/4, W/4]
        tokens = self.patch_embed(x)
        _, _, h1, w1 = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.norm_stem(tokens)

        # Encoder Stage 1
        for blk in self.enc_blocks1:
            tokens = blk(tokens, h1, w1)
        skip1 = tokens  # [B, H1*W1, C]

        # Encoder Stage 2
        tokens, h2, w2 = self.down1(tokens, h1, w1)
        for blk in self.enc_blocks2:
            tokens = blk(tokens, h2, w2)
        skip2 = tokens  # [B, H2*W2, 2C]

        # Encoder Stage 3
        tokens, h3, w3 = self.down2(tokens, h2, w2)
        for blk in self.enc_blocks3:
            tokens = blk(tokens, h3, w3)
        skip3 = tokens  # [B, H3*W3, 4C]

        # Bottleneck Stage 4
        tokens, h4, w4 = self.down3(tokens, h3, w3)
        for blk in self.bottleneck_blocks:
            tokens = blk(tokens, h4, w4)

        # Decoder Stage 3
        tokens, h3_up, w3_up = self.up3(tokens, h4, w4)
        tokens = self.fuse3(torch.cat([tokens, skip3], dim=-1))
        for blk in self.dec_blocks3:
            tokens = blk(tokens, h3_up, w3_up)

        # Decoder Stage 2
        tokens, h2_up, w2_up = self.up2(tokens, h3_up, w3_up)
        tokens = self.fuse2(torch.cat([tokens, skip2], dim=-1))
        for blk in self.dec_blocks2:
            tokens = blk(tokens, h2_up, w2_up)

        # Decoder Stage 1
        tokens, h1_up, w1_up = self.up1(tokens, h2_up, w2_up)
        tokens = self.fuse1(torch.cat([tokens, skip1], dim=-1))
        for blk in self.dec_blocks1:
            tokens = blk(tokens, h1_up, w1_up)

        # Final patch expanding to full resolution
        out_grid, _, _ = self.final_up(tokens, h1_up, w1_up)  # [B, H, W, embed_dim // 2]
        logits = self.head(out_grid)  # [B, H, W, num_classes]
        logits = logits.permute(0, 3, 1, 2).contiguous()  # [B, num_classes, H, W]

        return logits
