"""
Vision-RWKV: Efficient and Effective Vision Backbone with RWKV (CVPR 2024)
Paper: https://arxiv.org/abs/2404.02978
Official Repo: https://github.com/YumianChen/Vision-RWKV

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Implements spatial-decay bidirectional WKV recurrence, 2D spatial Q-shift, and
RWKV channel-mixing modules with linear computational complexity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def q_shift_2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D Q-Shift: shifts feature quarters along 4 cardinal directions (up, down, left, right)
    to mix adjacent spatial context without extra parameters.
    Input: [B, C, H, W]
    Output: [B, C, H, W]
    """
    b, c, h, w = x.shape
    c_quarter = c // 4
    out = torch.zeros_like(x)

    # Down shift (quarter 0)
    out[:, :c_quarter, 1:, :] = x[:, :c_quarter, :-1, :]

    # Up shift (quarter 1)
    out[:, c_quarter : 2 * c_quarter, :-1, :] = x[:, c_quarter : 2 * c_quarter, 1:, :]

    # Right shift (quarter 2)
    out[:, 2 * c_quarter : 3 * c_quarter, :, 1:] = x[:, 2 * c_quarter : 3 * c_quarter, :, :-1]

    # Left shift (quarter 3)
    out[:, 3 * c_quarter : 4 * c_quarter, :, :-1] = x[:, 3 * c_quarter : 4 * c_quarter, :, 1:]

    # Remaining channels if not divisible by 4
    if c > 4 * c_quarter:
        out[:, 4 * c_quarter :] = x[:, 4 * c_quarter :]

    return out


class VRWKV_SpatialMix(nn.Module):
    """
    Bidirectional Spatial-Mix block with learned spatial decay and bonus.
    Implements:
      L[c, t, i] = k[c, i] - (spatial_decay[c] / T) * dist(t, i) + (spatial_first[c] / T) * delta(t, i)
      y[c, t] = sum_i softmax(L[c, t, :]) * v[c, i]
      out = sigmoid(r) * y
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd

        # Interpolation / mix parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, n_embd, 1, 1) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, n_embd, 1, 1) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, n_embd, 1, 1) * 0.5)

        # Projections
        self.key = nn.Conv2d(n_embd, n_embd, kernel_size=1, bias=False)
        self.value = nn.Conv2d(n_embd, n_embd, kernel_size=1, bias=False)
        self.receptance = nn.Conv2d(n_embd, n_embd, kernel_size=1, bias=False)
        self.output = nn.Conv2d(n_embd, n_embd, kernel_size=1, bias=False)

        # Spatial decay and spatial bonus parameters
        self.spatial_decay = nn.Parameter(torch.randn(n_embd) * 0.1)
        self.spatial_first = nn.Parameter(torch.randn(n_embd) * 0.1)
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = h * w

        # 2D spatial shift
        x_shift = q_shift_2d(x)

        # Channel mixing with shift
        xk = x * self.time_mix_k + x_shift * (1.0 - self.time_mix_k)
        xv = x * self.time_mix_v + x_shift * (1.0 - self.time_mix_v)
        xr = x * self.time_mix_r + x_shift * (1.0 - self.time_mix_r)

        # Projections
        k = self.key(xk)          # [B, C, H, W]
        v = self.value(xv)        # [B, C, H, W]
        r = self.receptance(xr)   # [B, C, H, W]

        # Reshape to token sequence: [B, C, T]
        k_seq = k.flatten(2)
        v_seq = v.flatten(2)
        r_seq = r.flatten(2)

        # 2D Grid coordinates for Manhattan spatial distance
        coords_h = torch.arange(h, device=x.device, dtype=torch.float32).repeat_interleave(w)
        coords_w = torch.arange(w, device=x.device, dtype=torch.float32).repeat(h)
        dist_h = torch.abs(coords_h.unsqueeze(0) - coords_h.unsqueeze(1))
        dist_w = torch.abs(coords_w.unsqueeze(0) - coords_w.unsqueeze(1))
        dist = dist_h + dist_w  # [T, T]

        eye = torch.eye(t, device=x.device, dtype=torch.float32)

        # Attention logits with spatial decay
        # k_exp: [B, C, 1, T]
        k_exp = k_seq.unsqueeze(2)
        decay = (F.softplus(self.spatial_decay) / float(t)).view(1, c, 1, 1) * dist.view(1, 1, t, t)
        first = (self.spatial_first / float(t)).view(1, c, 1, 1) * eye.view(1, 1, t, t)

        logits = k_exp - decay + first
        attn = torch.softmax(logits, dim=-1)  # [B, C, T, T]

        # Aggregate values: [B, C, T, 1]
        v_exp = v_seq.unsqueeze(-1)
        y = torch.matmul(attn, v_exp).squeeze(-1)  # [B, C, T]

        # LayerNorm along channel dimension
        y_ln = self.norm(y.transpose(1, 2)).transpose(1, 2)  # [B, C, T]
        out_seq = torch.sigmoid(r_seq) * y_ln
        out = out_seq.view(b, c, h, w)

        return self.output(out)


class VRWKV_ChannelMix(nn.Module):
    """
    Feed-Forward Channel-Mix block with squared ReLU activation and receptance gating.
    """
    def __init__(self, n_embd: int, hidden_mult: int = 4):
        super().__init__()
        self.n_embd = n_embd
        hidden_dim = n_embd * hidden_mult

        self.time_mix_k = nn.Parameter(torch.ones(1, n_embd, 1, 1) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, n_embd, 1, 1) * 0.5)

        self.key = nn.Conv2d(n_embd, hidden_dim, kernel_size=1, bias=False)
        self.receptance = nn.Conv2d(n_embd, n_embd, kernel_size=1, bias=False)
        self.value = nn.Conv2d(hidden_dim, n_embd, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shift = q_shift_2d(x)
        xk = x * self.time_mix_k + x_shift * (1.0 - self.time_mix_k)
        xr = x * self.time_mix_r + x_shift * (1.0 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))  # Squared ReLU
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(xr))

        return r * kv


class VRWKVBlock(nn.Module):
    """Vision-RWKV encoder block pairing Spatial-Mix with Channel-Mix."""
    def __init__(self, n_embd: int):
        super().__init__()
        self.ln1 = nn.BatchNorm2d(n_embd)
        self.spatial_mix = VRWKV_SpatialMix(n_embd)
        self.ln2 = nn.BatchNorm2d(n_embd)
        self.channel_mix = VRWKV_ChannelMix(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spatial_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class ConvBlock(nn.Module):
    """Two 3x3 Conv-BN-ReLU layers."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VisionRWKV(nn.Module):
    """
    Vision-RWKV segmentation network for multi-spectral remote sensing.
    Utilizes bidirectional spatial-decay WKV recurrence blocks in the multi-scale
    encoder and an upsampling decoder with skip connections.
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

        # Stem: Conv to C0 at full resolution
        self.stem = ConvBlock(in_channels, channels[0])

        # Stage 1: down to H/2, W/2
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[0], channels[1]),
        )

        # Stage 2: down to H/4, W/4 with RWKV
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[1], channels[2]),
        )

        # Stage 3: down to H/8, W/8 with VRWKV Block
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[2], channels[3]),
            VRWKVBlock(channels[3]),
        )

        # Bottleneck: down to H/16, W/16 with VRWKV Block (16x16 = 256 tokens)
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[3], channels[3] * 2),
            VRWKVBlock(channels[3] * 2),
        )

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(channels[3] * 2, channels[3], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(channels[3] * 2, channels[3]),
            VRWKVBlock(channels[3]),
        )

        self.up2 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(channels[2] * 2, channels[2])

        self.up1 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(channels[1] * 2, channels[1])

        self.up0 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec0 = ConvBlock(channels[0] * 2, channels[0])

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.stem(x)             # [B, 32, 256, 256]
        x1 = self.down1(x0)           # [B, 64, 128, 128]
        x2 = self.down2(x1)           # [B, 128, 64, 64]
        x3 = self.down3(x2)           # [B, 256, 32, 32]
        b = self.bottleneck(x3)       # [B, 512, 16, 16]

        # Decoder
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
