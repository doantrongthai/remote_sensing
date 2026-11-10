"""
Vision-xLSTM: xLSTM as Generic Vision Backbone (NeurIPS 2024)
Paper: https://arxiv.org/abs/2406.04303
Official Repo: https://github.com/NX-AI/vision-lstm

Adapted for Sentinel-2 multi-spectral remote sensing imagery (4 input channels:
B4-Red, B3-Green, B2-Blue, B8-NIR) and binary flood water segmentation (num_classes=2).
Features bidirectional matrix-memory mLSTM (matrix Long Short-Term Memory) blocks
with exponential gating, key-value associative memory updates, and normalizer states.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bidirectional_mLSTM(nn.Module):
    """
    Bidirectional matrix Long Short-Term Memory (mLSTM) memory cell.
    Maintains a 2D memory matrix C_t and normalizer state n_t per head with
    exponential input/forget gating and output modulation:
      C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)
      n_t = f_t * n_{t-1} + i_t * k_t
      h_t = (C_t @ q_t) / max(|n_t^T @ q_t|, 1)
      y_t = o_t * h_t
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.head_dim = dim // num_heads

        # Projections for Q, K, V
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        # Gating projections
        self.i_gate = nn.Linear(dim, num_heads)
        self.f_gate = nn.Linear(dim, num_heads)
        self.o_gate = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def _scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Directional scan over sequence tokens.
        q, k, v: [B, H, T, d]
        i, f: [B, H, T]
        returns: [B, H, T, d]
        """
        b, h, t, d = q.shape
        out = torch.empty_like(v)

        c_state = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
        n_state = torch.zeros(b, h, d, device=q.device, dtype=q.dtype)

        for step in range(t):
            ft = f[:, :, step].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            it = i[:, :, step].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            kt = k[:, :, step]                              # [B, H, d]
            vt = v[:, :, step]                              # [B, H, d]
            qt = q[:, :, step]                              # [B, H, d]

            # Matrix memory state update
            # C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)
            outer = torch.matmul(vt.unsqueeze(-1), kt.unsqueeze(-2))
            c_state = ft * c_state + it * outer

            # Normalizer state update: n_t = f_t * n_{t-1} + i_t * k_t
            n_state = ft.squeeze(-1) * n_state + it.squeeze(-1) * kt

            # Readout: h_t = (C_t @ q_t) / max(|n_t^T @ q_t|, 1)
            cq = torch.matmul(c_state, qt.unsqueeze(-1)).squeeze(-1)
            nq = torch.abs(torch.sum(n_state * qt, dim=-1, keepdim=True)).clamp(min=1.0)
            out[:, :, step] = cq / nq

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        b, t, d = x.shape
        h = self.num_heads
        head_d = self.head_dim

        q = self.q(x).view(b, t, h, head_d).permute(0, 2, 1, 3)
        k = self.k(x).view(b, t, h, head_d).permute(0, 2, 1, 3)
        v = self.v(x).view(b, t, h, head_d).permute(0, 2, 1, 3)

        # Exponential input gate & sigmoid forget gate
        i = torch.exp(torch.clamp(self.i_gate(x).permute(0, 2, 1), -10.0, 10.0))
        f = torch.sigmoid(self.f_gate(x).permute(0, 2, 1))
        o = torch.sigmoid(self.o_gate(x))

        # Forward scan (top-left to bottom-right)
        h_fwd = self._scan(q, k, v, i, f)

        # Backward scan (bottom-right to top-left)
        q_rev = torch.flip(q, dims=[2])
        k_rev = torch.flip(k, dims=[2])
        v_rev = torch.flip(v, dims=[2])
        i_rev = torch.flip(i, dims=[2])
        f_rev = torch.flip(f, dims=[2])
        h_bwd = torch.flip(self._scan(q_rev, k_rev, v_rev, i_rev, f_rev), dims=[2])

        # Symmetrical bidirectional fusion
        h_bidir = 0.5 * (h_fwd + h_bwd)
        h_bidir = h_bidir.permute(0, 2, 1, 3).reshape(b, t, d)

        out = o * h_bidir
        return self.out_proj(out)


class ViLBlock(nn.Module):
    """
    Vision-xLSTM block:
    Pre-LayerNorm -> Bidirectional mLSTM -> Residual -> Pre-LayerNorm -> MLP -> Residual.
    """
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 4, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlstm = Bidirectional_mLSTM(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W]
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        tokens = tokens + self.mlstm(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))

        out = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out


class ConvBlock(nn.Module):
    """Two 3x3 Conv-BN-ReLU layers."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionxLSTM(nn.Module):
    """
    Vision-xLSTM segmentation network for remote sensing imagery.
    Equipped with bidirectional matrix-memory mLSTM blocks across deep semantic
    stages to retain long-range context without quadratic attention complexity.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        channels: list = [32, 64, 128, 256],
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem: Full resolution
        self.stem = ConvBlock(in_channels, channels[0])

        # Stage 1: down to H/2, W/2
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[0], channels[1]),
        )

        # Stage 2: down to H/4, W/4
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[1], channels[2]),
        )

        # Stage 3: down to H/8, W/8 with ViL mLSTM
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[2], channels[3]),
            ViLBlock(channels[3], num_heads=num_heads),
        )

        # Bottleneck: down to H/16, W/16 with ViL mLSTM
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(channels[3], channels[3] * 2),
            ViLBlock(channels[3] * 2, num_heads=num_heads),
        )

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(channels[3] * 2, channels[3], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(channels[3] * 2, channels[3]),
            ViLBlock(channels[3], num_heads=num_heads),
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
