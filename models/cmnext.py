"""
CMNeXt: Delivering Arbitrary-Modal Semantic Segmentation (CVPR 2023)
Adapted for Sentinel-2 Multimodal Remote Sensing (RGB + NIR).

Original Paper: "Delivering Arbitrary-Modal Semantic Segmentation"
Reference: https://github.com/InSAI-Lab/DELIVER

Key Components:
- Hub2Fuse asymmetric paradigm:
  - Primary RGB branch: Hierarchical Transformer encoder (MiT)
  - Auxiliary NIR branch: Parallel Pooling Mixer (PPX)
- Self-Query Hub (SQ-Hub): RGB queries auxiliary cues with dynamic modal gating
- Multi-scale All-MLP Decoder for dense semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding with overlapping convolutions.
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    """
    Spatial Reduction Self-Attention for RGB transformer stream.
    """
    def __init__(self, dim, num_heads=4, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer block for the primary RGB branch.
    """
    def __init__(self, dim, num_heads=4, sr_ratio=1, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x))
        return x


class ParallelPoolingMixer(nn.Module):
    """
    Parallel Pooling Mixer (PPX) from CMNeXt.
    Captures multi-scale spatial receptive fields using parallel pooling operations
    at minimal computational cost for auxiliary modalities.
    """
    def __init__(self, dim):
        super().__init__()
        self.pool1 = nn.Identity()
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.mixer = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # Parallel pooling pathways
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        p4 = self.pool4(x)

        mixed = self.mixer(torch.cat([p1, p2, p3, p4], dim=1))
        x = x + self.dwconv(mixed)
        x = x + self.ffn(x)
        return x


class SelfQueryHub(nn.Module):
    """
    Self-Query Hub (SQ-Hub) from CMNeXt.
    RGB representation acts as Query to probe and extract discriminative cues
    from the auxiliary modality, with adaptive reliability gating.
    """
    def __init__(self, dim, num_heads=4, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.BatchNorm2d(dim)

        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # Dynamic gating mechanism to suppress noise / sensor degradation
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, aux):
        b, c, h, w = rgb.shape
        # Compute Query from RGB
        q = self.q(rgb).reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)

        # Compute Key and Value from Auxiliary modality
        if self.sr_ratio > 1:
            aux_sr = self.sr_norm(self.sr(aux))
            _, _, hs, ws = aux_sr.shape
            kv = self.kv(aux_sr).reshape(b, 2, self.num_heads, c // self.num_heads, hs * ws)
        else:
            kv = self.kv(aux).reshape(b, 2, self.num_heads, c // self.num_heads, h * w)

        k = kv[:, 0].permute(0, 1, 3, 2)
        v = kv[:, 1].permute(0, 1, 3, 2)

        # Cross attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        cross = (attn @ v).permute(0, 1, 3, 2).reshape(b, c, h, w)
        cross = self.proj(cross)

        # Dynamic reliability gate
        gate_weight = self.gate(torch.cat([rgb, aux], dim=1))
        out = rgb + gate_weight * cross
        return self.fusion(out)


class CMNeXt(nn.Module):
    """
    CMNeXt: Delivering Arbitrary-Modal Semantic Segmentation (CVPR 2023).
    Adapted for 4-channel input: Channels 0,1,2 = RGB, Channel 3 = NIR.
    """
    def __init__(
        self,
        in_channels=4,
        num_classes=2,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=128
    ):
        super().__init__()
        self.num_classes = num_classes

        # RGB Patch Embeddings
        self.rgb_patch1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        self.rgb_patch2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.rgb_patch3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.rgb_patch4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Auxiliary NIR Stem & Downsampling (PPX branch)
        self.aux_stem = nn.Sequential(
            nn.Conv2d(1, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.aux_down2 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.aux_down3 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(inplace=True)
        )
        self.aux_down4 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.ReLU(inplace=True)
        )

        # RGB Transformer Blocks
        self.rgb_block1 = TransformerBlock(embed_dims[0], num_heads[0], sr_ratios[0])
        self.rgb_block2 = TransformerBlock(embed_dims[1], num_heads[1], sr_ratios[1])
        self.rgb_block3 = TransformerBlock(embed_dims[2], num_heads[2], sr_ratios[2])
        self.rgb_block4 = TransformerBlock(embed_dims[3], num_heads[3], sr_ratios[3])

        # Auxiliary PPX Blocks
        self.ppx1 = ParallelPoolingMixer(embed_dims[0])
        self.ppx2 = ParallelPoolingMixer(embed_dims[1])
        self.ppx3 = ParallelPoolingMixer(embed_dims[2])
        self.ppx4 = ParallelPoolingMixer(embed_dims[3])

        # Self-Query Hubs (SQ-Hub)
        self.hub1 = SelfQueryHub(embed_dims[0], num_heads[0], sr_ratios[0])
        self.hub2 = SelfQueryHub(embed_dims[1], num_heads[1], sr_ratios[1])
        self.hub3 = SelfQueryHub(embed_dims[2], num_heads[2], sr_ratios[2])
        self.hub4 = SelfQueryHub(embed_dims[3], num_heads[3], sr_ratios[3])

        # All-MLP Decoder
        self.linear_c4 = nn.Conv2d(embed_dims[3], decoder_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(embed_dims[2], decoder_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(embed_dims[1], decoder_dim, kernel_size=1)
        self.linear_c1 = nn.Conv2d(embed_dims[0], decoder_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

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
        b = x.shape[0]

        rgb = x[:, :3, :, :]
        nir = x[:, 3:4, :, :]

        # --- Stage 1 ---
        f_rgb, h1, w1 = self.rgb_patch1(rgb)
        f_rgb = self.rgb_block1(f_rgb, h1, w1).permute(0, 2, 1).reshape(b, -1, h1, w1)
        f_aux = self.aux_stem(nir)
        f_aux = self.ppx1(f_aux)
        fuse1 = self.hub1(f_rgb, f_aux)

        # --- Stage 2 ---
        f_rgb, h2, w2 = self.rgb_patch2(fuse1)
        f_rgb = self.rgb_block2(f_rgb, h2, w2).permute(0, 2, 1).reshape(b, -1, h2, w2)
        f_aux = self.aux_down2(f_aux)
        f_aux = self.ppx2(f_aux)
        fuse2 = self.hub2(f_rgb, f_aux)

        # --- Stage 3 ---
        f_rgb, h3, w3 = self.rgb_patch3(fuse2)
        f_rgb = self.rgb_block3(f_rgb, h3, w3).permute(0, 2, 1).reshape(b, -1, h3, w3)
        f_aux = self.aux_down3(f_aux)
        f_aux = self.ppx3(f_aux)
        fuse3 = self.hub3(f_rgb, f_aux)

        # --- Stage 4 ---
        f_rgb, h4, w4 = self.rgb_patch4(fuse3)
        f_rgb = self.rgb_block4(f_rgb, h4, w4).permute(0, 2, 1).reshape(b, -1, h4, w4)
        f_aux = self.aux_down4(f_aux)
        f_aux = self.ppx4(f_aux)
        fuse4 = self.hub4(f_rgb, f_aux)

        # --- Decoder ---
        c1 = self.linear_c1(fuse1)
        c2 = F.interpolate(self.linear_c2(fuse2), size=(h1, w1), mode='bilinear', align_corners=False)
        c3 = F.interpolate(self.linear_c3(fuse3), size=(h1, w1), mode='bilinear', align_corners=False)
        c4 = F.interpolate(self.linear_c4(fuse4), size=(h1, w1), mode='bilinear', align_corners=False)

        fused = self.linear_fuse(torch.cat([c1, c2, c3, c4], dim=1))
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return logits
