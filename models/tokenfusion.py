"""
TokenFusion: Multimodal Token Fusion for Vision Transformers (CVPR 2022)
Adapted for Sentinel-2 Multimodal Remote Sensing (RGB + NIR).

Original Paper: "Multimodal Token Fusion for Vision Transformers"
Reference: https://github.com/yikai-wang/TokenFusion

Key Components:
- Dual-stream Transformer architecture (RGB + NIR)
- Dynamic Token Pruning: Computes importance scores for individual spatial tokens
- Cross-Modal Token Replacement: Replaces uninformative tokens in one modality
  with projected tokens from the complementary modality
- Residual Positional Alignment: Retains spatial coordinate consistency
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
    Spatial Reduction Self-Attention for Transformer blocks.
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


class TokenFusionBlock(nn.Module):
    """
    TokenFusion Block:
    1. Unimodal Self-Attention + FFN for RGB and NIR
    2. Dynamic token importance score computation
    3. Cross-modal token substitution: uninformative tokens are replaced with
       projected tokens from the other modality with residual alignment.
    """
    def __init__(self, dim, num_heads=4, sr_ratio=1, mlp_ratio=4, threshold=0.5):
        super().__init__()
        self.threshold = threshold

        # RGB branch
        self.norm1_rgb = nn.LayerNorm(dim)
        self.attn_rgb = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2_rgb = nn.LayerNorm(dim)
        self.mlp_rgb = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

        # NIR branch
        self.norm1_nir = nn.LayerNorm(dim)
        self.attn_nir = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2_nir = nn.LayerNorm(dim)
        self.mlp_nir = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

        # Token importance score estimators
        self.score_rgb = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        self.score_nir = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        # Cross-modal projection layers for replaced tokens
        self.proj_nir_to_rgb = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.proj_rgb_to_nir = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        # Residual positional alignment
        self.align_rgb = nn.Linear(dim, dim)
        self.align_nir = nn.Linear(dim, dim)

    def forward(self, x_rgb, x_nir, h, w):
        # 1. Unimodal Transformer processing
        x_rgb = x_rgb + self.attn_rgb(self.norm1_rgb(x_rgb), h, w)
        x_rgb = x_rgb + self.mlp_rgb(self.norm2_rgb(x_rgb))

        x_nir = x_nir + self.attn_nir(self.norm1_nir(x_nir), h, w)
        x_nir = x_nir + self.mlp_nir(self.norm2_nir(x_nir))

        # 2. Dynamic Token Importance Estimation
        s_rgb = self.score_rgb(x_rgb)  # [B, N, 1]
        s_nir = self.score_nir(x_nir)  # [B, N, 1]

        # Straight-Through Estimator for binary selection with gradient flow
        mask_rgb = (s_rgb > self.threshold).float()
        mask_rgb = s_rgb + (mask_rgb - s_rgb).detach()

        mask_nir = (s_nir > self.threshold).float()
        mask_nir = s_nir + (mask_nir - s_nir).detach()

        # 3. Cross-Modal Token Replacement
        # Replace uninformative RGB tokens with projected NIR tokens
        projected_nir = self.proj_nir_to_rgb(x_nir) + self.align_rgb(x_rgb)
        x_rgb_fused = mask_rgb * x_rgb + (1.0 - mask_rgb) * projected_nir

        # Replace uninformative NIR tokens with projected RGB tokens
        projected_rgb = self.proj_rgb_to_nir(x_rgb) + self.align_nir(x_nir)
        x_nir_fused = mask_nir * x_nir + (1.0 - mask_nir) * projected_rgb

        return x_rgb_fused, x_nir_fused


class TokenFusion(nn.Module):
    """
    TokenFusion: Multimodal Token Fusion for Vision Transformers (CVPR 2022).
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

        # Patch Embeddings for RGB (3 channels)
        self.patch_rgb1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        self.patch_rgb2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_rgb3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_rgb4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Patch Embeddings for NIR (1 channel)
        self.patch_nir1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=1, embed_dim=embed_dims[0])
        self.patch_nir2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_nir3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_nir4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # TokenFusion stages
        self.stage1 = TokenFusionBlock(embed_dims[0], num_heads[0], sr_ratios[0])
        self.stage2 = TokenFusionBlock(embed_dims[1], num_heads[1], sr_ratios[1])
        self.stage3 = TokenFusionBlock(embed_dims[2], num_heads[2], sr_ratios[2])
        self.stage4 = TokenFusionBlock(embed_dims[3], num_heads[3], sr_ratios[3])

        # Stage fusion modules (combines post-replacement RGB and NIR)
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(embed_dims[0] * 2, embed_dims[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(embed_dims[1] * 2, embed_dims[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv3 = nn.Sequential(
            nn.Conv2d(embed_dims[2] * 2, embed_dims[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv4 = nn.Sequential(
            nn.Conv2d(embed_dims[3] * 2, embed_dims[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.ReLU(inplace=True)
        )

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
        t_rgb, h1, w1 = self.patch_rgb1(rgb)
        t_nir, _, _ = self.patch_nir1(nir)
        t_rgb, t_nir = self.stage1(t_rgb, t_nir, h1, w1)

        c1_rgb = t_rgb.permute(0, 2, 1).reshape(b, -1, h1, w1)
        c1_nir = t_nir.permute(0, 2, 1).reshape(b, -1, h1, w1)
        fuse1 = self.fuse_conv1(torch.cat([c1_rgb, c1_nir], dim=1))

        # --- Stage 2 ---
        t_rgb, h2, w2 = self.patch_rgb2(c1_rgb)
        t_nir, _, _ = self.patch_nir2(c1_nir)
        t_rgb, t_nir = self.stage2(t_rgb, t_nir, h2, w2)

        c2_rgb = t_rgb.permute(0, 2, 1).reshape(b, -1, h2, w2)
        c2_nir = t_nir.permute(0, 2, 1).reshape(b, -1, h2, w2)
        fuse2 = self.fuse_conv2(torch.cat([c2_rgb, c2_nir], dim=1))

        # --- Stage 3 ---
        t_rgb, h3, w3 = self.patch_rgb3(c2_rgb)
        t_nir, _, _ = self.patch_nir3(c2_nir)
        t_rgb, t_nir = self.stage3(t_rgb, t_nir, h3, w3)

        c3_rgb = t_rgb.permute(0, 2, 1).reshape(b, -1, h3, w3)
        c3_nir = t_nir.permute(0, 2, 1).reshape(b, -1, h3, w3)
        fuse3 = self.fuse_conv3(torch.cat([c3_rgb, c3_nir], dim=1))

        # --- Stage 4 ---
        t_rgb, h4, w4 = self.patch_rgb4(c3_rgb)
        t_nir, _, _ = self.patch_nir4(c3_nir)
        t_rgb, t_nir = self.stage4(t_rgb, t_nir, h4, w4)

        c4_rgb = t_rgb.permute(0, 2, 1).reshape(b, -1, h4, w4)
        c4_nir = t_nir.permute(0, 2, 1).reshape(b, -1, h4, w4)
        fuse4 = self.fuse_conv4(torch.cat([c4_rgb, c4_nir], dim=1))

        # --- Decoder ---
        c1 = self.linear_c1(fuse1)
        c2 = F.interpolate(self.linear_c2(fuse2), size=(h1, w1), mode='bilinear', align_corners=False)
        c3 = F.interpolate(self.linear_c3(fuse3), size=(h1, w1), mode='bilinear', align_corners=False)
        c4 = F.interpolate(self.linear_c4(fuse4), size=(h1, w1), mode='bilinear', align_corners=False)

        fused = self.linear_fuse(torch.cat([c1, c2, c3, c4], dim=1))
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return logits
