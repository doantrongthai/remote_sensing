"""
CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers (IEEE TIP 2023)
Adapted for Sentinel-2 Multimodal Remote Sensing (RGB + NIR).

Original Paper: "CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers"
Reference: https://github.com/huaaaliu/RGBX_Semantic_Segmentation

Key Components:
- Dual-stream hierarchical Vision Transformer encoder (RGB stream + NIR stream)
- Cross-Modal Feature Rectification Module (CM-FRM) at each stage (channel & spatial rectification)
- Feature Fusion Module (FFM) for cross-modal interaction
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
    Spatial Reduction Self-Attention for hierarchical Vision Transformers.
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
    Standard Transformer Encoder block with spatial-reduction attention and MLP.
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


class CMFRM(nn.Module):
    """
    Cross-Modal Feature Rectification Module (CM-FRM).
    Rectifies noisy or ambiguous features across spatial and channel dimensions
    using complementary cues from the other modality.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(8, channels // reduction)

        # Channel rectification MLP
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial rectification conv (mean + max pooling along channels -> 2 channels)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.post_rgb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.post_x = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_rgb, f_x):
        # 1. Channel Rectification: GAP descriptors
        gap_rgb = torch.mean(f_rgb, dim=(2, 3), keepdim=True)
        gap_x = torch.mean(f_x, dim=(2, 3), keepdim=True)

        w_c_rgb = self.channel_mlp(gap_rgb)  # Importance weights of RGB
        w_c_x = self.channel_mlp(gap_x)      # Importance weights of X (NIR)

        # Cross-rectify channel dimensions
        f_rgb_c = f_rgb * w_c_x
        f_x_c = f_x * w_c_rgb

        # 2. Spatial Rectification: channel-wise mean and max
        s_rgb_stats = torch.cat([
            torch.mean(f_rgb, dim=1, keepdim=True),
            torch.max(f_rgb, dim=1, keepdim=True)[0]
        ], dim=1)
        s_x_stats = torch.cat([
            torch.mean(f_x, dim=1, keepdim=True),
            torch.max(f_x, dim=1, keepdim=True)[0]
        ], dim=1)

        w_s_rgb = self.spatial_conv(s_rgb_stats)
        w_s_x = self.spatial_conv(s_x_stats)

        # Cross-rectify spatial dimensions
        f_rgb_s = f_rgb * w_s_x
        f_x_s = f_x * w_s_rgb

        # Residual combination
        f_rgb_rect = self.post_rgb(f_rgb + f_rgb_c + f_rgb_s)
        f_x_rect = self.post_x(f_x + f_x_c + f_x_s)

        return f_rgb_rect, f_x_rect


class FFM(nn.Module):
    """
    Feature Fusion Module (FFM).
    Performs cross-modal interaction and channel-adaptive feature fusion.
    """
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_rgb, f_x):
        # Concatenate along channel dimension
        cat = torch.cat([f_rgb, f_x], dim=1)
        gap = self.gap(cat)
        alpha = self.gate(gap)  # Cross-modal channel weight: in [0, 1]

        # Adaptive channel-weighted fusion
        f_fused = alpha * f_rgb + (1.0 - alpha) * f_x
        return self.fusion_conv(f_fused)


class CMX(nn.Module):
    """
    CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers.
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

        # NIR (X) Patch Embeddings (1 channel)
        self.x_patch1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=1, embed_dim=embed_dims[0])
        self.x_patch2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.x_patch3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.x_patch4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Dual-stream Transformer Blocks
        self.rgb_block1 = TransformerBlock(embed_dims[0], num_heads[0], sr_ratios[0])
        self.rgb_block2 = TransformerBlock(embed_dims[1], num_heads[1], sr_ratios[1])
        self.rgb_block3 = TransformerBlock(embed_dims[2], num_heads[2], sr_ratios[2])
        self.rgb_block4 = TransformerBlock(embed_dims[3], num_heads[3], sr_ratios[3])

        self.x_block1 = TransformerBlock(embed_dims[0], num_heads[0], sr_ratios[0])
        self.x_block2 = TransformerBlock(embed_dims[1], num_heads[1], sr_ratios[1])
        self.x_block3 = TransformerBlock(embed_dims[2], num_heads[2], sr_ratios[2])
        self.x_block4 = TransformerBlock(embed_dims[3], num_heads[3], sr_ratios[3])

        # Cross-Modal Feature Rectification Modules (CM-FRM)
        self.frm1 = CMFRM(embed_dims[0])
        self.frm2 = CMFRM(embed_dims[1])
        self.frm3 = CMFRM(embed_dims[2])
        self.frm4 = CMFRM(embed_dims[3])

        # Feature Fusion Modules (FFM)
        self.ffm1 = FFM(embed_dims[0])
        self.ffm2 = FFM(embed_dims[1])
        self.ffm3 = FFM(embed_dims[2])
        self.ffm4 = FFM(embed_dims[3])

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
        f_nir, _, _ = self.x_patch1(nir)
        f_rgb = self.rgb_block1(f_rgb, h1, w1).permute(0, 2, 1).reshape(b, -1, h1, w1)
        f_nir = self.x_block1(f_nir, h1, w1).permute(0, 2, 1).reshape(b, -1, h1, w1)
        f_rgb, f_nir = self.frm1(f_rgb, f_nir)
        fuse1 = self.ffm1(f_rgb, f_nir)

        # --- Stage 2 ---
        f_rgb, h2, w2 = self.rgb_patch2(f_rgb)
        f_nir, _, _ = self.x_patch2(f_nir)
        f_rgb = self.rgb_block2(f_rgb, h2, w2).permute(0, 2, 1).reshape(b, -1, h2, w2)
        f_nir = self.x_block2(f_nir, h2, w2).permute(0, 2, 1).reshape(b, -1, h2, w2)
        f_rgb, f_nir = self.frm2(f_rgb, f_nir)
        fuse2 = self.ffm2(f_rgb, f_nir)

        # --- Stage 3 ---
        f_rgb, h3, w3 = self.rgb_patch3(f_rgb)
        f_nir, _, _ = self.x_patch3(f_nir)
        f_rgb = self.rgb_block3(f_rgb, h3, w3).permute(0, 2, 1).reshape(b, -1, h3, w3)
        f_nir = self.x_block3(f_nir, h3, w3).permute(0, 2, 1).reshape(b, -1, h3, w3)
        f_rgb, f_nir = self.frm3(f_rgb, f_nir)
        fuse3 = self.ffm3(f_rgb, f_nir)

        # --- Stage 4 ---
        f_rgb, h4, w4 = self.rgb_patch4(f_rgb)
        f_nir, _, _ = self.x_patch4(f_nir)
        f_rgb = self.rgb_block4(f_rgb, h4, w4).permute(0, 2, 1).reshape(b, -1, h4, w4)
        f_nir = self.x_block4(f_nir, h4, w4).permute(0, 2, 1).reshape(b, -1, h4, w4)
        f_rgb, f_nir = self.frm4(f_rgb, f_nir)
        fuse4 = self.ffm4(f_rgb, f_nir)

        # --- MLP Decoder ---
        c1 = self.linear_c1(fuse1)
        c2 = F.interpolate(self.linear_c2(fuse2), size=(h1, w1), mode='bilinear', align_corners=False)
        c3 = F.interpolate(self.linear_c3(fuse3), size=(h1, w1), mode='bilinear', align_corners=False)
        c4 = F.interpolate(self.linear_c4(fuse4), size=(h1, w1), mode='bilinear', align_corners=False)

        fused = self.linear_fuse(torch.cat([c1, c2, c3, c4], dim=1))
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return logits
