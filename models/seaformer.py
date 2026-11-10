import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1, act=nn.ReLU):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            act(inplace=True)
        )

class SqueezeAxialAttention(nn.Module):
    """
    Squeeze-enhanced Axial Attention (SEA Attention) from SeaFormer (ICLR 2023).
    Squeezes 2D features into 1D horizontal and vertical axial tokens to achieve
    global receptive field with linear complexity for mobile segmentation.
    """
    def __init__(self, dim, num_heads=4, key_dim=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = dim // num_heads

        # Squeeze projections
        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.w_pool = nn.AdaptiveAvgPool2d((1, None))

        # Axial query, key, value for horizontal / vertical tokens
        self.qkv_h = nn.Conv2d(dim, key_dim * 2 + dim, 1, bias=False)
        self.qkv_w = nn.Conv2d(dim, key_dim * 2 + dim, 1, bias=False)

        self.proj = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Horizontal squeeze (axial height tokens)
        xh = self.h_pool(x)  # [B, C, H, 1]
        qkv_h = self.qkv_h(xh).squeeze(-1) # [B, 2*key_dim + C, H]
        qh, kh, vh = torch.split(qkv_h, [self.key_dim, self.key_dim, C], dim=1)
        
        # Self-attention along H
        attn_h = torch.bmm(qh.transpose(1, 2), kh) * (self.key_dim ** -0.5)
        attn_h = F.softmax(attn_h, dim=-1)
        out_h = torch.bmm(vh, attn_h.transpose(1, 2)).unsqueeze(-1) # [B, C, H, 1]
        out_h = out_h.expand(-1, -1, -1, W) # Broadcast to [B, C, H, W]

        # 2. Vertical squeeze (axial width tokens)
        xw = self.w_pool(x)  # [B, C, 1, W]
        qkv_w = self.qkv_w(xw).squeeze(-2) # [B, 2*key_dim + C, W]
        qw, kw, vw = torch.split(qkv_w, [self.key_dim, self.key_dim, C], dim=1)
        
        # Self-attention along W
        attn_w = torch.bmm(qw.transpose(1, 2), kw) * (self.key_dim ** -0.5)
        attn_w = F.softmax(attn_w, dim=-1)
        out_w = torch.bmm(vw, attn_w.transpose(1, 2)).unsqueeze(-2) # [B, C, 1, W]
        out_w = out_w.expand(-1, -1, H, -1) # Broadcast to [B, C, H, W]

        # 3. Fuse horizontal and vertical axial context
        fused = self.bn(self.proj(torch.cat([out_h, out_w], dim=1)))
        fused = self.act(fused)
        fused = self.dw_bn(self.dw(fused))
        return fused

class SeaFormerBlock(nn.Module):
    """SeaFormer Block: SEA Attention + Feed-Forward Network"""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SqueezeAxialAttention(dim, num_heads=num_heads)
        
        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SeaFormerSegHead(nn.Module):
    def __init__(self, dims, out_channels=128, num_classes=2):
        super().__init__()
        self.projs = nn.ModuleList([
            ConvBNAct(c, out_channels, k=1, p=0)
            for c in dims
        ])
        self.fuse = nn.Sequential(
            ConvBNAct(out_channels * len(dims), out_channels, k=3, p=1),
            ConvBNAct(out_channels, out_channels, k=3, p=1)
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features, target_size):
        ref_size = features[0].shape[2:]
        proj_feats = []
        for feat, proj in zip(features, self.projs):
            p = proj(feat)
            if p.shape[2:] != ref_size:
                p = F.interpolate(p, size=ref_size, mode='bilinear', align_corners=False)
            proj_feats.append(p)
        fused = self.fuse(torch.cat(proj_feats, dim=1))
        logits = self.classifier(fused)
        return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

class SeaFormer(nn.Module):
    """
    SeaFormer: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation (ICLR 2023).
    Squeeze-enhanced axial attention with hierarchical mobile transformer backbone.
    """
    def __init__(self, in_channels=4, num_classes=2, dims=[32, 64, 128, 256], depths=[2, 2, 4, 2]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem downsample 4x
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, dims[0] // 2, k=3, s=2, p=1),
            ConvBNAct(dims[0] // 2, dims[0], k=3, s=2, p=1)
        )

        # Stage 1: 1/4
        self.stage1 = nn.Sequential(*[
            ConvBNAct(dims[0], dims[0], k=3, s=1, p=1) for _ in range(depths[0])
        ])

        # Stage 2: 1/8
        self.down1 = ConvBNAct(dims[0], dims[1], k=3, s=2, p=1)
        self.stage2 = nn.Sequential(*[
            ConvBNAct(dims[1], dims[1], k=3, s=1, p=1) for _ in range(depths[1])
        ])

        # Stage 3: 1/16 (with SeaFormer Blocks)
        self.down2 = ConvBNAct(dims[1], dims[2], k=3, s=2, p=1)
        self.stage3 = nn.Sequential(*[
            SeaFormerBlock(dims[2], num_heads=4) for _ in range(depths[2])
        ])

        # Stage 4: 1/32 (with SeaFormer Blocks)
        self.down3 = ConvBNAct(dims[2], dims[3], k=3, s=2, p=1)
        self.stage4 = nn.Sequential(*[
            SeaFormerBlock(dims[3], num_heads=8) for _ in range(depths[3])
        ])

        # Head
        self.head = SeaFormerSegHead(dims, out_channels=128, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        c1 = self.stage1(self.stem(x))
        c2 = self.stage2(self.down1(c1))
        c3 = self.stage3(self.down2(c2))
        c4 = self.stage4(self.down3(c3))

        logits = self.head([c1, c2, c3, c4], target_size=(h, w))
        return logits
