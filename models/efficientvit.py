import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class CascadedGroupAttention(nn.Module):
    """
    Cascaded Group Attention (CGA) module from EfficientViT (CVPR 2023).
    Splits features across heads and cascades features while using multi-scale depthwise convs.
    """
    def __init__(self, dim, num_heads=4, key_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dim = dim
        self.head_dim = dim // num_heads

        # Multi-scale depthwise kernels for different heads
        kernels = [3, 5, 7, 9]
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(
                self.head_dim, self.head_dim,
                kernel_size=kernels[i % len(kernels)],
                padding=kernels[i % len(kernels)] // 2,
                groups=self.head_dim,
                bias=False
            ) for i in range(num_heads)
        ])
        
        self.qkv_convs = nn.ModuleList([
            nn.Conv2d(self.head_dim, key_dim * 2 + self.head_dim, kernel_size=1, bias=False)
            for _ in range(num_heads)
        ])
        
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        splits = torch.split(x, self.head_dim, dim=1)
        
        head_outputs = []
        prev_head = None
        for i, split in enumerate(splits):
            if prev_head is not None:
                split = split + prev_head
            
            # Multi-scale depthwise conv
            split_dw = self.dw_convs[i](split)
            qkv = self.qkv_convs[i](split_dw)
            
            q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.head_dim], dim=1)
            
            # Linear multi-scale attention (ReLU linear attention)
            q = F.relu(q).view(B, self.key_dim, N)
            k = F.relu(k).view(B, self.key_dim, N)
            v = v.view(B, self.head_dim, N)
            
            # Attention with linear complexity O(N * d^2)
            # context: [B, head_dim, key_dim]
            context = torch.bmm(v, k.transpose(1, 2))
            norm = (q * k.sum(dim=-1, keepdim=True)).sum(dim=1, keepdim=True) + 1e-6
            
            out = torch.bmm(context, q) / norm
            out = out.view(B, self.head_dim, H, W)
            
            head_outputs.append(out)
            prev_head = out

        fused = torch.cat(head_outputs, dim=1)
        return self.proj_bn(self.proj(fused))

class EfficientViTBlock(nn.Module):
    """
    EfficientViT block combining local conv mixer, Cascaded Group Attention, and FFN.
    """
    def __init__(self, dim, num_heads=4, key_dim=16, mlp_ratio=2.):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.attn = CascadedGroupAttention(dim, num_heads=num_heads, key_dim=key_dim)
        
        hidden_dim = int(dim * mlp_ratio)
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
        x = x + self.local_conv(x)
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class EfficientViTSegHead(nn.Module):
    def __init__(self, in_channels_list, out_channels=128, num_classes=2):
        super().__init__()
        self.projs = nn.ModuleList([
            ConvBNReLU(c, out_channels, kernel_size=1, padding=0)
            for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            ConvBNReLU(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

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

class EfficientViT(nn.Module):
    """
    EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction (CVPR 2023).
    Cascaded Group Attention with linear multi-scale attention.
    """
    def __init__(self, in_channels=4, num_classes=2, dims=[32, 64, 128, 256], depths=[1, 2, 3, 2]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Patch embedding / Stem downsampling to 1/4
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1)
        )

        # Stage 1: 1/4 resolution
        self.stage1 = nn.Sequential(*[
            EfficientViTBlock(dims[0], num_heads=2, key_dim=16) for _ in range(depths[0])
        ])

        # Stage 2: 1/8 resolution
        self.down1 = ConvBNReLU(dims[0], dims[1], kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(*[
            EfficientViTBlock(dims[1], num_heads=4, key_dim=16) for _ in range(depths[1])
        ])

        # Stage 3: 1/16 resolution
        self.down2 = ConvBNReLU(dims[1], dims[2], kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Sequential(*[
            EfficientViTBlock(dims[2], num_heads=4, key_dim=16) for _ in range(depths[2])
        ])

        # Stage 4: 1/32 resolution
        self.down3 = ConvBNReLU(dims[2], dims[3], kernel_size=3, stride=2, padding=1)
        self.stage4 = nn.Sequential(*[
            EfficientViTBlock(dims[3], num_heads=8, key_dim=16) for _ in range(depths[3])
        ])

        # Segmentation Head
        self.head = EfficientViTSegHead(dims, out_channels=128, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        c1 = self.stage1(self.stem(x))
        c2 = self.stage2(self.down1(c1))
        c3 = self.stage3(self.down2(c2))
        c4 = self.stage4(self.down3(c3))
        
        logits = self.head([c1, c2, c3, c4], target_size=(h, w))
        return logits
