import torch
import torch.nn as nn
import torch.nn.functional as F

class SDTA(nn.Module):
    """
    Split Depthwise Transpose Attention (SDTA) module from EdgeNeXt (ECCV-W 2022).
    Combines multi-scale depthwise convolutions with transposed (channel) multi-head self-attention.
    """
    def __init__(self, dim, num_heads=4, scales=(3, 5, 7)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Split channels into 3 groups for multi-scale DW convs
        self.split_dims = [dim // len(scales)] * len(scales)
        self.split_dims[-1] = dim - sum(self.split_dims[:-1])

        self.convs = nn.ModuleList([
            nn.Conv2d(self.split_dims[i], self.split_dims[i], kernel_size=scales[i],
                      padding=scales[i] // 2, groups=self.split_dims[i], bias=False)
            for i in range(len(scales))
        ])

        # Transposed Multi-Head Self-Attention projections
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Multi-scale split depthwise conv
        splits = torch.split(x, self.split_dims, dim=1)
        conv_out = torch.cat([conv(s) for conv, s in zip(self.convs, splits)], dim=1)
        
        # Transposed Multi-Head Self-Attention (Channel Attention)
        q = self.q(conv_out).view(B, self.num_heads, self.head_dim, H * W)
        k = self.k(conv_out).view(B, self.num_heads, self.head_dim, H * W)
        v = self.v(conv_out).view(B, self.num_heads, self.head_dim, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Attention across channels: [B, heads, head_dim, head_dim]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).view(B, C, H, W)
        return self.norm(self.proj(out))

class EdgeNeXtBlock(nn.Module):
    """EdgeNeXt Block: SDTA followed by Inverted Bottleneck FFN"""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.sdta = SDTA(dim, num_heads=num_heads)
        
        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.sdta(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class EdgeNeXtSegHead(nn.Module):
    def __init__(self, in_channels_list, out_channels=128, num_classes=2):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

class EdgeNeXt(nn.Module):
    """
    EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture (ECCV-W 2022).
    Split Depthwise Transpose Attention (SDTA) with multi-scale segmentation head.
    """
    def __init__(self, in_channels=4, num_classes=2, dims=[32, 64, 128, 256], depths=[2, 2, 4, 2]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem downsampling to 1/4 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )

        # Stage 1: 1/4 resolution
        self.stage1 = nn.Sequential(*[
            EdgeNeXtBlock(dims[0], num_heads=2) for _ in range(depths[0])
        ])

        # Stage 2: 1/8 resolution
        self.down1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[1])
        )
        self.stage2 = nn.Sequential(*[
            EdgeNeXtBlock(dims[1], num_heads=4) for _ in range(depths[1])
        ])

        # Stage 3: 1/16 resolution
        self.down2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[2])
        )
        self.stage3 = nn.Sequential(*[
            EdgeNeXtBlock(dims[2], num_heads=4) for _ in range(depths[2])
        ])

        # Stage 4: 1/32 resolution
        self.down3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[3])
        )
        self.stage4 = nn.Sequential(*[
            EdgeNeXtBlock(dims[3], num_heads=8) for _ in range(depths[3])
        ])

        # Decoder Head
        self.head = EdgeNeXtSegHead(dims, out_channels=128, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        c1 = self.stage1(self.stem(x))
        c2 = self.stage2(self.down1(c1))
        c3 = self.stage3(self.down2(c2))
        c4 = self.stage4(self.down3(c3))
        
        logits = self.head([c1, c2, c3, c4], target_size=(h, w))
        return logits
