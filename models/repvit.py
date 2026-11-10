import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w

class RepDWConv(nn.Module):
    """
    Structural re-parameterization depthwise convolution module for RepViT.
    During training: incorporates 3x3 DWConv, 1x1 DWConv, and identity.
    Can be seamlessly converted or run multi-branch.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn3x3 = nn.BatchNorm2d(channels)
        self.conv1x1 = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.bn1x1 = nn.BatchNorm2d(channels)
        self.identity = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x)) + self.identity(x)

class RepViTBlock(nn.Module):
    """
    RepViT Block (CVPR 2024):
    Token mixer with RepDWConv and SE, followed by Channel Mixer (FFN).
    """
    def __init__(self, dim, mlp_ratio=2., use_se=True):
        super().__init__()
        # Token Mixer
        self.token_mixer = RepDWConv(dim)
        self.se = SqueezeExcitation(dim) if use_se else nn.Identity()
        
        # Channel Mixer (FFN)
        hidden_dim = int(dim * mlp_ratio)
        self.channel_mixer = nn.Sequential(
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
        # Token mixer branch with residual
        x = x + self.se(self.token_mixer(x))
        # Channel mixer branch with residual
        x = x + self.channel_mixer(x)
        return x

class RepViTSegHead(nn.Module):
    """
    Multi-scale aggregation segmentation head for RepViT.
    """
    def __init__(self, in_channels_list, out_channels=128, num_classes=2):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ) for c in in_channels_list
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, feats, target_size):
        ref_size = feats[0].shape[2:]
        proj_feats = []
        for feat, proj in zip(feats, self.projs):
            p = proj(feat)
            if p.shape[2:] != ref_size:
                p = F.interpolate(p, size=ref_size, mode='bilinear', align_corners=False)
            proj_feats.append(p)
            
        fused = self.fusion(torch.cat(proj_feats, dim=1))
        logits = self.classifier(fused)
        return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

class RepViT(nn.Module):
    """
    RepViT: Revisiting Mobile CNN Architecture for Cellular Devices (CVPR 2024).
    Adapted with 4 input channels for remote sensing segmentation.
    """
    def __init__(self, in_channels=4, num_classes=2, dims=[48, 96, 192, 384], depths=[2, 2, 6, 2]):
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
        self.stage1 = nn.Sequential(*[RepViTBlock(dims[0]) for _ in range(depths[0])])

        # Stage 2: 1/8 resolution
        self.down1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.GELU()
        )
        self.stage2 = nn.Sequential(*[RepViTBlock(dims[1]) for _ in range(depths[1])])

        # Stage 3: 1/16 resolution
        self.down2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.GELU()
        )
        self.stage3 = nn.Sequential(*[RepViTBlock(dims[2]) for _ in range(depths[2])])

        # Stage 4: 1/32 resolution
        self.down3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[3]),
            nn.GELU()
        )
        self.stage4 = nn.Sequential(*[RepViTBlock(dims[3]) for _ in range(depths[3])])

        # Decoder Head
        self.head = RepViTSegHead(dims, out_channels=128, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        c1 = self.stage1(self.stem(x))
        c2 = self.stage2(self.down1(c1))
        c3 = self.stage3(self.down2(c2))
        c4 = self.stage4(self.down3(c3))
        
        logits = self.head([c1, c2, c3, c4], target_size=(h, w))
        return logits
