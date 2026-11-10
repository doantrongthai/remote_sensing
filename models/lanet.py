import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class LocalAttentionModule(nn.Module):
    """
    Local Attention Module (LAM) from LANet (IEEE TGRS 2021).
    Bridges low-level spatial detail features with high-level semantic context
    using local spatial and channel attention mechanisms tailored for remote sensing.
    """
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.high_proj = BasicConv(high_channels, out_channels, kernel_size=1)
        self.low_proj = BasicConv(low_channels, out_channels, kernel_size=1)

        # Local Spatial Attention
        self.spatial_att = nn.Sequential(
            BasicConv(out_channels * 2, out_channels // 2, kernel_size=3, padding=1),
            BasicConv(out_channels // 2, 1, kernel_size=3, padding=1, relu=False, bn=False),
            nn.Sigmoid()
        )

        # Local Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse_conv = BasicConv(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, low_feat, high_feat):
        h, w = low_feat.shape[2:]
        high_proj = self.high_proj(high_feat)
        if high_proj.shape[2:] != (h, w):
            high_proj = F.interpolate(high_proj, size=(h, w), mode='bilinear', align_corners=True)
            
        low_proj = self.low_proj(low_feat)
        concat = torch.cat([low_proj, high_proj], dim=1)

        # Spatial and Channel Attention modulation
        s_att = self.spatial_att(concat)
        c_att = self.channel_att(concat)
        
        modulated = low_proj * s_att * c_att
        out = self.fuse_conv(modulated + high_proj)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = BasicConv(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = BasicConv(planes, planes, kernel_size=3, stride=1, padding=1, relu=False)
        self.shortcut = nn.Identity() if stride == 1 and in_planes == planes else BasicConv(in_planes, planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))

class LANet(nn.Module):
    """
    LANet: Local Attention Network for Semantic Segmentation in Remote Sensing (IEEE TGRS 2021).
    Local Attention Module (LAM) for bridging low-level features and high-level semantic priors.
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        f = base_filters

        # Backbone
        self.stem = nn.Sequential(
            BasicConv(in_channels, f, kernel_size=3, stride=2, padding=1),
            BasicConv(f, f, kernel_size=3, stride=1, padding=1)
        )
        
        # Stages: 1/2, 1/4, 1/8, 1/16
        self.stage1 = nn.Sequential(ResBlock(f, f, stride=1), ResBlock(f, f, stride=1))          # 1/2
        self.stage2 = nn.Sequential(ResBlock(f, f * 2, stride=2), ResBlock(f * 2, f * 2))        # 1/4 (C2)
        self.stage3 = nn.Sequential(ResBlock(f * 2, f * 4, stride=2), ResBlock(f * 4, f * 4))    # 1/8 (C3)
        self.stage4 = nn.Sequential(ResBlock(f * 4, f * 8, stride=2), ResBlock(f * 8, f * 8))    # 1/16 (C4)

        # High-level Semantic Enhancement Module (Global context)
        self.semantic_enhancer = nn.Sequential(
            BasicConv(f * 8, f * 8, kernel_size=3, padding=1),
            BasicConv(f * 8, f * 4, kernel_size=1)
        )

        # Local Attention Modules
        self.lam3 = LocalAttentionModule(low_channels=f * 4, high_channels=f * 4, out_channels=f * 4)
        self.lam2 = LocalAttentionModule(low_channels=f * 2, high_channels=f * 4, out_channels=f * 2)

        # Final Decoder / Classifier
        self.head = nn.Sequential(
            BasicConv(f * 2, f * 2, kernel_size=3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(f * 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # Backbone
        x0 = self.stem(x)
        c1 = self.stage1(x0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        # Semantic context
        high_sem = self.semantic_enhancer(c4)

        # Local Attention bridge
        fused_3 = self.lam3(c3, high_sem)
        fused_2 = self.lam2(c2, fused_3)

        # Logits
        logits = self.head(fused_2)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
