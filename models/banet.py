import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, d=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class DependencyPath(nn.Module):
    """
    Dependency Path in BANet: Captures long-range spatial dependencies and semantic context.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = out_channels // 2
        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.key = nn.Conv2d(in_channels, inter_channels, 1)
        self.value = nn.Conv2d(in_channels, out_channels, 1)
        self.proj = ConvBNReLU(out_channels, out_channels, 1, p=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        K = self.key(x).view(B, -1, H * W)                     # [B, C', HW]
        V = self.value(x).view(B, -1, H * W)                   # [B, C, HW]

        energy = torch.bmm(Q, K)
        attn = F.softmax(energy, dim=-1)

        out = torch.bmm(V, attn.permute(0, 2, 1)).view(B, -1, H, W)
        return self.proj(self.gamma * out + self.value(x))

class BoundaryPath(nn.Module):
    """
    Boundary Path in BANet: Extracts fine edge and boundary details using multi-scale edge filtering.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, p=1)
        # Multi-scale boundary gradient convolutions
        self.dconv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.dconv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = ConvBNReLU(out_channels, out_channels, 1, p=0)

    def forward(self, x):
        f = self.conv1(x)
        edge = self.bn(self.dconv1(f) - self.dconv2(f))
        out = self.relu(edge + f)
        return self.conv_out(out)

class BilateralAwarenessModule(nn.Module):
    """
    Bilateral Awareness Module (BAM) in BANet:
    Cross-modulates contextual dependency features with sharp boundary guidance and vice versa.
    """
    def __init__(self, channels):
        super().__init__()
        self.bnd_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.dep_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.dep_refine = ConvBNReLU(channels, channels, 3, p=1)
        self.bnd_refine = ConvBNReLU(channels, channels, 3, p=1)
        self.fuse = ConvBNReLU(channels * 2, channels, 3, p=1)

    def forward(self, dep_feat, bnd_feat):
        # Mutual bilateral guidance
        g_bnd = self.bnd_gate(bnd_feat)
        g_dep = self.dep_gate(dep_feat)

        dep_enhanced = self.dep_refine(dep_feat * g_bnd + dep_feat)
        bnd_enhanced = self.bnd_refine(bnd_feat * g_dep + bnd_feat)

        return self.fuse(torch.cat([dep_enhanced, bnd_enhanced], dim=1))

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, out_c, 3, s=s, p=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Identity() if s == 1 and in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=s, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))

class BANet(nn.Module):
    """
    BANet: Bilateral Awareness Network for Semantic Segmentation in RS (IEEE TGRS 2021).
    Dependency Path (Contextual Semantics) + Boundary Path (Detail Awareness) + Bilateral Awareness Module.
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        f = base_filters

        # Backbone
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, f, 3, s=2, p=1),
            ConvBNReLU(f, f, 3, s=1, p=1)
        )
        # Stages: 1/2, 1/4, 1/8, 1/16
        self.stage1 = nn.Sequential(ResBlock(f, f, s=1), ResBlock(f, f, s=1))          # 1/2
        self.stage2 = nn.Sequential(ResBlock(f, f * 2, s=2), ResBlock(f * 2, f * 2))  # 1/4 (Boundary source)
        self.stage3 = nn.Sequential(ResBlock(f * 2, f * 4, s=2), ResBlock(f * 4, f * 4)) # 1/8
        self.stage4 = nn.Sequential(ResBlock(f * 4, f * 8, s=2), ResBlock(f * 8, f * 8)) # 1/16 (Dependency source)

        # Bilateral Branches
        mid_dim = f * 4
        self.dep_path = DependencyPath(f * 8, mid_dim)
        self.bnd_path = BoundaryPath(f * 2, mid_dim)

        # Bilateral Awareness Module
        self.bam = BilateralAwarenessModule(mid_dim)

        # Decoder Head
        self.classifier = nn.Sequential(
            ConvBNReLU(mid_dim, mid_dim, 3, p=1),
            nn.Dropout(0.1),
            nn.Conv2d(mid_dim, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        x0 = self.stem(x)
        c1 = self.stage1(x0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        # Dependency Path
        dep = self.dep_path(c4)
        # Boundary Path
        bnd = self.bnd_path(c2)

        # Align resolutions to 1/4 scale
        if dep.shape[2:] != bnd.shape[2:]:
            dep = F.interpolate(dep, size=bnd.shape[2:], mode='bilinear', align_corners=True)

        # Bilateral Awareness Fusion
        fused = self.bam(dep, bnd)

        # Classification
        logits = self.classifier(fused)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
