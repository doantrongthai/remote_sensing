import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) from PSPNet (CVPR 2017).
    Pools contextual representations at bin sizes [1x1, 2x2, 3x3, 6x6].
    """
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])

    def forward(self, x):
        h, w = x.shape[2:]
        pyramids = [x]
        for stage in self.stages:
            p = stage(x)
            pyramids.append(F.interpolate(p, size=(h, w), mode='bilinear', align_corners=True))
        return torch.cat(pyramids, dim=1)

class PSPNet(nn.Module):
    """
    PSPNet: Pyramid Scene Parsing Network (CVPR 2017).
    Dilated ResNet backbone with Pyramid Pooling Module (PPM).
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=64, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        f = base_filters

        # Stem (stride 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True)
        )

        # Stage 1: stride 1 (1/2 resolution)
        self.layer1 = nn.Sequential(
            BasicBlock(f, f, stride=1),
            BasicBlock(f, f, stride=1)
        )
        # Stage 2: stride 2 (1/4 resolution)
        self.layer2 = nn.Sequential(
            BasicBlock(f, f * 2, stride=2),
            BasicBlock(f * 2, f * 2, stride=1)
        )
        # Stage 3: stride 2 (1/8 resolution)
        self.layer3 = nn.Sequential(
            BasicBlock(f * 2, f * 4, stride=2),
            BasicBlock(f * 4, f * 4, stride=1)
        )
        # Stage 4: stride 1, dilation 2 (1/8 resolution)
        self.layer4 = nn.Sequential(
            BasicBlock(f * 4, f * 8, stride=1, dilation=2),
            BasicBlock(f * 8, f * 8, stride=1, dilation=2)
        )

        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(f * 8, pool_sizes=pool_sizes)

        # Classification Head
        ppm_channels = f * 8 * 2  # f * 8 + 4 * (f * 8 / 4)
        self.cls = nn.Sequential(
            nn.Conv2d(ppm_channels, f * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(f * 4, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        x0 = self.stem(x)
        c1 = self.layer1(x0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        ppm_out = self.ppm(c4)
        logits = self.cls(ppm_out)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
