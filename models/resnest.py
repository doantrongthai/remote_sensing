import torch
import torch.nn as nn
import torch.nn.functional as F

class SplAtConv2d(nn.Module):
    """
    Split-Attention Conv2d (SplAtConv2d) from ResNeSt (CVPRW 2022).
    Applies split-attention across radix groups with rSoftmax.
    """
    def __init__(self, in_channels, channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, radix=2, reduction_factor=4):
        super().__init__()
        self.radix = radix
        self.channels = channels
        self.groups = groups
        self.stride = stride

        self.conv = nn.Conv2d(
            in_channels, channels * radix, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups * radix, bias=False
        )
        self.bn0 = nn.BatchNorm2d(channels * radix)
        self.relu = nn.ReLU(inplace=True)

        inter_channels = max(16, channels * radix // reduction_factor)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=groups)

    def forward(self, x):
        x = self.relu(self.bn0(self.conv(x)))
        B, rC, H, W = x.shape
        C = self.channels

        if self.radix > 1:
            splits = torch.split(x, C, dim=1)
            # Elementwise sum across splits
            gap = sum(splits)
        else:
            gap = x

        # Global context pooling & channel excitation
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.relu(self.bn1(self.fc1(gap)))
        atten = self.fc2(gap)

        if self.radix > 1:
            # Reshape for rSoftmax: [B, radix, C, 1, 1]
            atten = atten.view(B, self.radix, C, 1, 1)
            atten = F.softmax(atten, dim=1)
            # Softmax-weighted aggregation
            out = sum(splits[r] * atten[:, r, :, :, :] for r in range(self.radix))
        else:
            atten = torch.sigmoid(atten)
            out = x * atten

        return out

class ResNeStBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, radix=2, cardinality=1,
                 reduction_factor=4, downsample=None):
        super().__init__()
        mid_planes = planes * cardinality
        
        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        
        self.splat = SplAtConv2d(
            mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1,
            radix=radix, groups=cardinality, reduction_factor=reduction_factor
        )
        
        self.conv2 = nn.Conv2d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.splat(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class ResNeSt(nn.Module):
    """
    ResNeSt: Split-Attention Networks (CVPRW 2022).
    Bottleneck blocks with Split-Attention and multi-scale segmentation decoder.
    """
    def __init__(self, in_channels=4, num_classes=2, base_planes=32, radix=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        p = base_planes

        # Deep stem: 3 consecutive 3x3 convs
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, p, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(p),
            nn.ReLU(inplace=True),
            nn.Conv2d(p, p, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(p),
            nn.ReLU(inplace=True),
            nn.Conv2d(p, p * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(p * 2),
            nn.ReLU(inplace=True)
        )

        # Stage 1 (1/4 resolution)
        ds1 = nn.Sequential(
            nn.Conv2d(p * 2, p * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(p * 4)
        )
        self.layer1 = nn.Sequential(
            ResNeStBottleneck(p * 2, p, stride=1, radix=radix, downsample=ds1),
            ResNeStBottleneck(p * 4, p, stride=1, radix=radix)
        )

        # Stage 2 (1/8 resolution)
        ds2 = nn.Sequential(
            nn.Conv2d(p * 4, p * 8, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(p * 8)
        )
        self.layer2 = nn.Sequential(
            ResNeStBottleneck(p * 4, p * 2, stride=2, radix=radix, downsample=ds2),
            ResNeStBottleneck(p * 8, p * 2, stride=1, radix=radix)
        )

        # Stage 3 (1/16 resolution)
        ds3 = nn.Sequential(
            nn.Conv2d(p * 8, p * 16, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(p * 16)
        )
        self.layer3 = nn.Sequential(
            ResNeStBottleneck(p * 8, p * 4, stride=2, radix=radix, downsample=ds3),
            ResNeStBottleneck(p * 16, p * 4, stride=1, radix=radix)
        )

        # Stage 4 (1/32 resolution)
        ds4 = nn.Sequential(
            nn.Conv2d(p * 16, p * 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(p * 32)
        )
        self.layer4 = nn.Sequential(
            ResNeStBottleneck(p * 16, p * 8, stride=2, radix=radix, downsample=ds4),
            ResNeStBottleneck(p * 32, p * 8, stride=1, radix=radix)
        )

        # Decoder Head: Multi-scale fusion
        head_dim = 128
        self.p1 = nn.Sequential(nn.Conv2d(p * 4, head_dim, 1, bias=False), nn.BatchNorm2d(head_dim), nn.ReLU(inplace=True))
        self.p2 = nn.Sequential(nn.Conv2d(p * 8, head_dim, 1, bias=False), nn.BatchNorm2d(head_dim), nn.ReLU(inplace=True))
        self.p3 = nn.Sequential(nn.Conv2d(p * 16, head_dim, 1, bias=False), nn.BatchNorm2d(head_dim), nn.ReLU(inplace=True))
        self.p4 = nn.Sequential(nn.Conv2d(p * 32, head_dim, 1, bias=False), nn.BatchNorm2d(head_dim), nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            nn.Conv2d(head_dim * 4, head_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(head_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        x0 = self.stem(x)
        c1 = self.layer1(x0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        ref_size = c1.shape[2:]
        f1 = self.p1(c1)
        f2 = F.interpolate(self.p2(c2), size=ref_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.p3(c3), size=ref_size, mode='bilinear', align_corners=False)
        f4 = F.interpolate(self.p4(c4), size=ref_size, mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([f1, f2, f3, f4], dim=1))
        return F.interpolate(fused, size=(h, w), mode='bilinear', align_corners=False)
