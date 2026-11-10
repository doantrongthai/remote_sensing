import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class _DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.dw(x)))
        x = self.relu2(self.bn2(self.pw(x)))
        return x

class _LinearBottleneck(nn.Module):
    """Inverted Bottleneck Block used in Fast-SCNN / MobileNetV2"""
    def __init__(self, in_channels, out_channels, stride=1, t=6):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * t
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        return self.conv(x)

class _PyramidPooling(nn.Module):
    """Pyramid Pooling Module in Fast-SCNN GFE"""
    def __init__(self, in_channels, out_channels=128, pools=(1, 2, 3, 6)):
        super().__init__()
        self.pools = pools
        sub_c = in_channels // len(pools)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(p),
                nn.Conv2d(in_channels, sub_c, 1, bias=False),
                nn.BatchNorm2d(sub_c),
                nn.ReLU(inplace=True)
            ) for p in pools
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + sub_c * len(pools), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        pyramids = [x]
        for stage in self.stages:
            p = stage(x)
            pyramids.append(F.interpolate(p, size=(h, w), mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(pyramids, dim=1))

class LearningToDownsample(nn.Module):
    """First branch of Fast-SCNN: fast resolution reduction preserving spatial cues"""
    def __init__(self, in_channels=4, dw_channels1=32, dw_channels2=48, out_channels=64):
        super().__init__()
        self.conv = _ConvBNReLU(in_channels, dw_channels1, kernel_size=3, stride=2, padding=1)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, stride=2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class GlobalFeatureExtractor(nn.Module):
    """Second branch of Fast-SCNN: deep bottleneck representation with PPM"""
    def __init__(self, in_channels=64, block_channels=(64, 96, 128), t=6):
        super().__init__()
        # 1/16 resolution
        self.block1 = nn.Sequential(
            _LinearBottleneck(in_channels, block_channels[0], stride=2, t=t),
            _LinearBottleneck(block_channels[0], block_channels[0], stride=1, t=t),
            _LinearBottleneck(block_channels[0], block_channels[0], stride=1, t=t)
        )
        # 1/32 resolution
        self.block2 = nn.Sequential(
            _LinearBottleneck(block_channels[0], block_channels[1], stride=2, t=t),
            _LinearBottleneck(block_channels[1], block_channels[1], stride=1, t=t),
            _LinearBottleneck(block_channels[1], block_channels[1], stride=1, t=t)
        )
        # 1/32 resolution continued
        self.block3 = nn.Sequential(
            _LinearBottleneck(block_channels[1], block_channels[2], stride=1, t=t),
            _LinearBottleneck(block_channels[2], block_channels[2], stride=1, t=t),
            _LinearBottleneck(block_channels[2], block_channels[2], stride=1, t=t)
        )
        self.ppm = _PyramidPooling(block_channels[2], out_channels=128)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.ppm(x)

class FeatureFusionModule(nn.Module):
    """Fuses high-res details (LDS) with low-res semantics (GFE)"""
    def __init__(self, high_channels=64, low_channels=128, out_channels=128):
        super().__init__()
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, 3, padding=1, groups=low_channels, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high, low):
        h, w = high.shape[2:]
        low_up = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
        return self.relu(self.high_conv(high) + self.low_conv(low_up))

class FastSCNN(nn.Module):
    """
    Fast-SCNN: Fast Semantic Segmentation Network (BMVC 2019).
    Learning to Downsample + Global Feature Extractor + Feature Fusion Module.
    """
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.learning_to_downsample = LearningToDownsample(in_channels=in_channels, dw_channels1=32, dw_channels2=48, out_channels=64)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels=64, block_channels=(64, 96, 128))
        self.feature_fusion = FeatureFusionModule(high_channels=64, low_channels=128, out_channels=128)
        
        self.classifier = nn.Sequential(
            _DSConv(128, 128, stride=1),
            _DSConv(128, 128, stride=1),
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        lds = self.learning_to_downsample(x)
        gfe = self.global_feature_extractor(lds)
        fused = self.feature_fusion(lds, gfe)
        logits = self.classifier(fused)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
