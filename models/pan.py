import torch
import torch.nn as nn
import torch.nn.functional as F

class FPA(nn.Module):
    """
    Feature Pyramid Attention (FPA) module (Li et al., 2018).
    Combines multi-scale pyramid convolutions with global average pooling.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.mid = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p3_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p2_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p1_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        b1 = F.interpolate(self.branch1(x), size=(h, w), mode='bilinear', align_corners=True)
        mid = self.mid(x)
        
        feat1 = self.p1(x)
        feat2 = self.p2(feat1)
        feat3 = self.p3(feat2)
        
        u3 = F.interpolate(self.p3_conv(feat3), size=feat2.shape[2:], mode='bilinear', align_corners=True) + feat2
        u2 = F.interpolate(self.p2_conv(u3), size=feat1.shape[2:], mode='bilinear', align_corners=True) + feat1
        u1 = F.interpolate(self.p1_conv(u2), size=(h, w), mode='bilinear', align_corners=True)
        
        return mid * u1 + b1

class GAU(nn.Module):
    """
    Global Attention Upsample (GAU) module (Li et al., 2018).
    Uses high-level global context to guide low-level feature selection.
    """
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.conv_high_proj = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, out_channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat):
        h, w = low_feat.shape[2:]
        high_up = F.interpolate(self.conv_high_proj(high_feat), size=(h, w), mode='bilinear', align_corners=True)
        gate = self.gate_conv(high_feat)
        low_att = self.conv_low(low_feat) * gate
        return high_up + low_att

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class PAN(nn.Module):
    """
    Pyramid Attention Network for Semantic Segmentation (Li et al., 2018).
    """
    def __init__(self, in_channels=2, num_classes=2, base_filters=32):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))
        
        self.fpa = FPA(f * 16, f * 8)
        self.gau3 = GAU(f * 8, f * 8, f * 4)
        self.gau2 = GAU(f * 4, f * 4, f * 2)
        self.gau1 = GAU(f * 2, f * 2, f)
        self.gau0 = GAU(f, f, f)
        
        self.classifier = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        fpa_out = self.fpa(x4)
        d3 = self.gau3(x3, fpa_out)
        d2 = self.gau2(x2, d3)
        d1 = self.gau1(x1, d2)
        d0 = self.gau0(x0, d1)
        
        return self.classifier(d0)
