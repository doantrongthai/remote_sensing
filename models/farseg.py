import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FSRM(nn.Module):
    """
    Foreground-Scene Relation Module (FSRM) as introduced in:
    Foreground-Aware Relation Network for Geospatial Object Segmentation (CVPR 2020 / TPAMI 2023).
    Official repo: https://github.com/Z-Zheng/FarSeg
    """
    def __init__(self, in_channels, scene_channels):
        super().__init__()
        self.conv_p = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_s = nn.Conv2d(scene_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, p, scene_feat):
        s = torch.sigmoid(self.conv_s(scene_feat))
        feat = self.conv_p(p) * s
        return p + self.gamma * feat

class FarSeg(nn.Module):
    """
    Foreground-Aware Relation Network (FarSeg) for Remote Sensing Segmentation.
    """
    def __init__(self, in_channels=2, num_classes=2, base_filters=32):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))     # C2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4)) # C3
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8)) # C4
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))# C5
        
        fpn_dim = f * 4
        self.lat5 = nn.Conv2d(f * 16, fpn_dim, 1)
        self.lat4 = nn.Conv2d(f * 8, fpn_dim, 1)
        self.lat3 = nn.Conv2d(f * 4, fpn_dim, 1)
        self.lat2 = nn.Conv2d(f * 2, fpn_dim, 1)
        
        self.scene_emb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(f * 16, fpn_dim, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.fsrm5 = FSRM(fpn_dim, fpn_dim)
        self.fsrm4 = FSRM(fpn_dim, fpn_dim)
        self.fsrm3 = FSRM(fpn_dim, fpn_dim)
        self.fsrm2 = FSRM(fpn_dim, fpn_dim)
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        x0 = self.inc(x)
        c2 = self.down1(x0)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        c5 = self.down4(c4)
        
        scene = self.scene_emb(c5)
        
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        
        p5 = self.fsrm5(p5, scene)
        p4 = self.fsrm4(p4, scene)
        p3 = self.fsrm3(p3, scene)
        p2 = self.fsrm2(p2, scene)
        
        fused = p2 + F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=True) + \
                F.interpolate(p4, size=p2.shape[2:], mode='bilinear', align_corners=True) + \
                F.interpolate(p5, size=p2.shape[2:], mode='bilinear', align_corners=True)
                
        out = self.fuse_conv(fused)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return self.classifier(out)
