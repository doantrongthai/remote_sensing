import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1):
        super().__init__(
            nn.Conv2d(in_chan, out_chan, ks, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

class DetailBranch(nn.Module):
    """Detail Branch captures low-level spatial details with wide shallow layers"""
    def __init__(self, in_channels=4):
        super().__init__()
        # Stage 1: 1/2 resolution
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1)
        )
        # Stage 2: 1/4 resolution
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1),
            ConvBNReLU(64, 64, 3, stride=1, padding=1)
        )
        # Stage 3: 1/8 resolution
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2, padding=1),
            ConvBNReLU(128, 128, 3, stride=1, padding=1),
            ConvBNReLU(128, 128, 3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class StemBlock(nn.Module):
    """Stem Block of Semantic Branch"""
    def __init__(self, in_channels=4, out_channels=16):
        super().__init__()
        self.conv_in = ConvBNReLU(in_channels, out_channels, 3, stride=2, padding=1)
        self.branch_conv = nn.Sequential(
            ConvBNReLU(out_channels, out_channels // 2, 1, stride=1, padding=0),
            ConvBNReLU(out_channels // 2, out_channels, 3, stride=2, padding=1)
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(out_channels * 2, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        b1 = self.branch_conv(x)
        b2 = self.branch_pool(x)
        return self.conv_out(torch.cat([b1, b2], dim=1))

class GELayer(nn.Module):
    """Gather-and-Expansion Layer (GE Block) for Semantic Branch"""
    def __init__(self, in_chan, out_chan, stride=1, exp=6):
        super().__init__()
        self.stride = stride
        mid_chan = in_chan * exp

        if stride == 1:
            self.conv = nn.Sequential(
                ConvBNReLU(in_chan, mid_chan, 3, stride=1, padding=1),
                nn.Conv2d(mid_chan, mid_chan, 3, stride=1, padding=1, groups=mid_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_chan, out_chan, 1, bias=False),
                nn.BatchNorm2d(out_chan)
            )
            self.shortcut = nn.Identity() if in_chan == out_chan else ConvBNReLU(in_chan, out_chan, 1, stride=1, padding=0)
        else:
            self.conv = nn.Sequential(
                ConvBNReLU(in_chan, mid_chan, 3, stride=1, padding=1),
                nn.Conv2d(mid_chan, mid_chan, 3, stride=stride, padding=1, groups=mid_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.Conv2d(mid_chan, mid_chan, 3, stride=1, padding=1, groups=mid_chan, bias=False),
                nn.BatchNorm2d(mid_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_chan, out_chan, 1, bias=False),
                nn.BatchNorm2d(out_chan)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, 3, stride=stride, padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(in_chan, out_chan, 1, bias=False),
                nn.BatchNorm2d(out_chan)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

class ContextEmbeddingBlock(nn.Module):
    """Context Embedding Block at the end of Semantic Branch"""
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            ConvBNReLU(in_chan, out_chan, 1, stride=1, padding=0),
            nn.Conv2d(out_chan, out_chan, 1, bias=True),
            nn.Sigmoid()
        )
        self.conv = ConvBNReLU(in_chan, out_chan, 3, stride=1, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        w = self.fc(self.pool(x))
        return feat * w + feat

class SemanticBranch(nn.Module):
    """Semantic Branch captures global contextual semantics"""
    def __init__(self, in_channels=4):
        super().__init__()
        # Stem: 1/4 resolution, 16 channels
        self.stem = StemBlock(in_channels, 16)
        
        # Stage 3: 1/8 resolution, 32 channels
        self.stage3 = nn.Sequential(
            GELayer(16, 32, stride=2, exp=6),
            GELayer(32, 32, stride=1, exp=6)
        )
        # Stage 4: 1/16 resolution, 64 channels
        self.stage4 = nn.Sequential(
            GELayer(32, 64, stride=2, exp=6),
            GELayer(64, 64, stride=1, exp=6)
        )
        # Stage 5: 1/32 resolution, 128 channels
        self.stage5 = nn.Sequential(
            GELayer(64, 128, stride=2, exp=6),
            GELayer(128, 128, stride=1, exp=6),
            GELayer(128, 128, stride=1, exp=6),
            GELayer(128, 128, stride=1, exp=6)
        )
        # CE Block
        self.ce = ContextEmbeddingBlock(128, 128)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.ce(x)
        return x

class BGALayer(nn.Module):
    """Bilateral Guided Aggregation (BGA) Layer"""
    def __init__(self, in_chan=128):
        super().__init__()
        # Detail branch transformations
        self.detail_dw = nn.Sequential(
            ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1),
            nn.Conv2d(in_chan, in_chan, 3, stride=1, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan)
        )
        self.detail_down = nn.Sequential(
            ConvBNReLU(in_chan, in_chan, 3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Semantic branch transformations
        self.sem_up = nn.Sequential(
            ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.sem_dw = nn.Sequential(
            ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1),
            nn.Conv2d(in_chan, in_chan, 3, stride=1, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Sigmoid()
        )

        self.conv_out = ConvBNReLU(in_chan, in_chan, 3, stride=1, padding=1)

    def forward(self, detail, sem):
        h, w = detail.shape[2:]
        
        # Detail guidance
        det_dw = self.detail_dw(detail)
        sem_sig_up = F.interpolate(self.sem_up(sem), size=(h, w), mode='bilinear', align_corners=False)
        det_out = det_dw * sem_sig_up
        
        # Semantic guidance
        det_down = self.detail_down(detail)
        sem_sig = self.sem_dw(sem)
        if det_down.shape[2:] != sem.shape[2:]:
            det_down = F.interpolate(det_down, size=sem.shape[2:], mode='bilinear', align_corners=False)
        sem_out = det_down * sem_sig
        sem_out = F.interpolate(sem_out, size=(h, w), mode='bilinear', align_corners=False)
        
        return self.conv_out(det_out + sem_out)

class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_chan, mid_chan, 3, stride=1, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(mid_chan, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)

class BiSeNetV2(nn.Module):
    """
    BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic Segmentation (IJCV 2021).
    Detail Branch + Semantic Branch + Bilateral Guided Aggregation (BGA).
    """
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.detail = DetailBranch(in_channels=in_channels)
        self.semantic = SemanticBranch(in_channels=in_channels)
        self.bga = BGALayer(in_chan=128)
        self.head = SegmentHead(128, 128, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        feat_detail = self.detail(x)
        feat_sem = self.semantic(x)
        fused = self.bga(feat_detail, feat_sem)
        logits = self.head(fused)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
