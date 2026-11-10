import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)

class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block:
    1x1 Conv (channel reduction) -> Transposed Conv 3x3 (upsample) -> 1x1 Conv (channel projection)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 4
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class LinkNet(nn.Module):
    """
    LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation (VCIP 2017).
    ResNet-style encoder directly linked into transposed convolutional decoders via elementwise additions.
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        f = base_filters

        # Initial Stem (1/4 resolution)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Encoder stages
        self.encoder1 = nn.Sequential(BasicBlock(f, f, stride=1), BasicBlock(f, f, stride=1))          # 1/4
        self.encoder2 = nn.Sequential(BasicBlock(f, f * 2, stride=2), BasicBlock(f * 2, f * 2))        # 1/8
        self.encoder3 = nn.Sequential(BasicBlock(f * 2, f * 4, stride=2), BasicBlock(f * 4, f * 4))    # 1/16
        self.encoder4 = nn.Sequential(BasicBlock(f * 4, f * 8, stride=2), BasicBlock(f * 8, f * 8))    # 1/32

        # Decoder stages (inverting encoder)
        self.decoder4 = DecoderBlock(f * 8, f * 4)  # 1/32 -> 1/16
        self.decoder3 = DecoderBlock(f * 4, f * 2)  # 1/16 -> 1/8
        self.decoder2 = DecoderBlock(f * 2, f)      # 1/8 -> 1/4
        self.decoder1 = nn.Sequential(              # 1/4 -> 1/4
            nn.Conv2d(f, f // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(f // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(f // 4, f // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(f // 4, f, kernel_size=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True)
        )

        # Final restoration to full resolution (1/4 -> 1/1)
        self.final_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(f, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_deconv2 = nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        h, w = x.shape[2:]

        # Initial stem
        x0 = self.in_conv(x)

        # Encoder
        e1 = self.encoder1(x0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with direct residual linking
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Full resolution output
        y = self.final_deconv1(d1)
        y = self.final_conv(y)
        logits = self.final_deconv2(y)

        if logits.shape[2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
        return logits
