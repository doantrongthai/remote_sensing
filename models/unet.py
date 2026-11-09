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

class UNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, base_filters=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 8, base_filters * 16))
        
        self.up1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters * 16, base_filters * 8)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters * 8, base_filters * 4)
        
        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters * 4, base_filters * 2)
        
        self.up4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_filters * 2, base_filters)
        
        self.outc = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        d4 = self.up1(x5)
        d4 = self.conv_up1(torch.cat([x4, d4], dim=1))
        
        d3 = self.up2(d4)
        d3 = self.conv_up2(torch.cat([x3, d3], dim=1))
        
        d2 = self.up3(d3)
        d2 = self.conv_up3(torch.cat([x2, d2], dim=1))
        
        d1 = self.up4(d2)
        d1 = self.conv_up4(torch.cat([x1, d1], dim=1))
        
        logits = self.outc(d1)
        return logits
