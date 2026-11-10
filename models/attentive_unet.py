import torch
import torch.nn as nn
import torch.nn.functional as F

class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) block as utilized in:
    Deep attentive fusion network for flood detection on uni-temporal Sentinel-1 data
    (Ritu Yadav et al., Frontiers in Remote Sensing 2022).
    Official repo: https://github.com/RituYadav92/Uni-Temporal-Flood-Detection-Sentinel-1_Frontiers22
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

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

class AttentiveUNet(nn.Module):
    """
    Attentive U-Net for Sentinel-1 Flood Segmentation (Yadav et al., Frontiers 2022).
    Applies spatial and channel attention across multi-scale skip connections and stages.
    """
    def __init__(self, in_channels=2, num_classes=2, base_filters=32):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(in_channels, f)
        self.att0 = SCSEModule(f)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.att1 = SCSEModule(f * 2)
        
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.att2 = SCSEModule(f * 4)
        
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.att3 = SCSEModule(f * 8)
        
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))
        self.att4 = SCSEModule(f * 16)
        
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(f * 16, f * 8)
        self.dec_att1 = SCSEModule(f * 8)
        
        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(f * 8, f * 4)
        self.dec_att2 = SCSEModule(f * 4)
        
        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(f * 4, f * 2)
        self.dec_att3 = SCSEModule(f * 2)
        
        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(f * 2, f)
        self.dec_att4 = SCSEModule(f)
        
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.att0(self.inc(x))
        x1 = self.att1(self.down1(x0))
        x2 = self.att2(self.down2(x1))
        x3 = self.att3(self.down3(x2))
        x4 = self.att4(self.down4(x3))
        
        d3 = self.up1(x4)
        d3 = self.dec_att1(self.conv_up1(torch.cat([x3, d3], dim=1)))
        
        d2 = self.up2(d3)
        d2 = self.dec_att2(self.conv_up2(torch.cat([x2, d2], dim=1)))
        
        d1 = self.up3(d2)
        d1 = self.dec_att3(self.conv_up3(torch.cat([x1, d1], dim=1)))
        
        d0 = self.up4(d1)
        d0 = self.dec_att4(self.conv_up4(torch.cat([x0, d0], dim=1)))
        
        return self.outc(d0)
