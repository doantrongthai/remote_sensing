import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

class UNetMobileNetV2(nn.Module):
    """
    U-Net with MobileNetV2 backbone as used in Feliren et al. (IEEE GRSL 2025 Table I).
    Accepts 4 channels (RGB + NIR) or any arbitrary channel count.
    """
    def __init__(self, in_channels=4, num_classes=2, pretrained=True):
        super().__init__()
        mb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        
        # Modify the first conv layer to accept in_channels
        orig_conv = mb.features[0][0]
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels, 
                orig_conv.out_channels, 
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride, 
                padding=orig_conv.padding, 
                bias=False
            )
            with torch.no_grad():
                if pretrained:
                    new_conv.weight[:, :min(3, in_channels), :, :] = orig_conv.weight[:, :min(3, in_channels), :, :]
                    if in_channels > 3:
                        for c in range(3, in_channels):
                            new_conv.weight[:, c:c+1, :, :] = orig_conv.weight.mean(dim=1, keepdim=True)
            mb.features[0][0] = new_conv
            
        self.stage0 = mb.features[0]      # H/2,  C=32
        self.stage1 = mb.features[1:3]    # H/4,  C=24
        self.stage2 = mb.features[3:7]    # H/8,  C=32
        self.stage3 = mb.features[7:14]   # H/16, C=96
        self.stage4 = mb.features[14:]    # H/32, C=1280
        
        # Decoder blocks with skip connections matching the feature stages
        self.up1 = nn.ConvTranspose2d(1280, 96, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(96 + 96, 96)
        
        self.up2 = nn.ConvTranspose2d(96, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(32 + 32, 32)
        
        self.up3 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(24 + 24, 24)
        
        self.up4 = nn.ConvTranspose2d(24, 32, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(32 + 32, 32)
        
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(16, 16)
        
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        orig_sz = x.shape[-2:]
        s0 = self.stage0(x)     # H/2
        s1 = self.stage1(s0)    # H/4
        s2 = self.stage2(s1)    # H/8
        s3 = self.stage3(s2)    # H/16
        s4 = self.stage4(s3)    # H/32
        
        d1 = self.up1(s4)
        if d1.shape[-2:] != s3.shape[-2:]:
            d1 = F.interpolate(d1, size=s3.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, s3], dim=1))
        
        d2 = self.up2(d1)
        if d2.shape[-2:] != s2.shape[-2:]:
            d2 = F.interpolate(d2, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        
        d3 = self.up3(d2)
        if d3.shape[-2:] != s1.shape[-2:]:
            d3 = F.interpolate(d3, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, s1], dim=1))
        
        d4 = self.up4(d3)
        if d4.shape[-2:] != s0.shape[-2:]:
            d4 = F.interpolate(d4, size=s0.shape[-2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, s0], dim=1))
        
        d5 = self.up5(d4)
        d5 = self.dec5(d5)
        
        out = self.classifier(d5)
        if out.shape[-2:] != orig_sz:
            out = F.interpolate(out, size=orig_sz, mode='bilinear', align_corners=False)
        return out
