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

class ProgressiveCrossAttentionBlock(nn.Module):
    """
    Progressive Cross-Attention Block as formalized in Feliren et al. (IEEE GRSL 2025):
    1. Self-attention on intermediate features {X_r, X_n}
    2. Cross-attention modulation between attended streams
    3. Element-wise additive fusion
    """
    def __init__(self, channels):
        super().__init__()
        # Self-attention convolution kernels (3x3 kernel with padding 1)
        self.conv_self_r = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_self_n = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Cross-attention convolution kernels
        self.conv_cross_r2n = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_cross_n2r = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x_r, x_n):
        # 1) Self-attention Stage
        a_r = torch.sigmoid(self.conv_self_r(x_r))
        a_n = torch.sigmoid(self.conv_self_n(x_n))
        x_hat_r = x_r * a_r
        x_hat_n = x_n * a_n
        
        # 2) Cross-attention Stage
        a_r2n = torch.sigmoid(self.conv_cross_r2n(x_hat_r))
        a_n2r = torch.sigmoid(self.conv_cross_n2r(x_hat_n))
        x_tilde_r = x_hat_r * a_n2r
        x_tilde_n = x_hat_n * a_r2n
        
        # 3) Cross Attended Fusion
        fused = x_tilde_r + x_tilde_n
        return fused

class ProCANet(nn.Module):
    """
    ProCANet (Progressive Cross Attention Network for Flood Segmentation).
    Architecture:
      - Encoder 1: RGB + NIR bands (4 channels)
      - Encoder 2: NIR band only (1 channel)
      - Progressive self/cross attention at each hierarchical scale
      - UNet-style decoder with fused skip connections
    Reference:
      Feliren et al., "Progressive Cross Attention Network for Flood Segmentation
      using Multispectral Satellite Imagery", IEEE GRSL 2025.
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=32):
        super().__init__()
        f = base_filters
        
        # Encoder 1: RGB + NIR
        self.enc1_conv0 = DoubleConv(in_channels, f)
        self.enc1_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.enc1_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.enc1_down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.enc1_down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))
        
        # Encoder 2: NIR only (1 channel)
        self.enc2_conv0 = DoubleConv(1, f)
        self.enc2_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.enc2_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.enc2_down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.enc2_down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))
        
        # Progressive Cross-Attention at each pooling stage
        self.pca0 = ProgressiveCrossAttentionBlock(f)
        self.pca1 = ProgressiveCrossAttentionBlock(f * 2)
        self.pca2 = ProgressiveCrossAttentionBlock(f * 4)
        self.pca3 = ProgressiveCrossAttentionBlock(f * 8)
        self.pca4 = ProgressiveCrossAttentionBlock(f * 16)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec_conv1 = DoubleConv(f * 16, f * 8)
        
        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec_conv2 = DoubleConv(f * 8, f * 4)
        
        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec_conv3 = DoubleConv(f * 4, f * 2)
        
        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec_conv4 = DoubleConv(f * 2, f)
        
        self.classifier = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        orig_sz = x.shape[-2:]
        # Split stream 1 (RGB+NIR) and stream 2 (NIR)
        # Assumes NIR is the last channel of the 4-band input
        if x.shape[1] >= 4:
            x_r = x[:, 0:4, :, :]
            x_n = x[:, 3:4, :, :]
        else:
            x_r = x
            x_n = x[:, -1:, :, :]
            
        r0 = self.enc1_conv0(x_r)
        n0 = self.enc2_conv0(x_n)
        f0 = self.pca0(r0, n0)
        
        r1 = self.enc1_down1(r0)
        n1 = self.enc2_down1(n0)
        f1 = self.pca1(r1, n1)
        
        r2 = self.enc1_down2(r1)
        n2 = self.enc2_down2(n1)
        f2 = self.pca2(r2, n2)
        
        r3 = self.enc1_down3(r2)
        n3 = self.enc2_down3(n2)
        f3 = self.pca3(r3, n3)
        
        r4 = self.enc1_down4(r3)
        n4 = self.enc2_down4(n3)
        f4 = self.pca4(r4, n4)
        
        d1 = self.up1(f4)
        if d1.shape[-2:] != f3.shape[-2:]:
            d1 = F.interpolate(d1, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec_conv1(torch.cat([d1, f3], dim=1))
        
        d2 = self.up2(d1)
        if d2.shape[-2:] != f2.shape[-2:]:
            d2 = F.interpolate(d2, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec_conv2(torch.cat([d2, f2], dim=1))
        
        d3 = self.up3(d2)
        if d3.shape[-2:] != f1.shape[-2:]:
            d3 = F.interpolate(d3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec_conv3(torch.cat([d3, f1], dim=1))
        
        d4 = self.up4(d3)
        if d4.shape[-2:] != f0.shape[-2:]:
            d4 = F.interpolate(d4, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        d4 = self.dec_conv4(torch.cat([d4, f0], dim=1))
        
        out = self.classifier(d4)
        if out.shape[-2:] != orig_sz:
            out = F.interpolate(out, size=orig_sz, mode='bilinear', align_corners=False)
        return out
