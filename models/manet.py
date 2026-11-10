import torch
import torch.nn as nn
import torch.nn.functional as F

class PAM_Module(nn.Module):
    """
    Position Attention Module (PAM) with linear kernel attention as introduced in:
    Multiattention Network for Semantic Segmentation of Fine-Resolution Remote Sensing Images
    (Ronghao Li et al., IEEE TGRS 2021).
    Official repo: https://github.com/lironui/Multi-Attention-Network
    """
    def __init__(self, in_places, scale=8, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.scale = max(1, scale)
        mid_ch = max(1, in_places // self.scale)
        self.eps = eps
        self.query_conv = nn.Conv2d(in_places, mid_ch, kernel_size=1)
        self.key_conv = nn.Conv2d(in_places, mid_ch, kernel_size=1)
        self.value_conv = nn.Conv2d(in_places, in_places, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.query_conv(x).view(B, -1, H * W)
        K = self.key_conv(x).view(B, -1, H * W)
        V = self.value_conv(x).view(B, -1, H * W)

        Q = F.softplus(Q).permute(0, 2, 1)  # [B, HW, C']
        K = F.softplus(K)                  # [B, C', HW]

        KV = torch.einsum('bmn, bcn->bmc', K, V)
        norm = 1.0 / (torch.einsum('bnm, bm->bn', Q, K.sum(dim=-1)) + self.eps)
        weight = torch.einsum('bnm, bmc->bcn', Q, KV)
        out = weight * norm.unsqueeze(1)
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class CAM_Module(nn.Module):
    """
    Channel Attention Module (CAM) as utilized in MANet / DANet for remote sensing.
    """
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(B, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.pam = PAM_Module(in_ch)
        self.cam = CAM_Module()

    def forward(self, x):
        return self.pam(x) + self.cam(x)

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

class MANet(nn.Module):
    """
    Multi-Attention Network for Remote Sensing Segmentation (IEEE TGRS 2021).
    Flexible for arbitrary in_channels (S1 SAR 2-band, S2 Optical 13-band, or S1+S2 15-band).
    """
    def __init__(self, in_channels=2, num_classes=2, base_filters=32):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))
        
        # Dual attention applied at bottleneck and stage 3
        self.attn_bottleneck = PAM_CAM_Layer(f * 16)
        self.attn_skip3 = PAM_CAM_Layer(f * 8)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(f * 16, f * 8)
        
        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(f * 8, f * 4)
        
        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(f * 4, f * 2)
        
        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(f * 2, f)
        
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x4_att = self.attn_bottleneck(x4)
        x3_att = self.attn_skip3(x3)
        
        d3 = self.up1(x4_att)
        d3 = self.conv_up1(torch.cat([x3_att, d3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.conv_up2(torch.cat([x2, d2], dim=1))
        
        d1 = self.up3(d2)
        d1 = self.conv_up3(torch.cat([x1, d1], dim=1))
        
        d0 = self.up4(d1)
        d0 = self.conv_up4(torch.cat([x0, d0], dim=1))
        
        return self.outc(d0)
