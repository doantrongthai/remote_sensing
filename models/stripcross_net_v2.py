import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricCrossShapedStem(nn.Module):
    """
    Asymmetric Cross-Shaped Convolution Stem (ACS-Stem).
    Employs decoupled 1D horizontal and vertical asymmetric convolutions (1x7, 7x1, 1x13, 13x1)
    alongside isotropic 3x3 convolutions to capture elongated linear water bodies
    (rivers, drainage networks, flood inundation corridors) with high receptive field and low FLOPs.
    """
    def __init__(self, in_channels=4, out_channels=32):
        super().__init__()
        mid_ch = out_channels // 4
        
        # Branch 1: 1x7 then 7x1
        self.branch_7 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        
        # Branch 2: 1x13 then 13x1
        self.branch_13 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=(1, 13), padding=(0, 6), bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(13, 1), padding=(6, 0), bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        
        # Branch 3: Standard isotropic 3x3 conv
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        
        # Branch 4: 1x1 point conv
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        b1 = self.branch_7(x)
        b2 = self.branch_13(x)
        b3 = self.branch_3x3(x)
        b4 = self.branch_1x1(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fuse(out)


class OrthogonalStripCrossAttention(nn.Module):
    """
    Orthogonal Strip Cross-Attention (OS-CA).
    Decomposes 2D spatial attention into orthogonal horizontal strip and vertical strip
    cross-attentions. Features are projected along rows and columns, enabling
    long-range topological continuity across expansive water systems without quadratic memory overhead.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        inter_ch = max(16, in_channels // reduction)
        self.inter_ch = inter_ch
        
        # Horizontal strip projection
        self.h_q = nn.Conv2d(in_channels, inter_ch, kernel_size=1)
        self.h_k = nn.Conv2d(in_channels, inter_ch, kernel_size=1)
        self.h_v = nn.Conv2d(in_channels, inter_ch, kernel_size=1)
        
        # Vertical strip projection
        self.v_q = nn.Conv2d(in_channels, inter_ch, kernel_size=1)
        self.v_k = nn.Conv2d(in_channels, inter_ch, kernel_size=1)
        self.v_v = nn.Conv2d(in_channels, inter_ch, kernel_size=1)
        
        # Horizontal and vertical pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]
        
        # Cross-gating between horizontal and vertical contexts
        self.gate_h = nn.Conv2d(inter_ch, inter_ch, kernel_size=1)
        self.gate_v = nn.Conv2d(inter_ch, inter_ch, kernel_size=1)
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(inter_ch * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        
        # Horizontal context along width
        hq = self.h_q(x) # [B, inter_ch, H, W]
        hk = self.h_k(x)
        hv = self.h_v(x)
        
        h_pool_k = self.pool_w(hk) # [B, inter_ch, 1, W]
        h_attn = torch.sigmoid(hq * h_pool_k)
        feat_h = hv * h_attn
        
        # Vertical context along height
        vq = self.v_q(x)
        vk = self.v_k(x)
        vv = self.v_v(x)
        
        v_pool_k = self.pool_h(vk) # [B, inter_ch, H, 1]
        v_attn = torch.sigmoid(vq * v_pool_k)
        feat_v = vv * v_attn
        
        # Orthogonal cross-gating: horizontal features gated by vertical context and vice-versa
        gh = torch.sigmoid(self.gate_h(feat_v))
        gv = torch.sigmoid(self.gate_v(feat_h))
        
        fused_h = feat_h * gh
        fused_v = feat_v * gv
        
        fused = torch.cat([fused_h, fused_v], dim=1)
        out = x + self.gamma * self.out_proj(fused)
        return out


class DirectionalContextFusionDecoder(nn.Module):
    """
    Directional Context Fusion Decoder (DCF-Dec).
    Performs progressive multi-scale directional decoding. At each stage,
    directional strip pooling filters (H x 1 and 1 x W) isolate structural flow
    orientation and modulate skip connections before channel fusion.
    """
    def __init__(self, encoder_channels, num_classes=2):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        
        # Stage 3 decoder
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.strip_mod3 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(c3),
            nn.Sigmoid()
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        
        # Stage 2 decoder
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.strip_mod2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        
        # Stage 1 decoder
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.strip_mod1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(c1, num_classes, kernel_size=1)
        )

    def forward(self, e1, e2, e3, e4):
        # Stage 3
        d3 = self.up3(e4)
        e3_mod = e3 * (1.0 + self.strip_mod3(e3))
        d3 = self.dec3(torch.cat([d3, e3_mod], dim=1))
        
        # Stage 2
        d2 = self.up2(d3)
        e2_mod = e2 * (1.0 + self.strip_mod2(e2))
        d2 = self.dec2(torch.cat([d2, e2_mod], dim=1))
        
        # Stage 1
        d1 = self.up1(d2)
        e1_mod = e1 * (1.0 + self.strip_mod1(e1))
        d1 = self.dec1(torch.cat([d1, e1_mod], dim=1))
        
        logits = self.cls_head(d1)
        return logits


class StripCrossNetV2(nn.Module):
    """
    StripCross-Net-V2: Anisotropic Strip-Convolution and Orthogonal Cross-Attention Network.
    
    Trio Modules:
    1. ACS-Stem: Asymmetric Cross-Shaped Convolution Stem
    2. OS-CA: Orthogonal Strip Cross-Attention
    3. DCF-Dec: Directional Context Fusion Decoder
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        # 1. Asymmetric Cross-Shaped Convolution Stem
        self.stem = AsymmetricCrossShapedStem(in_channels=in_channels, out_channels=c1)
        
        # Encoder stages
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU(),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU()
        )
        
        # 2. Orthogonal Strip Cross-Attention at stages 3 and 4
        self.os_ca3 = OrthogonalStripCrossAttention(c3)
        self.os_ca4 = OrthogonalStripCrossAttention(c4)
        
        # 3. Directional Context Fusion Decoder
        self.decoder = DirectionalContextFusionDecoder(
            encoder_channels=[c1, c2, c3, c4],
            num_classes=num_classes
        )

    def forward(self, x):
        # 1. ACS Stem
        e1 = self.stem(x)         # [B, c1, H, W]
        
        # Encoder
        e2 = self.down1(e1)        # [B, c2, H/2, W/2]
        
        e3 = self.down2(e2)        # [B, c3, H/4, W/4]
        e3 = self.os_ca3(e3)
        
        e4 = self.down3(e3)        # [B, c4, H/8, W/8]
        e4 = self.os_ca4(e4)
        
        # 3. DCF Decoder
        logits = self.decoder(e1, e2, e3, e4)
        return logits
