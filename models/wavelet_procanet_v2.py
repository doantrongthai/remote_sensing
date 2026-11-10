import torch
import torch.nn as nn
import torch.nn.functional as F

class DualTreeComplexWaveletStem(nn.Module):
    """
    Dual-Tree Complex Wavelet Transform Stem (DTCWT-Stem).
    Performs 2D dual-tree complex wavelet decomposition into 6 directional sub-bands
    oriented at +/-15 deg, +/-45 deg, and +/-75 deg.
    Combines real and imaginary trees to achieve approximate shift-invariance and
    directional edge selectivity for multispectral satellite imagery.
    """
    def __init__(self, in_channels=4, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Quadrature mirror filter coefficients (LeGall / Antonini style biorthogonal filters)
        # Tree A (symmetric)
        h0 = torch.tensor([-0.0883883476, 0.0883883476, 0.695879989, 0.695879989, 0.0883883476, -0.0883883476])
        h1 = torch.tensor([-0.0883883476, -0.0883883476, 0.695879989, -0.695879989, 0.0883883476, 0.0883883476])
        # Tree B (quarter-sample shift Hilbert pair)
        g0 = torch.tensor([0.01122679, 0.01122679, -0.08838835, 0.08838835, 0.57560737, 0.57560737])
        g1 = torch.tensor([-0.57560737, 0.57560737, 0.08838835, 0.08838835, -0.01122679, 0.01122679])
        
        # Construct 2D separable filter pairs (Tree A x Tree A, Tree B x Tree B, etc.)
        def make_2d(f_r, f_c):
            return torch.outer(f_r, f_c).unsqueeze(0).unsqueeze(0)  # [1, 1, 6, 6]
        
        # Lowpass filter
        ll_aa = make_2d(h0, h0)
        ll_bb = make_2d(g0, g0)
        ll_ab = make_2d(h0, g0)
        ll_ba = make_2d(g0, h0)
        
        # Highpass combinations for 6 directional sub-bands
        # Subband 1 (+15 deg): LH
        lh_aa = make_2d(h0, h1)
        lh_bb = make_2d(g0, g1)
        lh_ba = make_2d(g0, h1)
        lh_ab = make_2d(h0, g1)
        
        # Subband 2 (+45 deg): HH
        hh_aa = make_2d(h1, h1)
        hh_bb = make_2d(g1, g1)
        hh_ba = make_2d(g1, h1)
        hh_ab = make_2d(h1, g1)
        
        # Subband 3 (+75 deg): HL
        hl_aa = make_2d(h1, h0)
        hl_bb = make_2d(g1, g0)
        hl_ba = make_2d(g1, h0)
        hl_ab = make_2d(h1, g0)
        
        # Register fixed filter banks as buffers
        self.register_buffer('ll_filt', (ll_aa + ll_bb + ll_ab + ll_ba) / 4.0)
        
        # Real and imaginary components of 6 directional orientations
        # 1: +15 deg
        self.register_buffer('dir1_re', (lh_aa - lh_bb) / 1.41421356)
        self.register_buffer('dir1_im', (lh_ba + lh_ab) / 1.41421356)
        # 2: +45 deg
        self.register_buffer('dir2_re', (hh_aa - hh_bb) / 1.41421356)
        self.register_buffer('dir2_im', (hh_ba + hh_ab) / 1.41421356)
        # 3: +75 deg
        self.register_buffer('dir3_re', (hl_aa - hl_bb) / 1.41421356)
        self.register_buffer('dir3_im', (hl_ba + hl_ab) / 1.41421356)
        # 4: -15 deg
        self.register_buffer('dir4_re', (lh_aa + lh_bb) / 1.41421356)
        self.register_buffer('dir4_im', (lh_ba - lh_ab) / 1.41421356)
        # 5: -45 deg
        self.register_buffer('dir5_re', (hh_aa + hh_bb) / 1.41421356)
        self.register_buffer('dir5_im', (hh_ba - hh_ab) / 1.41421356)
        # 6: -75 deg
        self.register_buffer('dir6_re', (hl_aa + hl_bb) / 1.41421356)
        self.register_buffer('dir6_im', (hl_ba - hl_ab) / 1.41421356)
        
        # Sub-band output channels per input channel = 1 (LL) + 6 (directional magnitudes) = 7
        total_subband_ch = in_channels * 7
        
        # Learnable projection and adaptation
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels + total_subband_ch, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 4, H, W]
        b, c, h, w = x.shape
        eps = 1e-6
        subband_feats = []
        
        # Apply 2D directional wavelet filters channel-wise
        for ch in range(c):
            x_ch = x[:, ch:ch+1, :, :]  # [B, 1, H, W]
            
            # Low-frequency approximation
            ll = F.conv2d(x_ch, self.ll_filt, padding=2)
            
            # 6 Directional sub-band magnitudes (approximate shift-invariant envelope)
            d1_re = F.conv2d(x_ch, self.dir1_re, padding=2)
            d1_im = F.conv2d(x_ch, self.dir1_im, padding=2)
            m1 = torch.sqrt(d1_re ** 2 + d1_im ** 2 + eps)
            
            d2_re = F.conv2d(x_ch, self.dir2_re, padding=2)
            d2_im = F.conv2d(x_ch, self.dir2_im, padding=2)
            m2 = torch.sqrt(d2_re ** 2 + d2_im ** 2 + eps)
            
            d3_re = F.conv2d(x_ch, self.dir3_re, padding=2)
            d3_im = F.conv2d(x_ch, self.dir3_im, padding=2)
            m3 = torch.sqrt(d3_re ** 2 + d3_im ** 2 + eps)
            
            d4_re = F.conv2d(x_ch, self.dir4_re, padding=2)
            d4_im = F.conv2d(x_ch, self.dir4_im, padding=2)
            m4 = torch.sqrt(d4_re ** 2 + d4_im ** 2 + eps)
            
            d5_re = F.conv2d(x_ch, self.dir5_re, padding=2)
            d5_im = F.conv2d(x_ch, self.dir5_im, padding=2)
            m5 = torch.sqrt(d5_re ** 2 + d5_im ** 2 + eps)
            
            d6_re = F.conv2d(x_ch, self.dir6_re, padding=2)
            d6_im = F.conv2d(x_ch, self.dir6_im, padding=2)
            m6 = torch.sqrt(d6_re ** 2 + d6_im ** 2 + eps)
            
            subband_feats.extend([ll, m1, m2, m3, m4, m5, m6])
            
        subband_tensor = torch.cat(subband_feats, dim=1) # [B, 28, H, W]
        # Align spatial dimensions if padding caused 1px shift
        if subband_tensor.shape[2:] != (h, w):
            subband_tensor = F.interpolate(subband_tensor, size=(h, w), mode='bilinear', align_corners=False)
            
        stem_in = torch.cat([x, subband_tensor], dim=1)
        out = self.stem_conv(stem_in)
        return out, subband_tensor


class DirectionalSubbandCrossAttention(nn.Module):
    """
    Directional Sub-band Cross-Attention (DSB-CA).
    Cross-attends spatial features with the multi-orientation complex wavelet
    coefficients to reinforce boundary orientation alignment and suppress
    speckle/clutter noise.
    """
    def __init__(self, in_channels, subband_channels, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.subband_channels = subband_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(subband_channels, in_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(subband_channels, in_channels, kernel_size=1, bias=False)
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, x, subband_feats):
        # x: [B, C, H, W]
        # subband_feats: [B, C_sb, H_s, W_s]
        b, c, h, w = x.shape
        if subband_feats.shape[2:] != (h, w):
            subband_feats = F.interpolate(subband_feats, size=(h, w), mode='bilinear', align_corners=False)
            
        q = self.q_proj(x)
        k = self.k_proj(subband_feats)
        v = self.v_proj(subband_feats)
        
        # Directional feature modulation
        attn_weight = torch.sigmoid(q * k * self.scale)
        att_v = v * attn_weight
        
        # Channel gating
        channel_attn = self.gate(att_v)
        refined = x + att_v * channel_attn
        
        return self.out_proj(refined)


class ComplexWaveletInverseDecoder(nn.Module):
    """
    Complex Wavelet Inverse Reconstructor Decoder (CWIR-Decoder).
    Synthesizes multi-scale feature hierarchies by combining progressive
    transposed convolutions with directional sub-band residual injection,
    restoring sharp boundaries and preventing edge diffusion.
    """
    def __init__(self, encoder_channels, subband_channels, num_classes=2):
        super().__init__()
        # encoder_channels: [c1, c2, c3, c4]
        c1, c2, c3, c4 = encoder_channels
        
        # Stage 3 decoder
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        
        # Stage 2 decoder
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        
        # Stage 1 decoder
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
        # High-resolution boundary reconstruction injection
        self.recon_proj = nn.Sequential(
            nn.Conv2d(subband_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(c1, num_classes, kernel_size=1)
        )

    def forward(self, e1, e2, e3, e4, subband_feats):
        # Bottleneck upsampling
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Complex wavelet reconstruction residual
        sb_proj = self.recon_proj(subband_feats)
        if sb_proj.shape[2:] != d1.shape[2:]:
            sb_proj = F.interpolate(sb_proj, size=d1.shape[2:], mode='bilinear', align_corners=False)
            
        fused = torch.cat([d1, sb_proj], dim=1)
        logits = self.head(fused)
        return logits


class WaveletProCANetV2(nn.Module):
    """
    Wavelet-ProCANet-V2: Directional Dual-Tree Wavelet Progressive Context Attention Network.
    
    Trio Modules:
    1. DTCWT-Stem: Dual-Tree Complex Wavelet Stem (6 directional sub-bands at +/-15, +/-45, +/-75 deg)
    2. DSB-CA: Directional Sub-band Cross-Attention
    3. CWIR-Decoder: Complex Wavelet Inverse Reconstructor Decoder
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        # 1. Dual-Tree Complex Wavelet Stem
        self.stem = DualTreeComplexWaveletStem(in_channels=in_channels, out_channels=c1)
        subband_ch = in_channels * 7  # 4 * 7 = 28
        
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
        
        # 2. Directional Sub-band Cross-Attention at bottleneck and high-level stages
        self.dsb_ca3 = DirectionalSubbandCrossAttention(c3, subband_ch)
        self.dsb_ca4 = DirectionalSubbandCrossAttention(c4, subband_ch)
        
        # 3. Complex Wavelet Inverse Reconstructor Decoder
        self.decoder = ComplexWaveletInverseDecoder(
            encoder_channels=[c1, c2, c3, c4],
            subband_channels=subband_ch,
            num_classes=num_classes
        )

    def forward(self, x):
        # Input: [B, 4, H, W]
        # 1. DTCWT Stem
        e1, subband_feats = self.stem(x)  # [B, c1, H, W], [B, 28, H, W]
        
        # Encoder
        e2 = self.down1(e1)               # [B, c2, H/2, W/2]
        e3 = self.down2(e2)               # [B, c3, H/4, W/4]
        e3 = self.dsb_ca3(e3, subband_feats)
        
        e4 = self.down3(e3)               # [B, c4, H/8, W/8]
        e4 = self.dsb_ca4(e4, subband_feats)
        
        # 3. CWIR Decoder
        logits = self.decoder(e1, e2, e3, e4, subband_feats)
        return logits
