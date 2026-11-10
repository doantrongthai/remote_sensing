import torch
import torch.nn as nn
import torch.nn.functional as F

class RedNIRSlopeDisentanglementBlock(nn.Module):
    """
    Red-NIR Slope Disentanglement Block (RNS-Block).
    Analyzes the Red-NIR spectral trajectory to disentangle suspended sediment scattering
    from pure liquid water absorption. Computes spectral slope, curvature, and NDTI,
    projecting features into decoupled sediment and water subspaces.
    """
    def __init__(self, in_channels=4, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 4 raw bands + 3 slope/curvature indices = 7 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(7, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Dual orthogonal projection heads
        self.water_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.sediment_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 4, H, W] -> Red=0, Green=1, Blue=2, NIR=3
        eps = 1e-6
        red = x[:, 0:1, :, :]
        green = x[:, 1:2, :, :]
        nir = x[:, 3:4, :, :]
        
        # Spectral slope in Red-NIR interval
        slope_r_nir = nir - red
        # Spectral curvature (second derivative across Green -> Red -> NIR)
        curvature = green - 2.0 * red + nir
        # Normalized Difference Turbidity Index (NDTI)
        ndti = (red - green) / (red + green + eps)
        
        traj_feats = torch.cat([x, slope_r_nir, curvature, ndti], dim=1)
        h = self.encoder(traj_feats)
        
        f_water = self.water_proj(h)
        f_sediment = self.sediment_proj(h)
        
        return f_water, f_sediment


class SedimentDecoupledCrossAttention(nn.Module):
    """
    Sediment-Decoupled Cross-Attention (SD-CA).
    Decouples suspended sediment scattering features from water body representations.
    Uses the sediment stream as an inhibitory gate to prevent misclassifying
    muddy floodwaters as agricultural soil or wet land.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_water = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_water = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        self.k_sediment = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_vis, f_water, f_sediment):
        # f_vis: [B, C, H, W]
        # f_water: [B, C, H_w, W_w]
        # f_sediment: [B, C, H_s, W_s]
        b, c, h, w = f_vis.shape
        if f_water.shape[2:] != (h, w):
            f_water = F.interpolate(f_water, size=(h, w), mode='bilinear', align_corners=False)
        if f_sediment.shape[2:] != (h, w):
            f_sediment = F.interpolate(f_sediment, size=(h, w), mode='bilinear', align_corners=False)
            
        q = self.q_conv(f_vis)
        kw = self.k_water(f_water)
        vw = self.v_water(f_water)
        ks = self.k_sediment(f_sediment)
        
        # Water affinity
        water_affinity = torch.sigmoid(q * kw)
        # Sediment inhibitory gate
        sediment_gate = torch.sigmoid(q * ks)
        
        # Effective cross-attention with sediment attenuation
        effective_att = water_affinity * (1.0 - 0.5 * sediment_gate)
        att_out = vw * effective_att
        
        out = self.out_conv(f_vis + self.gamma * att_out)
        return out


class DualThresholdWaterRefinementModule(nn.Module):
    """
    Dual-Threshold Water Refinement Module (DTW-RM).
    Applies differentiable dual-threshold hysteresis. Identifies high-confidence
    water seeds and candidate boundary pixels, propagating connectivity across
    turbid regions through differentiable affinity diffusion steps.
    """
    def __init__(self, in_channels, num_classes=2, num_steps=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_steps
        
        # Predicts high threshold seed mask and low threshold candidate mask
        self.seed_head = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1)  # 0: high-confidence core, 1: candidate boundary
        )
        
        # Affinity propagation kernel
        self.affinity_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 9, kernel_size=1),  # 3x3 local neighbor affinities
            nn.Softmax(dim=1)
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, feat):
        # feat: [B, C, H, W]
        b, c, h, w = feat.shape
        threshold_logits = self.seed_head(feat)
        m_core = torch.sigmoid(threshold_logits[:, 0:1, :, :])   # Core seed
        m_cand = torch.sigmoid(threshold_logits[:, 1:2, :, :])   # Candidates
        
        # Differentiable region growing
        m_curr = m_core
        # 3x3 unfold for local diffusion
        for _ in range(self.num_steps):
            m_unfold = F.unfold(m_curr, kernel_size=3, padding=1)  # [B, 9, H*W]
            m_unfold = m_unfold.view(b, 9, h, w)
            affinities = self.affinity_conv(feat)                  # [B, 9, H, W]
            diffused = (m_unfold * affinities).sum(dim=1, keepdim=True)
            # Propagate into candidates only
            m_curr = torch.clamp(m_curr + 0.3 * (diffused * m_cand), 0.0, 1.0)
            
        fused = torch.cat([feat, m_curr, m_cand], dim=1)
        logits = self.cls_head(fused)
        return logits


class TurbidWaterNet(nn.Module):
    """
    TurbidWater-Net: Sediment-Decoupled Spectral Trajectory Network for Turbid Flood Detection.
    
    Trio Modules:
    1. RNS-Block: Red-NIR Slope Disentanglement Block
    2. SD-CA: Sediment-Decoupled Cross-Attention
    3. DTW-RM: Dual-Threshold Water Refinement Module
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        # 1. Red-NIR Slope Disentanglement Block
        self.rns_block = RedNIRSlopeDisentanglementBlock(in_channels=in_channels, out_channels=c1)
        
        # Visual Encoder stages
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
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
        
        # 2. Sediment-Decoupled Cross-Attention
        self.sd_ca3 = SedimentDecoupledCrossAttention(c3)
        self.sd_ca4 = SedimentDecoupledCrossAttention(c4)
        
        # Downsamplers for water & sediment streams
        self.w_down3 = nn.Conv2d(c1, c3, kernel_size=1)
        self.s_down3 = nn.Conv2d(c1, c3, kernel_size=1)
        self.w_down4 = nn.Conv2d(c1, c4, kernel_size=1)
        self.s_down4 = nn.Conv2d(c1, c4, kernel_size=1)
        
        # Decoder stages
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
        # 3. Dual-Threshold Water Refinement Module
        self.dtw_rm = DualThresholdWaterRefinementModule(c1, num_classes=num_classes)

    def forward(self, x):
        # 1. RNS-Block
        f_water, f_sediment = self.rns_block(x)
        
        # Visual stem
        e1 = self.stem(torch.cat([x, f_water], dim=1))
        
        # Encoder
        e2 = self.down1(e1)
        
        e3 = self.down2(e2)
        e3 = self.sd_ca3(e3, self.w_down3(f_water), self.s_down3(f_sediment))
        
        e4 = self.down3(e3)
        e4 = self.sd_ca4(e4, self.w_down4(f_water), self.s_down4(f_sediment))
        
        # Decoder
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 3. DTW-RM
        logits = self.dtw_rm(d1)
        return logits
