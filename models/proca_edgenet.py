import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ProCA-EdgeNet: Edge-Gated Progressive Cross-Attention Network for Flood Segmentation.

Water bodies exhibit sharp physical boundaries in the Near-Infrared (NIR) band due to strong
absorption characteristics of surface water compared to surrounding terrain. ProCA-EdgeNet
leverages multi-scale Sobel and Laplacian edge extraction directly on the raw NIR band to gate
and sharpen cross-modal feature exchange at water-land interfaces.
"""


class ConvBNReLU(nn.Module):
    """Standard 2D Convolution + BatchNorm + ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive 3x3 Conv-BN-ReLU layers."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, kernel_size=3, padding=1),
            ConvBNReLU(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class NIREdgeExtractor(nn.Module):
    """
    Multi-operator edge extraction module operating directly on the raw NIR band.
    Combines horizontal Sobel (Gx), vertical Sobel (Gy), and discrete Laplacian (Lap)
    filters to extract deterministic high-frequency boundary gradients.
    """
    def __init__(self, out_channels=32):
        super().__init__()
        # Sobel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.0
        # Sobel Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3) / 8.0
        # Laplacian: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3) / 4.0

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)

        # Learnable multi-scale edge projection (3 operator channels -> out_channels)
        self.edge_proj = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, nir):
        """
        Args:
            nir: NIR band tensor [B, 1, H, W]
        Returns:
            edge_features: Processed multi-operator edge features [B, out_channels, H, W]
        """
        gx = F.conv2d(nir, self.sobel_x, padding=1)
        gy = F.conv2d(nir, self.sobel_y, padding=1)
        sobel_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        lap = torch.abs(F.conv2d(nir, self.laplacian, padding=1))

        # Stack [Sobel Mag, Gx, Lap] to retain orientation and magnitude
        edge_raw = torch.cat([sobel_mag, torch.abs(gx) + torch.abs(gy), lap], dim=1)
        return self.edge_proj(edge_raw)


class EdgeGatedProCA(nn.Module):
    """
    Edge-Gated Progressive Cross-Attention Module.

    Integrates boundary gradient guidance into the progressive cross-attention mechanism:
    1) Self-attention computes initial intra-modality importance maps.
    2) An edge gate E_gate in [0, 1] is predicted from multi-scale edge features.
    3) Cross-attention attention masks are amplified at high-gradient boundary locations:
       a_{N->R} = a_{N->R} * (1.0 + E_gate)
       a_{R->N} = a_{R->N} * (1.0 + E_gate)
    4) Fused features are combined with projected edge features to retain sharp waterline contours.
    """
    def __init__(self, channels, edge_ch=32):
        super().__init__()
        self.channels = channels

        # Stage 1: Self-attention
        self.self_conv_r = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.self_conv_n = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Stage 2: Cross-attention
        self.cross_conv_r2n = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.cross_conv_n2r = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Edge gating projection
        self.edge_gate_conv = nn.Sequential(
            nn.Conv2d(edge_ch, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.edge_feature_proj = nn.Sequential(
            nn.Conv2d(edge_ch, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Stage 3: Fusion refinement
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_r, x_n, edge_feat):
        """
        Args:
            x_r: RGB+NIR feature map [B, C, H, W]
            x_n: NIR feature map [B, C, H, W]
            edge_feat: Edge features [B, edge_ch, H, W]
        Returns:
            fused: Edge-gated cross-attended features [B, C, H, W]
        """
        # Interpolate edge features if spatial dimension does not match
        if edge_feat.shape[-2:] != x_r.shape[-2:]:
            edge_feat = F.interpolate(edge_feat, size=x_r.shape[-2:], mode='bilinear', align_corners=False)

        # 1. Self-attention
        a_r = self.self_conv_r(x_r)
        a_n = self.self_conv_n(x_n)
        x_r_self = x_r * a_r
        x_n_self = x_n * a_n

        # 2. Edge gate
        e_gate = self.edge_gate_conv(edge_feat)

        # 3. Edge-modulated cross-attention
        a_r2n = self.cross_conv_r2n(x_r_self) * (1.0 + e_gate)
        a_n2r = self.cross_conv_n2r(x_n_self) * (1.0 + e_gate)

        x_r_cross = x_r_self * a_n2r
        x_n_cross = x_n_self * a_r2n

        # 4. Composite fusion with edge injection
        edge_residual = self.edge_feature_proj(edge_feat)
        fused = x_r_cross + x_n_cross + edge_residual
        return self.fuse_refine(fused)


class DecoderBlock(nn.Module):
    """Upsampling followed by feature concatenation and double convolution."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((in_ch // 2) + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ProCAEdgeNet(nn.Module):
    """
    ProCA-EdgeNet: Edge-Gated Progressive Cross-Attention Network.

    Combines dual-stream RGB+NIR and NIR encoders with an NIR Edge Extraction
    Branch (Sobel + Laplacian operators) and EdgeGatedProCA modules across all stages.

    Args:
        in_channels (int): Input multispectral channels (default: 4 for Sentinel-2 B4, B3, B2, B8).
        num_classes (int): Number of semantic classes (default: 2 for flood water vs land).
        base_channels (int): Base filters in the initial encoder stage (default: 32).
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels
        edge_ch = 32

        self.s1_ch = in_channels
        self.s2_ch = 1 if in_channels >= 4 else max(1, in_channels // 2)

        # Raw NIR edge extraction branch
        self.edge_extractor = NIREdgeExtractor(out_channels=edge_ch)

        # Stream 1: RGB + NIR Encoder
        self.enc1_conv1 = DoubleConv(self.s1_ch, c)
        self.pool1_r = nn.MaxPool2d(2)
        self.enc1_conv2 = DoubleConv(c, c * 2)
        self.pool2_r = nn.MaxPool2d(2)
        self.enc1_conv3 = DoubleConv(c * 2, c * 4)
        self.pool3_r = nn.MaxPool2d(2)
        self.enc1_conv4 = DoubleConv(c * 4, c * 8)
        self.pool4_r = nn.MaxPool2d(2)

        # Stream 2: NIR Encoder
        self.enc2_conv1 = DoubleConv(self.s2_ch, c)
        self.pool1_n = nn.MaxPool2d(2)
        self.enc2_conv2 = DoubleConv(c, c * 2)
        self.pool2_n = nn.MaxPool2d(2)
        self.enc2_conv3 = DoubleConv(c * 2, c * 4)
        self.pool3_n = nn.MaxPool2d(2)
        self.enc2_conv4 = DoubleConv(c * 4, c * 8)
        self.pool4_n = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck_r = DoubleConv(c * 8, c * 16)
        self.bottleneck_n = DoubleConv(c * 8, c * 16)

        # Multi-scale EdgeGatedProCA blocks
        self.proca1 = EdgeGatedProCA(c, edge_ch=edge_ch)
        self.proca2 = EdgeGatedProCA(c * 2, edge_ch=edge_ch)
        self.proca3 = EdgeGatedProCA(c * 4, edge_ch=edge_ch)
        self.proca4 = EdgeGatedProCA(c * 8, edge_ch=edge_ch)
        self.proca_bottleneck = EdgeGatedProCA(c * 16, edge_ch=edge_ch)

        # Decoder path
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        # Segmentation Head
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def _split_modalities(self, x):
        """Extract RGB+NIR and NIR band from input."""
        if self.in_channels >= 4:
            x_rgb_nir = x[:, :4, :, :]
            x_nir = x[:, 3:4, :, :]
        elif self.in_channels == 3:
            x_rgb_nir = x
            x_nir = x[:, 2:3, :, :]
        else:
            x_rgb_nir = x
            x_nir = x[:, -1:, :, :]
        return x_rgb_nir, x_nir

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Multispectral image tensor [B, in_channels, H, W]
        Returns:
            Tensor: Raw logits [B, num_classes, H, W]
        """
        x_r, x_n = self._split_modalities(x)

        # Extract multi-operator raw NIR edge features
        edge_feat = self.edge_extractor(x_n)

        # Stage 1
        x1_r = self.enc1_conv1(x_r)
        x1_n = self.enc2_conv1(x_n)
        skip1 = self.proca1(x1_r, x1_n, edge_feat)

        # Stage 2
        x2_r = self.enc1_conv2(self.pool1_r(x1_r))
        x2_n = self.enc2_conv2(self.pool1_n(x1_n))
        skip2 = self.proca2(x2_r, x2_n, edge_feat)

        # Stage 3
        x3_r = self.enc1_conv3(self.pool2_r(x2_r))
        x3_n = self.enc2_conv3(self.pool2_n(x2_n))
        skip3 = self.proca3(x3_r, x3_n, edge_feat)

        # Stage 4
        x4_r = self.enc1_conv4(self.pool3_r(x3_r))
        x4_n = self.enc2_conv4(self.pool3_n(x3_n))
        skip4 = self.proca4(x4_r, x4_n, edge_feat)

        # Bottleneck
        b_r = self.bottleneck_r(self.pool4_r(x4_r))
        b_n = self.bottleneck_n(self.pool4_n(x4_n))
        b_fused = self.proca_bottleneck(b_r, b_n, edge_feat)

        # Decoder
        d4 = self.dec4(b_fused, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        return self.head(d1)
