import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
KAN-ProCANet: Kolmogorov-Arnold Network Progressive Cross-Attention Network for Flood Segmentation.

In multispectral flood segmentation, the non-linear relationship between visible optical reflectance
(RGB) and near-infrared absorption (NIR) varies dramatically with water turbidity, suspended sediment,
chlorophyll content, and substrate depth. Standard linear weights and fixed static activation
functions approximate such multi-modal physics poorly. KAN-ProCANet replaces standard linear
projections in progressive cross-attention with Kolmogorov-Arnold Network (KAN) layers featuring
learnable univariate B-spline curves along every channel edge.
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


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear Layer with Learnable B-spline Basis Functions.

    Computes for each input channel:
        phi(x) = base_weight * silu(x) + spline_weight * B_splines(x)
    where B_splines(x) is evaluated using Cox-de Boor recursion over a fixed knot grid.

    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
        grid_size (int): Number of intervals in spline grid (default: 3).
        spline_order (int): Degree of B-spline polynomials (default: 2 for quadratic B-splines).
        scale_noise (float): Random noise scale for spline weight initialization.
    """
    def __init__(self, in_features, out_features, grid_size=3, spline_order=2, scale_noise=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Construct knot grid in normalized [-1.0, 1.0] range with padding for spline order
        h = 2.0 / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h - 1.0)
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Base linear transformation weight
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        # Spline coefficients weight matrix
        num_spline_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features * num_spline_bases) * (scale_noise / math.sqrt(in_features))
        )

    def b_splines(self, x):
        """
        Evaluate Cox-de Boor B-spline recurrence for input tensor x.
        Args:
            x: Tensor of shape [*batch_dims, in_features]
        Returns:
            bases: Tensor of shape [*batch_dims, in_features * (grid_size + spline_order)]
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features).unsqueeze(-1)  # [N, in_features, 1]

        # Order 0 indicator basis
        bases = ((x_flat >= self.grid[:, :-1]) & (x_flat < self.grid[:, 1:])).to(x.dtype)

        # Cox-de Boor recurrence relation for higher degrees
        for k in range(1, self.spline_order + 1):
            denom1 = self.grid[:, k:-1] - self.grid[:, :-(k + 1)]
            denom2 = self.grid[:, k + 1:] - self.grid[:, 1:-k]
            term1 = (x_flat - self.grid[:, :-(k + 1)]) / torch.clamp(denom1, min=1e-6) * bases[:, :, :-1]
            term2 = (self.grid[:, k + 1:] - x_flat) / torch.clamp(denom2, min=1e-6) * bases[:, :, 1:]
            bases = term1 + term2

        # Flatten bases across in_features and spline functions
        return bases.reshape(*orig_shape[:-1], -1)

    def forward(self, x):
        """
        Args:
            x: Input tensor [*batch_dims, in_features]
        Returns:
            Output tensor [*batch_dims, out_features]
        """
        base_output = F.linear(F.silu(x), self.base_weight)
        spline_basis = self.b_splines(x)
        spline_output = F.linear(spline_basis, self.spline_weight)
        return base_output + spline_output


class KANConv1x1(nn.Module):
    """
    1x1 2D Convolution layer driven by a KANLinear B-spline module.
    Normalizes spatial feature channels via BatchNorm and tanh into [-1, 1],
    then applies learnable non-linear univariate B-splines across channels.
    """
    def __init__(self, in_channels, out_channels, grid_size=3, spline_order=2):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.kan = KANLinear(in_channels, out_channels, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        # Normalize into [-1, 1] for stable B-spline knot activation
        x_norm = torch.tanh(self.bn(x))
        # Permute [B, C, H, W] -> [B, H, W, C]
        x_perm = x_norm.permute(0, 2, 3, 1).contiguous()
        out = self.kan(x_perm)
        # Permute back [B, H, W, C_out] -> [B, C_out, H, W]
        return out.permute(0, 3, 1, 2).contiguous()


class KANCrossAttention(nn.Module):
    """
    KAN-Enhanced Progressive Cross-Attention Module.

    Replaces static linear/conv cross-attention projections with learnable
    continuous B-splines:
    1) Self-attention:
       a_R = sigmoid(DepthwiseConv(KAN_R(X_R)))
       a_N = sigmoid(DepthwiseConv(KAN_N(X_N)))
       X^_R = X_R * a_R,  X^_N = X_N * a_N
    2) Cross-attention:
       a_{R->N} = sigmoid(DepthwiseConv(KAN_{R->N}(X^_R)))
       a_{N->R} = sigmoid(DepthwiseConv(KAN_{N->R}(X^_N)))
       X~_R = X^_R * a_{N->R}
       X~_N = X^_N * a_{R->N}
    3) KAN Non-linear Composite Fusion:
       fused = KAN_{fuse}(X~_R + X~_N), followed by BatchNorm and ReLU.
    """
    def __init__(self, channels, grid_size=3, spline_order=2):
        super().__init__()
        self.channels = channels

        # Stage 1: Self-attention via KAN + Depthwise Conv
        self.self_kan_r = nn.Sequential(
            KANConv1x1(channels, channels, grid_size=grid_size, spline_order=spline_order),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.self_kan_n = nn.Sequential(
            KANConv1x1(channels, channels, grid_size=grid_size, spline_order=spline_order),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Stage 2: Cross-attention via KAN + Depthwise Conv
        self.cross_kan_r2n = nn.Sequential(
            KANConv1x1(channels, channels, grid_size=grid_size, spline_order=spline_order),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.cross_kan_n2r = nn.Sequential(
            KANConv1x1(channels, channels, grid_size=grid_size, spline_order=spline_order),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Stage 3: KAN Fusion refinement
        self.fuse_kan = nn.Sequential(
            KANConv1x1(channels, channels, grid_size=grid_size, spline_order=spline_order),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_r, x_n):
        """
        Args:
            x_r: RGB+NIR feature map [B, C, H, W]
            x_n: NIR feature map [B, C, H, W]
        Returns:
            fused: KAN-cross-attended feature map [B, C, H, W]
        """
        # 1. Self-attention
        a_r = self.self_kan_r(x_r)
        a_n = self.self_kan_n(x_n)
        x_r_self = x_r * a_r
        x_n_self = x_n * a_n

        # 2. Cross-attention
        a_r2n = self.cross_kan_r2n(x_r_self)
        a_n2r = self.cross_kan_n2r(x_n_self)

        x_r_cross = x_r_self * a_n2r
        x_n_cross = x_n_self * a_r2n

        # 3. KAN Fusion Refinement
        fused = x_r_cross + x_n_cross
        return self.fuse_kan(fused)


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


class KANProCANet(nn.Module):
    """
    KAN-ProCANet: Kolmogorov-Arnold Network Progressive Cross-Attention Network.

    Features dual-stream encoders with KANCrossAttention modules equipped with
    learnable continuous B-spline linear projections across all progressive stages.

    Args:
        in_channels (int): Multispectral input channels (default: 4 for Sentinel-2 B4, B3, B2, B8).
        num_classes (int): Segmentation output classes (default: 2 for flood water vs land).
        base_channels (int): Base channel depth (default: 32).
        grid_size (int): Spline grid intervals for KANLinear (default: 3).
        spline_order (int): Degree of B-splines (default: 2 for quadratic B-splines).
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32, grid_size=3, spline_order=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels

        self.s1_ch = in_channels
        self.s2_ch = 1 if in_channels >= 4 else max(1, in_channels // 2)

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

        # Multi-scale KANCrossAttention blocks
        self.proca1 = KANCrossAttention(c, grid_size=grid_size, spline_order=spline_order)
        self.proca2 = KANCrossAttention(c * 2, grid_size=grid_size, spline_order=spline_order)
        self.proca3 = KANCrossAttention(c * 4, grid_size=grid_size, spline_order=spline_order)
        self.proca4 = KANCrossAttention(c * 8, grid_size=grid_size, spline_order=spline_order)
        self.proca_bottleneck = KANCrossAttention(c * 16, grid_size=grid_size, spline_order=spline_order)

        # Decoder path
        self.dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.dec1 = DecoderBlock(c * 2, c, c)

        # Segmentation Head
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def _split_modalities(self, x):
        """Extract RGB+NIR and NIR bands."""
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

        # Stage 1
        x1_r = self.enc1_conv1(x_r)
        x1_n = self.enc2_conv1(x_n)
        skip1 = self.proca1(x1_r, x1_n)

        # Stage 2
        x2_r = self.enc1_conv2(self.pool1_r(x1_r))
        x2_n = self.enc2_conv2(self.pool1_n(x1_n))
        skip2 = self.proca2(x2_r, x2_n)

        # Stage 3
        x3_r = self.enc1_conv3(self.pool2_r(x2_r))
        x3_n = self.enc2_conv3(self.pool2_n(x2_n))
        skip3 = self.proca3(x3_r, x3_n)

        # Stage 4
        x4_r = self.enc1_conv4(self.pool3_r(x3_r))
        x4_n = self.enc2_conv4(self.pool3_n(x3_n))
        skip4 = self.proca4(x4_r, x4_n)

        # Bottleneck
        b_r = self.bottleneck_r(self.pool4_r(x4_r))
        b_n = self.bottleneck_n(self.pool4_n(x4_n))
        b_fused = self.proca_bottleneck(b_r, b_n)

        # Decoder path
        d4 = self.dec4(b_fused, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        return self.head(d1)
