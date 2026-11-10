import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelNIRDirectionalGradientStream(nn.Module):
    """
    Sobel-NIR Directional Gradient Stream (SNDG).
    Extracts multi-directional spatial gradients (0 deg, 45 deg, 90 deg, 135 deg)
    and computes continuous vector fields (gradient magnitude, normal, tangent)
    predominantly from the high-contrast NIR (B8) and Green (B3) bands.
    Processes these directional vector fields through a dedicated feature encoder.
    """
    def __init__(self, out_channels=32):
        super().__init__()
        # Sobel directional filter kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        sobel_45 = torch.tensor([[ 0.,  1.,  2.],
                                 [-1.,  0.,  1.],
                                 [-2., -1.,  0.]]).view(1, 1, 3, 3)
        sobel_135 = torch.tensor([[-2., -1.,  0.],
                                  [-1.,  0.,  1.],
                                  [ 0.,  1.,  2.]]).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('sobel_45', sobel_45)
        self.register_buffer('sobel_135', sobel_135)
        
        # 2 bands (Green=1, NIR=3) * 6 gradient/vector channels = 12 channels
        # (gx, gy, g45, g135, mag, tangent_x, tangent_y)
        in_dim = 14
        self.stream_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 4, H, W] -> channel 1 is Green, channel 3 is NIR
        eps = 1e-6
        green = x[:, 1:2, :, :]
        nir = x[:, 3:4, :, :]
        
        vector_feats = []
        for band in [green, nir]:
            gx = F.conv2d(band, self.sobel_x, padding=1)
            gy = F.conv2d(band, self.sobel_y, padding=1)
            g45 = F.conv2d(band, self.sobel_45, padding=1)
            g135 = F.conv2d(band, self.sobel_135, padding=1)
            
            mag = torch.sqrt(gx ** 2 + gy ** 2 + eps)
            nx = gx / mag
            ny = gy / mag
            # Tangent vector orthogonal to normal: [-ny, nx]
            tx = -ny
            ty = nx
            
            vector_feats.extend([gx, gy, g45, g135, mag, tx, ty])
            
        vector_tensor = torch.cat(vector_feats, dim=1)  # [B, 14, H, W]
        edge_stream_feat = self.stream_conv(vector_tensor)
        return edge_stream_feat, vector_tensor


class EdgeVectorFieldCrossAttention(nn.Module):
    """
    Edge-Vector Field Cross-Attention (EVF-CA).
    Steers spatial attention along the boundary tangent vector field,
    enforcing anisotropic feature propagation along water shorelines
    while suppressing cross-boundary bleeding.
    """
    def __init__(self, in_channels, edge_channels):
        super().__init__()
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(edge_channels, in_channels, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(edge_channels, in_channels, kernel_size=1, bias=False)
        
        # Tangent directional guidance projection
        self.tangent_proj = nn.Sequential(
            nn.Conv2d(edge_channels, 2, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )
        
        self.sobel_x = nn.Parameter(torch.tensor([[-1., 0., 1.],
                                                 [-2., 0., 2.],
                                                 [-1., 0., 1.]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[-1., -2., -1.],
                                                 [ 0.,  0.,  0.],
                                                 [ 1.,  2.,  1.]]).view(1, 1, 3, 3), requires_grad=False)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_feat):
        # x: [B, C, H, W]
        # edge_feat: [B, C_e, H_e, W_e]
        b, c, h, w = x.shape
        if edge_feat.shape[2:] != (h, w):
            edge_feat = F.interpolate(edge_feat, size=(h, w), mode='bilinear', align_corners=False)
            
        q = self.q_conv(x)
        k = self.k_conv(edge_feat)
        v = self.v_conv(edge_feat)
        
        # Tangent vector field [B, 2, H, W]
        t_vec = self.tangent_proj(edge_feat)
        tx = t_vec[:, 0:1, :, :]
        ty = t_vec[:, 1:2, :, :]
        
        # Spatial directional derivative of q along tangent
        q_mean = q.mean(dim=1, keepdim=True)
        dq_x = F.conv2d(q_mean, self.sobel_x, padding=1)
        dq_y = F.conv2d(q_mean, self.sobel_y, padding=1)
        tangent_alignment = torch.abs(tx * dq_x + ty * dq_y)
        tangent_weight = torch.sigmoid(tangent_alignment)
        
        # Anisotropic attention modulation
        sim = torch.sigmoid(q * k) * tangent_weight
        att_out = v * sim
        
        out = self.out_conv(x + self.gamma * att_out)
        return out


class LevelSetImplicitSurfaceDecoder(nn.Module):
    """
    Level-Set Implicit Surface Decoder (LS-Dec).
    Represents flood boundaries as the zero-level set of a signed distance function phi.
    Executes continuous differentiable level-set convection steps:
      phi_{k+1} = phi_k - dt * (V . grad(phi_k) - beta * Curvature(phi_k))
    smoothing geometric contours and guaranteeing topological consistency.
    """
    def __init__(self, in_channels, num_classes=2, num_steps=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_steps
        
        # Predicts initial level-set surface phi_0 [B, 1, H, W] and velocity field V [B, 2, H, W]
        self.surface_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1)  # channel 0: phi_0, channel 1-2: (Vx, Vy)
        )
        
        # Spatial finite difference filters for gradients
        dx = torch.tensor([[-0.5, 0., 0.5]]).view(1, 1, 1, 3)
        dy = torch.tensor([[-0.5], [0.], [0.5]]).view(1, 1, 3, 1)
        self.register_buffer('dx', dx)
        self.register_buffer('dy', dy)
        
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.05))
        
        # Final calibrated classification projection
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def _compute_derivatives(self, phi):
        eps = 1e-6
        # Finite difference gradient
        phi_x = F.conv2d(phi, self.dx, padding=(0, 1))
        phi_y = F.conv2d(phi, self.dy, padding=(1, 0))
        grad_norm = torch.sqrt(phi_x ** 2 + phi_y ** 2 + eps)
        
        # Normalized normal: (nx, ny)
        nx = phi_x / grad_norm
        ny = phi_y / grad_norm
        
        # Mean curvature kappa = div(n) = d(nx)/dx + d(ny)/dy
        nx_x = F.conv2d(nx, self.dx, padding=(0, 1))
        ny_y = F.conv2d(ny, self.dy, padding=(1, 0))
        curvature = nx_x + ny_y
        
        return phi_x, phi_y, curvature

    def forward(self, feat):
        # feat: [B, C, H, W]
        surf_init = self.surface_head(feat)
        phi = surf_init[:, 0:1, :, :]
        v_x = torch.tanh(surf_init[:, 1:2, :, :])
        v_y = torch.tanh(surf_init[:, 2:3, :, :])
        
        # Differentiable level-set convection evolution
        for _ in range(self.num_steps):
            phi_x, phi_y, curvature = self._compute_derivatives(phi)
            convection = v_x * phi_x + v_y * phi_y
            phi = phi - self.dt * (convection - self.beta * curvature)
            
        # Water probability prior is positive where phi > 0
        water_prob = torch.sigmoid(phi)
        land_prob = torch.sigmoid(-phi)
        
        fused = torch.cat([feat, water_prob, land_prob], dim=1)
        logits = self.cls_head(fused)
        return logits


class EdgeFlowNet(nn.Module):
    """
    EdgeFlow-Net: Edge-Vector Field Guided Implicit Level-Set Flood Segmentation Network.
    
    Trio Modules:
    1. SNDG: Sobel-NIR Directional Gradient Stream
    2. EVF-CA: Edge-Vector Field Cross-Attention
    3. LS-Dec: Level-Set Implicit Surface Decoder
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        # 1. Sobel-NIR Directional Gradient Stream
        self.sndg = SobelNIRDirectionalGradientStream(out_channels=c1)
        
        # Primary visual encoder stages
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU()
        )
        
        # Edge stream downsampling stages
        self.edge_down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        self.edge_down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU()
        )
        self.edge_down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU()
        )
        
        # 2. Edge-Vector Field Cross-Attention at stages 3 and 4
        self.evf_ca3 = EdgeVectorFieldCrossAttention(c3, c3)
        self.evf_ca4 = EdgeVectorFieldCrossAttention(c4, c4)
        
        # Decoder
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
            nn.Conv2d(c1 + c1 + c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )
        
        # 3. Level-Set Implicit Surface Decoder
        self.ls_dec = LevelSetImplicitSurfaceDecoder(c1, num_classes=num_classes, num_steps=2)

    def forward(self, x):
        # Input: [B, 4, H, W]
        # 1. Sobel-NIR Directional Gradient Stream
        edge_feat, _ = self.sndg(x)       # [B, c1, H, W]
        
        # Primary visual stream
        v1 = self.stem(x)                  # [B, c1, H, W]
        
        v2 = self.down1(v1)                # [B, c2, H/2, W/2]
        e2 = self.edge_down1(edge_feat)    # [B, c2, H/2, W/2]
        
        v3 = self.down2(v2)                # [B, c3, H/4, W/4]
        e3 = self.edge_down2(e2)           # [B, c3, H/4, W/4]
        v3 = self.evf_ca3(v3, e3)
        
        v4 = self.down3(v3)                # [B, c4, H/8, W/8]
        e4 = self.edge_down3(e3)           # [B, c4, H/8, W/8]
        v4 = self.evf_ca4(v4, e4)
        
        # Decoder
        d3 = self.up3(v4)
        d3 = self.dec3(torch.cat([d3, v3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, v2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, v1, edge_feat], dim=1))
        
        # 3. Level-Set Implicit Surface Decoder
        logits = self.ls_dec(d1)
        return logits
