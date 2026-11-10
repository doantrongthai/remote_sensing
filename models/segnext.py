import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCA(nn.Module):
    """
    Multi-Scale Convolutional Attention (MSCA) Module.
    SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation (NeurIPS 2022).
    """
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        
        # Branch 1
        self.conv0_1 = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim)
        
        # Branch 2
        self.conv1_1 = nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=dim)
        
        # Branch 3
        self.conv2_1 = nn.Conv2d(dim, dim, kernel_size=(1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, kernel_size=(21, 1), padding=(10, 0), groups=dim)
        
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return u * attn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MSCABlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MSCA(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=4, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class NMFHamburger(nn.Module):
    """
    Hamburger module using Non-negative Matrix Factorization (NMF) concept
    as presented in LightHamHead (SegNeXt / NeurIPS 2022).
    """
    def __init__(self, dim, rank=32, steps=3):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.steps = steps
        self.bases = nn.Parameter(torch.randn(1, dim, rank))
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_mat = F.relu(x, inplace=False).view(B, C, N)
        
        D = torch.softmax(self.bases.expand(B, -1, -1), dim=1)
        with torch.no_grad():
            C_mat = torch.softmax(torch.bmm(D.transpose(1, 2), x_mat), dim=1)
            eps = 1e-6
            for _ in range(self.steps):
                Dt = D.transpose(1, 2)
                DtX = torch.bmm(Dt, x_mat)
                DtD = torch.bmm(Dt, D)
                DtDC = torch.bmm(DtD, C_mat) + eps
                C_mat = C_mat * (DtX / DtDC)

        recon = torch.bmm(D, C_mat).view(B, C, H, W)
        recon = self.bn(self.proj(recon))
        return x + recon

class LightHamHead(nn.Module):
    """
    LightHamHead decoder: projects multi-scale features, fuses them,
    processes via Hamburger module, and outputs segmentation logits.
    """
    def __init__(self, in_channels_list, ham_dim=128, num_classes=2):
        super().__init__()
        self.proj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, ham_dim, 1, bias=False),
                nn.BatchNorm2d(ham_dim),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        
        self.squeeze = nn.Sequential(
            nn.Conv2d(ham_dim * len(in_channels_list), ham_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ham_dim),
            nn.ReLU(inplace=True)
        )
        self.hamburger = NMFHamburger(ham_dim, rank=32, steps=3)
        self.cls_head = nn.Sequential(
            nn.Conv2d(ham_dim, ham_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ham_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(ham_dim, num_classes, 1)
        )

    def forward(self, features, out_size):
        ref_h, ref_w = features[0].shape[2:]
        proj_feats = []
        for feat, proj in zip(features, self.proj_convs):
            p = proj(feat)
            if p.shape[2:] != (ref_h, ref_w):
                p = F.interpolate(p, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            proj_feats.append(p)
            
        fused = self.squeeze(torch.cat(proj_feats, dim=1))
        fused = self.hamburger(fused)
        logits = self.cls_head(fused)
        return F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)

class SegNeXt(nn.Module):
    """
    SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation (NeurIPS 2022).
    Multi-Scale Convolutional Attention (MSCA) backbone with LightHamHead.
    """
    def __init__(self, in_channels=4, num_classes=2, embed_dims=[32, 64, 160, 256], depths=[2, 2, 4, 2]):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stages
        self.stage1 = nn.Sequential(*[MSCABlock(embed_dims[0]) for _ in range(depths[0])])
        self.stage2 = nn.Sequential(*[MSCABlock(embed_dims[1]) for _ in range(depths[1])])
        self.stage3 = nn.Sequential(*[MSCABlock(embed_dims[2]) for _ in range(depths[2])])
        self.stage4 = nn.Sequential(*[MSCABlock(embed_dims[3]) for _ in range(depths[3])])

        # LightHamHead decoder
        self.decode_head = LightHamHead(in_channels_list=embed_dims, ham_dim=128, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        
        # Stage 1: 1/4 resolution
        x1 = self.patch_embed1(x)
        x1 = self.stage1(x1)
        
        # Stage 2: 1/8 resolution
        x2 = self.patch_embed2(x1)
        x2 = self.stage2(x2)
        
        # Stage 3: 1/16 resolution
        x3 = self.patch_embed3(x2)
        x3 = self.stage3(x3)
        
        # Stage 4: 1/32 resolution
        x4 = self.patch_embed4(x3)
        x4 = self.stage4(x4)
        
        out = self.decode_head([x1, x2, x3, x4], (h, w))
        return out
