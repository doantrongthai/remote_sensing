import torch
import torch.nn as nn
import torch.nn.functional as F

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=2, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=4, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)

class SegFormerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, sr_ratio=1, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x))
        return x

class SegFormer(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        
        self.block1 = SegFormerBlock(dim=embed_dims[0], num_heads=num_heads[0], sr_ratio=sr_ratios[0])
        self.block2 = SegFormerBlock(dim=embed_dims[1], num_heads=num_heads[1], sr_ratio=sr_ratios[1])
        self.block3 = SegFormerBlock(dim=embed_dims[2], num_heads=num_heads[2], sr_ratio=sr_ratios[2])
        self.block4 = SegFormerBlock(dim=embed_dims[3], num_heads=num_heads[3], sr_ratio=sr_ratios[3])
        
        dec_dim = 128
        self.linear_c4 = nn.Conv2d(embed_dims[3], dec_dim, 1)
        self.linear_c3 = nn.Conv2d(embed_dims[2], dec_dim, 1)
        self.linear_c2 = nn.Conv2d(embed_dims[1], dec_dim, 1)
        self.linear_c1 = nn.Conv2d(embed_dims[0], dec_dim, 1)
        
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(dec_dim * 4, dec_dim, 1, bias=False),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(dec_dim, num_classes, 1)

    def forward(self, x):
        orig_size = x.shape[-2:]
        b = x.shape[0]
        
        c1, h1, w1 = self.patch_embed1(x)
        c1 = self.block1(c1, h1, w1).permute(0, 2, 1).reshape(b, -1, h1, w1)
        
        c2, h2, w2 = self.patch_embed2(c1)
        c2 = self.block2(c2, h2, w2).permute(0, 2, 1).reshape(b, -1, h2, w2)
        
        c3, h3, w3 = self.patch_embed3(c2)
        c3 = self.block3(c3, h3, w3).permute(0, 2, 1).reshape(b, -1, h3, w3)
        
        c4, h4, w4 = self.patch_embed4(c3)
        c4 = self.block4(c4, h4, w4).permute(0, 2, 1).reshape(b, -1, h4, w4)
        
        _c4 = F.interpolate(self.linear_c4(c4), size=(h1, w1), mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=(h1, w1), mode='bilinear', align_corners=False)
        _c2 = F.interpolate(self.linear_c2(c2), size=(h1, w1), mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1)
        
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        out = self.classifier(_c)
        return F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
