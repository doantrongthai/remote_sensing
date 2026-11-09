import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveMoEAdapter(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.wave_mlp = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

    def forward(self, x, wavelengths=None):
        feat = self.conv_in(x)
        if wavelengths is not None:
            w_emb = self.wave_mlp(wavelengths.unsqueeze(-1)).mean(dim=1, keepdim=True)
            w_emb = w_emb.unsqueeze(-1)
            feat = feat * (1.0 + torch.sigmoid(w_emb))
        return feat

class DOFAAdapter(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, embed_dim=256, depth=4):
        super().__init__()
        self.adapter = WaveMoEAdapter(in_channels, embed_dim)
        self.patch_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=16, stride=16)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=16, stride=16),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x, wavelengths=None):
        orig_size = x.shape[-2:]
        feat = self.adapter(x, wavelengths)
        patches = self.patch_embed(feat)
        b, c, h, w = patches.shape
        
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        encoded_feat = tokens.transpose(1, 2).reshape(b, c, h, w)
        
        out = self.head(encoded_feat)
        if out.shape[-2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
        return out
