import torch
import torch.nn as nn
import torch.nn.functional as F

class PrithviAdapter(nn.Module):
    def __init__(self, in_channels=6, num_classes=2, embed_dim=768, depth=4):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = 16
        
        if in_channels != 6:
            self.input_adapter = nn.Conv2d(in_channels, 6, kernel_size=1)
        else:
            self.input_adapter = nn.Identity()
            
        self.patch_proj = nn.Conv2d(6, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        orig_size = x.shape[-2:]
        if x.shape[1] == 13:
            s2_indices = [1, 2, 3, 8, 11, 12]
            x = x[:, s2_indices, :, :]
        else:
            x = self.input_adapter(x)
            
        patches = self.patch_proj(x)
        b, c, h, w = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        
        encoded = self.transformer(tokens)
        feature_map = encoded.transpose(1, 2).reshape(b, c, h, w)
        
        logits = self.decoder(feature_map)
        if logits.shape[-2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
        return logits
