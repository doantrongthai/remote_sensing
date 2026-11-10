import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionAttentionModule(nn.Module):
    """
    Position Attention Module (PAM) from DANet (CVPR 2019).
    Computes spatial relationships between any two positions.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (CAM) from DANet (CVPR 2019).
    Computes channel dependencies across all feature channels.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        
        proj_value = x.view(B, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class BasicResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class DANetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        inter_channels = in_channels // 4

        # PAM branch
        self.conv_pam1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.pam = PositionAttentionModule(inter_channels)
        self.conv_pam2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # CAM branch
        self.conv_cam1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.cam = ChannelAttentionModule(inter_channels)
        self.conv_cam2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion and classification
        self.cls = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feat_pam = self.conv_pam2(self.pam(self.conv_pam1(x)))
        feat_cam = self.conv_cam2(self.cam(self.conv_cam1(x)))
        fused = feat_pam + feat_cam
        return self.cls(fused)

class DANet(nn.Module):
    """
    DANet: Dual Attention Network for Scene Segmentation (CVPR 2019).
    Parallel Position Attention Module (PAM) and Channel Attention Module (CAM).
    """
    def __init__(self, in_channels=4, num_classes=2, base_filters=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        f = base_filters

        # ResNet-style backbone with output stride 8
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True)
        )
        
        # Layer 1: stride 1 (1/2 scale)
        self.layer1 = nn.Sequential(
            BasicResBlock(f, f, stride=1),
            BasicResBlock(f, f, stride=1)
        )
        # Layer 2: stride 2 (1/4 scale)
        self.layer2 = nn.Sequential(
            BasicResBlock(f, f * 2, stride=2),
            BasicResBlock(f * 2, f * 2, stride=1)
        )
        # Layer 3: stride 2 (1/8 scale)
        self.layer3 = nn.Sequential(
            BasicResBlock(f * 2, f * 4, stride=2),
            BasicResBlock(f * 4, f * 4, stride=1)
        )
        # Layer 4: stride 1, dilation 2 (1/8 scale)
        self.layer4 = nn.Sequential(
            BasicResBlock(f * 4, f * 8, stride=1, dilation=2),
            BasicResBlock(f * 8, f * 8, stride=1, dilation=2)
        )

        self.head = DANetHead(f * 8, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        logits = self.head(x)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
