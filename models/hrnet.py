import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, num_blocks, num_channels, block=BasicBlock):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            self._make_branch(i, block, num_blocks[i], num_channels[i])
            for i in range(num_branches)
        ])
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels)
        self.relu = nn.ReLU(inplace=True)

    def _make_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(num_blocks):
            layers.append(block(num_channels, num_channels))
        return nn.Sequential(*layers)

    def _make_fuse_layers(self, num_branches, num_channels):
        fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    # Upsampling from lower resolution j to higher resolution i
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i])
                    ))
                elif j < i:
                    # Downsampling from higher resolution j to lower resolution i
                    down_layers = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            down_layers.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels[i])
                            ))
                        else:
                            down_layers.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*down_layers))
                else:
                    fuse_layer.append(None)
            fuse_layers.append(fuse_layer)
        return fuse_layers

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    transformed = self.fuse_layers[i][j](x[j])
                    y = y + F.interpolate(transformed, size=x[i].shape[2:], mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class HRNetV2(nn.Module):
    """
    HRNetV2-W18 for Semantic Segmentation (TPAMI 2020 / CVPR 2019).
    Maintains parallel multi-resolution convolutions with continuous cross-resolution fusions.
    """
    def __init__(self, in_channels=4, num_classes=2, base_channels=18):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: Bottlenecks at 1/4 resolution
        stage1_downsample = nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=stage1_downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        # Transition 1
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, c * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 2),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 2: 2 branches
        self.stage2 = HighResolutionModule(
            num_branches=2,
            num_blocks=[2, 2],
            num_channels=[c, c * 2]
        )

        # Transition 2
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 4),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 3: 3 branches
        self.stage3 = HighResolutionModule(
            num_branches=3,
            num_blocks=[2, 2, 2],
            num_channels=[c, c * 2, c * 4]
        )

        # Transition 3
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(c * 4, c * 8, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 8),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage 4: 4 branches
        self.stage4 = HighResolutionModule(
            num_branches=4,
            num_blocks=[2, 2, 2, 2],
            num_channels=[c, c * 2, c * 4, c * 8]
        )

        # HRNetV2 Representation Head
        total_channels = c + c * 2 + c * 4 + c * 8  # 18 + 36 + 72 + 144 = 270
        self.head = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, 1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(total_channels, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # Stage 1 -> Stage 2
        x_list = [self.transition1[0](x), self.transition1[1](x)]
        y_list = self.stage2(x_list)

        # Stage 2 -> Stage 3
        x_list = [
            self.transition2[0](y_list[0]),
            self.transition2[1](y_list[1]),
            self.transition2[2](y_list[1])
        ]
        y_list = self.stage3(x_list)

        # Stage 3 -> Stage 4
        x_list = [
            self.transition3[0](y_list[0]),
            self.transition3[1](y_list[1]),
            self.transition3[2](y_list[2]),
            self.transition3[3](y_list[2])
        ]
        y_list = self.stage4(x_list)

        # Upsample all stages to branch 0 (1/4 resolution)
        out0 = y_list[0]
        out_size = out0.shape[2:]
        out1 = F.interpolate(y_list[1], size=out_size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(y_list[2], size=out_size, mode='bilinear', align_corners=True)
        out3 = F.interpolate(y_list[3], size=out_size, mode='bilinear', align_corners=True)

        concat_feats = torch.cat([out0, out1, out2, out3], dim=1)
        logits = self.head(concat_feats)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)

# Alias
HRNet = HRNetV2
