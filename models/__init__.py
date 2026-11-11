# Proposed Champion Architecture (Shadow-Water Physics-Guided)
from models.shadowfree_net import ShadowFreeNet
from models.shadowfree_net_v2 import ShadowFreeNetV2

# Direct SOTA Baseline (Feliren et al., IEEE GRSL 2025)
from models.procanet import ProCANet

# Classical Semantic Segmentation Baselines
from models.unet import UNet
from models.unetplusplus import UNetPlusPlus, UNet2Plus
from models.deeplabv3plus import DeepLabV3Plus
from models.pspnet import PSPNet
from models.hrnet import HRNetV2, HRNet

# Remote Sensing & Attention Baselines
from models.farseg import FarSeg
from models.attentive_unet import AttentiveUNet
from models.manet import MANet
from models.pan import PAN

# Vision Transformer Baselines
from models.segformer import SegFormer
from models.transunet import TransUNet
from models.swin_unet import SwinUnet

__all__ = [
    # Proposed
    'ShadowFreeNet',
    'ShadowFreeNetV2',
    # SOTA Baseline
    'ProCANet',
    # Classical Baselines
    'UNet',
    'UNetPlusPlus',
    'UNet2Plus',
    'DeepLabV3Plus',
    'PSPNet',
    'HRNetV2',
    'HRNet',
    # RS & Attention Baselines
    'FarSeg',
    'AttentiveUNet',
    'MANet',
    'PAN',
    # Transformer Baselines
    'SegFormer',
    'TransUNet',
    'SwinUnet',
]
