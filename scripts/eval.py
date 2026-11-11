import sys
import os
import argparse
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.sen1floods11 import Sen1Floods11Dataset
from data.window_generator import generate_quadrants, stitch_quadrants
from metrics.evaluator import FloodMetricEvaluator
from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormer
from models.prithvi_wrapper import PrithviAdapter
from models.dofa_wrapper import DOFAAdapter
from models.manet import MANet
from models.attentive_unet import AttentiveUNet
from models.pan import PAN
from models.farseg import FarSeg
from models.hydropins_net import HydroPINSNet
from models.wavelet_procanet_v2 import WaveletProCANetV2
from models.edgeflow_net import EdgeFlowNet
from models.stripcross_net_v2 import StripCrossNetV2
from models.wavekan_hydronet import WaveKANHydroNet
from models.spectralgate_net import SpectralGateNet
from models.turbidwater_net import TurbidWaterNet
from models.shadowfree_net import ShadowFreeNet

import models

def build_model(model_name, modality, num_classes=2):
    if modality == 's1':
        in_channels = 2
    elif modality in ['s2_4ch', '4ch']:
        in_channels = 4
    elif modality == 's2':
        in_channels = 4
    elif modality == 's1s2':
        in_channels = 15
    else:
        in_channels = 4
        
    canonical = model_name.lower().replace('_', '').replace('-', '')
    for export_name in models.__all__:
        if export_name.lower().replace('_', '') == canonical:
            cls = getattr(models, export_name)
            return cls(in_channels=in_channels, num_classes=num_classes)
            
    aliases = {
        'attentiveunet': models.AttentiveUNet,
        'attentive_s1': models.AttentiveUNet,
        'proca': models.ProCANet,
        'shadowfree': models.ShadowFreeNet,
        'unet++': models.UNetPlusPlus,
        'hrnet': models.HRNetV2,
        'swinunet': models.SwinUnet,
    }
    if canonical in aliases:
        return aliases[canonical](in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: '{model_name}'. Available: {sorted(set(models.__all__))}")

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    dataset = Sen1Floods11Dataset(
        root_dir=args.data_dir,
        split=args.split,
        modality=args.modality,
        crop_size=None
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(args.model, args.modality)
    
    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        
    model.to(device)
    model.eval()
    
    evaluator = FloodMetricEvaluator(ignore_index=255)
    
    with torch.no_grad():
        for img, mask in loader:
            img = img.squeeze(0).to(device)
            mask = mask.squeeze(0).to(device)
            
            if args.use_quadrants:
                quadrants = generate_quadrants(img)
                quad_preds = []
                for q in quadrants:
                    q_in = q.unsqueeze(0)
                    out = model(q_in).squeeze(0)
                    quad_preds.append(out)
                pred_chip = stitch_quadrants(quad_preds)
                pred_classes = torch.argmax(pred_chip, dim=0)
                evaluator.update(pred_classes.cpu().numpy(), mask.cpu().numpy())
            else:
                img_in = img.unsqueeze(0)
                out = model(img_in).squeeze(0)
                pred_classes = torch.argmax(out, dim=0)
                evaluator.update(pred_classes.cpu().numpy(), mask.cpu().numpy())
                
    results = evaluator.compute()
    return results

def get_default_data_dir():
    candidates = [
        '/kaggle/input/datasets/doantrongthai/sen1floods11',
        '/kaggle/input/sen1floods11',
        '/kaggle/working/data',
        r'D:\Downloads\remote_sensing\sen1floods11_256x256',
        r'D:\Downloads\remote_sensing\sen1floods11_hand_labeled',
        './data'
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return '/kaggle/input/datasets/doantrongthai/sen1floods11'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=get_default_data_dir())
    parser.add_argument('--split', type=str, default='test', choices=['test', 'bolivia', 'valid'])
    parser.add_argument('--modality', type=str, default='s1', choices=['s1', 's2', 's1s2'])
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeplabv3plus', 'segformer', 'prithvi', 'dofa'])
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--use_quadrants', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_json', type=str, default='')
    
    args = parser.parse_args()
    res = evaluate(args)
    
    print(json.dumps(res, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(res, f, indent=2)
