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

def build_model(model_name, modality, num_classes=2):
    in_channels = 2 if modality == 's1' else (13 if modality == 's2' else 15)
    name = model_name.lower()
    if name == 'unet':
        return UNet(in_channels=in_channels, num_classes=num_classes)
    elif name == 'deeplabv3plus':
        return DeepLabV3Plus(in_channels=in_channels, num_classes=num_classes)
    elif name == 'segformer':
        return SegFormer(in_channels=in_channels, num_classes=num_classes)
    elif name == 'prithvi':
        return PrithviAdapter(in_channels=in_channels, num_classes=num_classes)
    elif name == 'dofa':
        return DOFAAdapter(in_channels=in_channels, num_classes=num_classes)
    elif name == 'manet':
        return MANet(in_channels=in_channels, num_classes=num_classes)
    elif name in ['attentive_unet', 'attentiveunet', 'attentive_s1']:
        return AttentiveUNet(in_channels=in_channels, num_classes=num_classes)
    elif name == 'pan':
        return PAN(in_channels=in_channels, num_classes=num_classes)
    elif name == 'farseg':
        return FarSeg(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

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
