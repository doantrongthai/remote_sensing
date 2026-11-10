import sys
import os
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def get_grad_scaler():
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        except Exception:
            pass
    from torch.cuda.amp import GradScaler
    return GradScaler(enabled=torch.cuda.is_available())

def get_autocast():
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        try:
            return torch.amp.autocast('cuda', enabled=torch.cuda.is_available())
        except Exception:
            pass
    from torch.cuda.amp import autocast
    return autocast(enabled=torch.cuda.is_available())

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.sen1floods11 import Sen1Floods11Dataset
from losses.losses import BCEDiceLoss
from metrics.evaluator import FloodMetricEvaluator
from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormer
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

def get_model(name, in_channels, num_classes=2):
    canonical = name.lower().replace('_', '').replace('-', '')
    # Check all exported model classes
    for export_name in models.__all__:
        if export_name.lower().replace('_', '') == canonical:
            cls = getattr(models, export_name)
            return cls(in_channels=in_channels, num_classes=num_classes)
            
    # Common short aliases
    aliases = {
        'attentiveunet': models.AttentiveUNet,
        'attentive_s1': models.AttentiveUNet,
        'proca': models.ProCANet,
        'proca_edge': models.ProCAEdgeNet,
        'hydro_proca': models.HydroProCANet,
        'wavelet_proca': models.WaveletProCANet,
        'strip_proca': models.StripProCANet,
        'bmg_proca': models.BMGProCANet,
        'kan_proca': models.KANProCANet,
        'hydropins': models.HydroPINSNet,
        'edgeflow': models.EdgeFlowNet,
        'stripcross': models.StripCrossNetV2,
        'wavekan': models.WaveKANHydroNet,
        'spectralgate': models.SpectralGateNet,
        'turbidwater': models.TurbidWaterNet,
        'shadowfree': models.ShadowFreeNet,
        'unet++': models.UNetPlusPlus,
        'hrnet': models.HRNetV2,
    }
    if canonical in aliases:
        return aliases[canonical](in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: '{name}'. Available models: {sorted(models.__all__)}")

def run_eval(model, loader, device):
    model.eval()
    ev = FloodMetricEvaluator(ignore_index=255)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs).argmax(dim=1)
            ev.update(preds.cpu().numpy(), masks.cpu().numpy())
    return ev.compute()

def train_single_model(model_name, args, device):
    if args.modality == 's1':
        in_channels = 2
    elif args.modality in ['s2', 's2_4ch']:
        in_channels = args.bands
    elif args.modality == 's1s2':
        in_channels = 2 + args.bands
    else:
        in_channels = args.bands

    model = get_model(model_name, in_channels=in_channels, num_classes=2).to(device)
    
    train_crop = args.crop_size if (args.crop_size and args.crop_size > 0) else None
    train_ds = Sen1Floods11Dataset(args.data_dir, split='train', modality=args.modality, bands=args.bands, crop_size=train_crop)
    val_ds = Sen1Floods11Dataset(args.data_dir, split='valid', modality=args.modality, bands=args.bands, crop_size=None)
    test_ds = Sen1Floods11Dataset(args.data_dir, split='test', modality=args.modality, bands=args.bands, crop_size=None)
    bolivia_ds = Sen1Floods11Dataset(args.data_dir, split='bolivia', modality=args.modality, bands=args.bands, crop_size=None)
    
    num_workers = args.num_workers
    pin_mem = torch.cuda.is_available()
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    bolivia_loader = DataLoader(bolivia_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    
    criterion = BCEDiceLoss(ignore_index=255)
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
    if args.scheduler == 'cosine_restarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
    scaler = get_grad_scaler()
    autocast_ctx = get_autocast
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt = os.path.join(args.save_dir, f"{model_name}_{args.modality}_best.pth")
    best_val_iou = -1.0
    
    print(f"\n--- Training {model_name.upper()} | Modality: {args.modality.upper()} ({args.bands} bands: RGB+NIR) | Epochs: {args.epochs} | Batch: {args.batch_size} ---")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast_ctx():
                preds = model(imgs)
                loss = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        val_res = run_eval(model, val_loader, device)
        val_iou = val_res['water_iou']
        ep_time = round(time.time() - t0, 1)
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_ckpt)
            flag = "*"
        else:
            flag = " "
            
        print(f"Ep {epoch:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} (mIoU: {val_res['m_iou']:.4f}) {flag} | {ep_time}s")
        
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_res = run_eval(model, test_loader, device)
    bolivia_res = run_eval(model, bolivia_loader, device)
    
    summary = {
        'model': model_name,
        'modality': args.modality,
        'bands': args.bands,
        'best_val_water_iou': round(best_val_iou, 4),
        'test_accuracy': round(test_res.get('accuracy', 0.0), 4),
        'test_f1': round(test_res.get('f1_score', 0.0), 4),
        'test_water_iou': round(test_res['water_iou'], 4),
        'test_m_iou': round(test_res['m_iou'], 4),
        'bolivia_water_iou': round(bolivia_res['water_iou'], 4),
        'bolivia_f1': round(bolivia_res.get('f1_score', 0.0), 4),
        'bolivia_m_iou': round(bolivia_res['m_iou'], 4)
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Sen1Floods11 Remote Sensing Benchmark Engine")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=get_default_data_dir(),
        help='Root directory containing LabelHand, S1Hand, S2Hand, splits'
    )
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument(
        '--model', 
        type=str, 
        default='all', 
        help="Model name ('all', 'unet', 'deeplabv3plus', 'segformer', 'attentive_unet', 'manet', 'pan', 'farseg', or comma-separated list)"
    )
    parser.add_argument('--modality', type=str, default='s2', choices=['s1', 's2', 's2_4ch', 's1s2'])
    parser.add_argument('--bands', type=int, default=4, choices=[4, 13], help='4 for RGB+NIR, 13 for full Sentinel-2')
    parser.add_argument('--crop_size', type=int, default=0, help='Patch crop size (0 for full 256x256)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'cosine_restarts'], help='Scheduler')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print(f"Device: {device} | Data: {args.data_dir} | Modality: {args.modality} ({args.bands} bands) | LR: {args.lr}")
    label_check = os.path.join(args.data_dir, 'LabelHand')
    print(f"Data verification -> LabelHand: {label_check} (Exists: {os.path.exists(label_check)})")
    
    if args.model.lower() == 'all':
        models_to_run = ['unet', 'deeplabv3plus', 'segformer', 'attentive_unet', 'manet', 'pan', 'farseg']
    elif ',' in args.model:
        models_to_run = [m.strip().lower() for m in args.model.split(',') if m.strip()]
    else:
        models_to_run = [args.model.lower()]
    all_summaries = []
    
    for m in models_to_run:
        res = train_single_model(m, args, device)
        all_summaries.append(res)
        
    print("\n" + "="*95)
    print(f"{'BENCHMARK SUMMARY (SEN1FLOODS11)':^95}")
    print("="*95)
    print(f"{'Model':<16} | {'Modality':<8} | {'Bands':<6} | {'Val IoU':<10} | {'Test IoU':<10} | {'Bolivia IoU':<12} | {'Test F1':<10} | {'Test Acc':<10}")
    print("-" * 95)
    for s in all_summaries:
        m_name = s['model'].upper()
        print(f"{m_name:<16} | {s['modality'].upper():<8} | {s['bands']:<6} | {s['best_val_water_iou']:<10.4f} | {s['test_water_iou']:<10.4f} | {s['bolivia_water_iou']:<12.4f} | {s['test_f1']:<10.4f} | {s['test_accuracy']:<10.4f}")
    print("="*95 + "\n")
    
    out_json = os.path.join(args.save_dir, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Results saved to: {out_json}")

if __name__ == '__main__':
    main()
