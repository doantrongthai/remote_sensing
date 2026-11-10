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

def get_model(name, in_channels, num_classes=2):
    n = name.lower()
    if n == 'unet':
        return UNet(in_channels=in_channels, num_classes=num_classes)
    elif n == 'deeplabv3plus':
        return DeepLabV3Plus(in_channels=in_channels, num_classes=num_classes)
    elif n == 'segformer':
        return SegFormer(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

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
    in_channels = 2 if args.modality == 's1' else (13 if args.modality == 's2' else 15)
    model = get_model(model_name, in_channels=in_channels, num_classes=2).to(device)
    
    train_ds = Sen1Floods11Dataset(args.data_dir, split='train', modality=args.modality)
    val_ds = Sen1Floods11Dataset(args.data_dir, split='valid', modality=args.modality)
    test_ds = Sen1Floods11Dataset(args.data_dir, split='test', modality=args.modality)
    bolivia_ds = Sen1Floods11Dataset(args.data_dir, split='bolivia', modality=args.modality)
    
    num_workers = args.num_workers
    pin_mem = torch.cuda.is_available()
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    bolivia_loader = DataLoader(bolivia_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    
    criterion = BCEDiceLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = get_grad_scaler()
    autocast_ctx = get_autocast
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt = os.path.join(args.save_dir, f"{model_name}_{args.modality}_best.pth")
    best_val_iou = -1.0
    
    print(f"\n--- Training {model_name.upper()} | Modality: {args.modality.upper()} | Epochs: {args.epochs} | Batch: {args.batch_size} ---")
    
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
        'best_val_water_iou': round(best_val_iou, 4),
        'test_water_iou': round(test_res['water_iou'], 4),
        'test_m_iou': round(test_res['m_iou'], 4),
        'bolivia_water_iou': round(bolivia_res['water_iou'], 4),
        'bolivia_m_iou': round(bolivia_res['m_iou'], 4)
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=get_default_data_dir(),
        help='Root directory containing LabelHand, S1Hand, S2Hand, splits'
    )
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'deeplabv3plus', 'segformer', 'all'])
    parser.add_argument('--modality', type=str, default='s1', choices=['s1', 's2', 's1s2'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print(f"Device: {device} | Data: {args.data_dir}")
    label_check = os.path.join(args.data_dir, 'LabelHand')
    print(f"Data verification -> LabelHand: {label_check} (Exists: {os.path.exists(label_check)})")
    
    models_to_run = ['unet', 'deeplabv3plus', 'segformer'] if args.model == 'all' else [args.model]
    all_summaries = []
    
    for m in models_to_run:
        res = train_single_model(m, args, device)
        all_summaries.append(res)
        
    print("\n" + "="*80)
    print(f"{'BENCHMARK SUMMARY':^80}")
    print("="*80)
    print(f"{'Model':<15} | {'Modality':<8} | {'Val Water IoU':<14} | {'Test Water IoU':<15} | {'Bolivia Water IoU':<17}")
    print("-" * 80)
    for s in all_summaries:
        print(f"{s['model'].upper():<15} | {s['modality'].upper():<8} | {s['best_val_water_iou']:<14} | {s['test_water_iou']:<15} | {s['bolivia_water_iou']:<17}")
    print("="*80 + "\n")
    
    out_json = os.path.join(args.save_dir, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Results saved to: {out_json}")

if __name__ == '__main__':
    main()
