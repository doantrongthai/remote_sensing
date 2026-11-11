import sys
import os
import time
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
import models

def get_model(name, in_channels, num_classes=2):
    canonical = name.lower().replace('_', '').replace('-', '')
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
    raise ValueError(f"Unknown model: '{name}'. Available models: {sorted(set(models.__all__))}")

def run_eval(model, loader, device):
    model.eval()
    ev = FloodMetricEvaluator(ignore_index=255)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            if isinstance(out, dict):
                logits = out['out']
            elif isinstance(out, (list, tuple)):
                logits = out[0]
            else:
                logits = out
            preds = logits.argmax(dim=1)
            ev.update(preds.cpu().numpy(), masks.cpu().numpy())
    return ev.compute()

def train_single_run(model_name, seed, args, device):
    set_seed(seed)
    
    if args.modality == 's1':
        in_channels = 2
    elif args.modality in ['s2', 's2_4ch']:
        in_channels = args.bands
    elif args.modality == 's1s2':
        in_channels = 2 + args.bands
    else:
        in_channels = args.bands

    model = get_model(model_name, in_channels=in_channels, num_classes=2).to(device)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    train_crop = args.crop_size if (args.crop_size and args.crop_size > 0) else None
    train_ds = Sen1Floods11Dataset(args.data_dir, split='train', modality=args.modality, bands=args.bands, crop_size=train_crop)
    val_ds = Sen1Floods11Dataset(args.data_dir, split='valid', modality=args.modality, bands=args.bands, crop_size=None)
    test_ds = Sen1Floods11Dataset(args.data_dir, split='test', modality=args.modality, bands=args.bands, crop_size=None)
    bolivia_ds = Sen1Floods11Dataset(args.data_dir, split='bolivia', modality=args.modality, bands=args.bands, crop_size=None)
    
    g = torch.Generator()
    g.manual_seed(seed)
    num_workers = args.num_workers
    pin_mem = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_mem,
        worker_init_fn=seed_worker, generator=g
    )
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
    best_ckpt = os.path.join(args.save_dir, f"{model_name}_{args.modality}_seed{seed}_best.pth")
    best_val_iou = -1.0
    
    print(f"\n>>> Model: {model_name.upper()} | Seed: {seed} | Params: {params_m:.2f}M | Epochs: {args.epochs} | Batch: {args.batch_size}")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast_ctx():
                out = model(imgs)
                if isinstance(out, dict):
                    logits = out['out']
                elif isinstance(out, (list, tuple)):
                    logits = out[0]
                else:
                    logits = out
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
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
            
        print(f"  [Seed {seed}] Ep {epoch:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} (mIoU: {val_res['m_iou']:.4f}) {flag} | {ep_time}s")
        
    # Evaluate best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_res = run_eval(model, test_loader, device)
    bolivia_res = run_eval(model, bolivia_loader, device)
    
    run_summary = {
        'seed': seed,
        'model': model_name,
        'modality': args.modality,
        'bands': args.bands,
        'params_M': round(params_m, 2),
        'best_val_water_iou': round(best_val_iou, 4),
        'test_water_iou': round(test_res['water_iou'], 4),
        'bolivia_water_iou': round(bolivia_res['water_iou'], 4),
        'test_f1': round(test_res.get('f1_score', 0.0), 4),
        'test_accuracy': round(test_res.get('accuracy', 0.0), 4),
        'test_m_iou': round(test_res['m_iou'], 4),
        'bolivia_f1': round(bolivia_res.get('f1_score', 0.0), 4),
        'bolivia_m_iou': round(bolivia_res['m_iou'], 4)
    }
    
    # Cleanup memory
    del model, optimizer, scheduler, train_loader, val_loader, test_loader, bolivia_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return run_summary

def train_model_multiseed(model_name, seeds, args, device):
    seed_runs = []
    for s in seeds:
        res = train_single_run(model_name, s, args, device)
        seed_runs.append(res)
        
    val_ious = [r['best_val_water_iou'] for r in seed_runs]
    test_ious = [r['test_water_iou'] for r in seed_runs]
    bolivia_ious = [r['bolivia_water_iou'] for r in seed_runs]
    test_f1s = [r['test_f1'] for r in seed_runs]
    test_accs = [r['test_accuracy'] for r in seed_runs]
    
    aggregate = {
        'model': model_name,
        'modality': args.modality,
        'bands': args.bands,
        'params_M': seed_runs[0]['params_M'],
        'seeds': seeds,
        'num_seeds': len(seeds),
        'mean': {
            'val_iou': round(float(np.mean(val_ious)), 4),
            'test_iou': round(float(np.mean(test_ious)), 4),
            'bolivia_iou': round(float(np.mean(bolivia_ious)), 4),
            'test_f1': round(float(np.mean(test_f1s)), 4),
            'test_accuracy': round(float(np.mean(test_accs)), 4)
        },
        'std': {
            'val_iou': round(float(np.std(val_ious)), 4),
            'test_iou': round(float(np.std(test_ious)), 4),
            'bolivia_iou': round(float(np.std(bolivia_ious)), 4),
            'test_f1': round(float(np.std(test_f1s)), 4),
            'test_accuracy': round(float(np.std(test_accs)), 4)
        },
        'seed_runs': seed_runs
    }
    
    # Print Multi-seed Model Summary
    print("\n" + "="*80)
    print(f"MULTISEED SUMMARY: {model_name.upper()} ({len(seeds)} Seeds: {seeds})")
    print("="*80)
    print(f"{'Seed':<6} | {'Val IoU':<10} | {'Test IoU':<10} | {'Bolivia IoU':<12} | {'Test F1':<10} | {'Test Acc':<10}")
    print("-" * 80)
    for r in seed_runs:
        print(f"{r['seed']:<6} | {r['best_val_water_iou']:<10.4f} | {r['test_water_iou']:<10.4f} | {r['bolivia_water_iou']:<12.4f} | {r['test_f1']:<10.4f} | {r['test_accuracy']:<10.4f}")
    print("-" * 80)
    m = aggregate['mean']
    s = aggregate['std']
    print(f"MEAN   | {m['val_iou']:<10.4f} | {m['test_iou']:<10.4f} | {m['bolivia_iou']:<12.4f} | {m['test_f1']:<10.4f} | {m['test_accuracy']:<10.4f}")
    print(f"STD    | ±{s['val_iou']:<9.4f} | ±{s['test_iou']:<9.4f} | ±{s['bolivia_iou']:<11.4f} | ±{s['test_f1']:<9.4f} | ±{s['test_accuracy']:<9.4f}")
    print("="*80 + "\n")
    
    return aggregate

def save_benchmark_results(new_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    out_json = os.path.join(save_dir, "benchmark_results.json")
    
    existing = {}
    if os.path.exists(out_json):
        try:
            with open(out_json, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing = loaded
                elif isinstance(loaded, list):
                    for item in loaded:
                        if 'model' in item:
                            existing[item['model']] = item
        except Exception:
            existing = {}
            
    for item in new_results:
        existing[item['model']] = item
        
    with open(out_json, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Benchmark results successfully updated at: {out_json}")

def main():
    parser = argparse.ArgumentParser(description="Sen1Floods11 Multi-Seed Remote Sensing Benchmark")
    parser.add_argument('--data_dir', type=str, default=get_default_data_dir(), help='Path to Sen1Floods11 dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument(
        '--model', 
        type=str, 
        default='shadowfree_net', 
        help="Model name ('all', 'shadowfree_net', 'procanet', 'unet', 'unetplusplus', etc.)"
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='List of random seeds (e.g. --seeds 1 2 3 4 5)')
    parser.add_argument('--seed', type=int, default=None, help='Single seed shorthand (e.g. --seed 1)')
    parser.add_argument('--modality', type=str, default='s2', choices=['s1', 's2', 's2_4ch', 's1s2'])
    parser.add_argument('--bands', type=int, default=4, choices=[4, 13], help='4 for RGB+NIR, 13 for full Sentinel-2')
    parser.add_argument('--crop_size', type=int, default=0, help='Patch crop size (0 for full 256x256)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Max gradient norm for stability (default: 1.0)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'cosine_restarts'])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    
    # Resolve seeds
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [1, 2, 3, 4, 5]  # Default to 5 seeds
        
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    print(f"Device: {device} | Seeds: {seeds} | Data: {args.data_dir} | Modality: {args.modality} ({args.bands} bands)")
    
    all_standard_models = [
        'shadowfree_net',
        'procanet',
        'unet',
        'unetplusplus',
        'deeplabv3plus',
        'pspnet',
        'hrnet',
        'segformer',
        'transunet',
        'swin_unet',
        'farseg',
        'attentive_unet',
        'manet',
        'pan'
    ]
    
    if args.model.lower() == 'all':
        models_to_run = all_standard_models
    elif ',' in args.model:
        models_to_run = [m.strip().lower() for m in args.model.split(',') if m.strip()]
    else:
        models_to_run = [args.model.lower()]
        
    multiseed_results = []
    
    for m in models_to_run:
        res = train_model_multiseed(m, seeds, args, device)
        multiseed_results.append(res)
        # Save after every model so progress is persisted
        save_benchmark_results(multiseed_results, args.save_dir)
        
    # Grand Final Leaderboard Table across all models
    print("\n" + "="*115)
    print(f"{'GRAND MULTI-SEED BENCHMARK LEADERBOARD (SEN1FLOODS11)':^115}")
    print(f"{'SEEDS EVALUATED: ' + str(seeds):^115}")
    print("="*115)
    print(f"{'Model':<18} | {'Params':<8} | {'Test IoU (Mean±Std)':<22} | {'Bolivia IoU (Mean±Std)':<24} | {'Test F1 (Mean±Std)':<20} | {'Val IoU (Mean±Std)':<20}")
    print("-" * 115)
    
    sorted_results = sorted(multiseed_results, key=lambda x: x['mean']['test_iou'], reverse=True)
    for res in sorted_results:
        m_name = res['model'].upper()
        p = f"{res['params_M']:.2f}M"
        m = res['mean']
        s = res['std']
        test_str = f"{m['test_iou']:.4f} ± {s['test_iou']:.4f}"
        bolivia_str = f"{m['bolivia_iou']:.4f} ± {s['bolivia_iou']:.4f}"
        f1_str = f"{m['test_f1']:.4f} ± {s['test_f1']:.4f}"
        val_str = f"{m['val_iou']:.4f} ± {s['val_iou']:.4f}"
        print(f"{m_name:<18} | {p:<8} | {test_str:<22} | {bolivia_str:<24} | {f1_str:<20} | {val_str:<20}")
    print("="*115 + "\n")

if __name__ == '__main__':
    main()
