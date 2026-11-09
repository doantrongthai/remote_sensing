import sys
import os
import time
import argparse
import subprocess
import json
import hashlib
import numpy as np
import scipy.stats as stats
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.efficiency import benchmark_efficiency
from scripts.train import train_single_model

def get_git_hash():
    try:
        h = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        return h
    except Exception:
        return "UNKNOWN"

def get_env_versions():
    import torchvision
    return {
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'numpy': np.__version__,
        'scipy': stats.__version__
    }

def run_multi_seed_benchmark(args):
    seeds = args.seeds
    models = args.models
    results_dir = os.path.abspath(args.results_dir)
    runs_dir = os.path.abspath(args.runs_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    git_hash = get_git_hash()
    env_versions = get_env_versions()
    
    all_runs_data = []
    model_chip_ious = {m: [] for m in models}
    
    for model_name in models:
        model_runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
            run_id = f"{model_name}_{args.modality}_seed{seed}"
            run_folder = os.path.join(runs_dir, run_id)
            os.makedirs(run_folder, exist_ok=True)
            
            meta = {
                'run_id': run_id,
                'model': model_name,
                'modality': args.modality,
                'seed': seed,
                'git_hash': git_hash,
                'env_versions': env_versions,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr
            }
            with open(os.path.join(run_folder, 'run_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
                
            args.save_dir = run_folder
            run_summary = train_single_model(model_name, args, device)
            run_summary['seed'] = seed
            run_summary['run_id'] = run_id
            
            all_runs_data.append(run_summary)
            model_runs.append(run_summary)
            
        test_ious = [r['test_water_iou'] for r in model_runs]
        mean_iou = np.mean(test_ious)
        std_iou = np.std(test_ious)
        ci_95 = 1.96 * (std_iou / np.sqrt(len(test_ious))) if len(test_ious) > 1 else 0.0
        
        print(f"\nModel {model_name.upper()} over {len(seeds)} seeds: Mean Water IoU = {mean_iou:.4f} +/- {std_iou:.4f} (95% CI: [{mean_iou-ci_95:.4f}, {mean_iou+ci_95:.4f}])")

    runs_csv_path = os.path.join(results_dir, "runs.csv")
    df_new = pd.DataFrame(all_runs_data)
    if os.path.exists(runs_csv_path):
        df_old = pd.read_csv(runs_csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(runs_csv_path, index=False)
    else:
        df_new.to_csv(runs_csv_path, index=False)
        
    print(f"\nAll benchmark runs completed. Results recorded to {runs_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'D:\Downloads\remote_sensing\sen1floods11_256x256')
    parser.add_argument('--models', nargs='+', default=['unet'])
    parser.add_argument('--modality', type=str, default='s1', choices=['s1', 's2', 's1s2'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--runs_dir', type=str, default='./runs')
    parser.add_argument('--time_budget_minutes', type=int, default=0)
    args = parser.parse_args()
    
    run_multi_seed_benchmark(args)
