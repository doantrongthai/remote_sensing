import sys
import os
import time
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormer
from models.prithvi_wrapper import PrithviAdapter
from models.dofa_wrapper import DOFAAdapter
from models.manet import MANet
from models.attentive_unet import AttentiveUNet
from models.pan import PAN
from models.farseg import FarSeg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model, input_tensor, warmup=2, runs=3):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            
    mean_latency = sum(times) / len(times)
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    return mean_latency, fps

def run_benchmarks(out_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running latency and parameter benchmark on: {device}")
    
    models = {
        'UNet (S1, 2-band)': (UNet(in_channels=2, num_classes=2), 2),
        'UNet (S1+S2, 15-band)': (UNet(in_channels=15, num_classes=2), 15),
        'DeepLabV3+ (S1, 2-band)': (DeepLabV3Plus(in_channels=2, num_classes=2), 2),
        'SegFormer (S1, 2-band)': (SegFormer(in_channels=2, num_classes=2), 2),
        'AttentiveUNet (S1, 2-band)': (AttentiveUNet(in_channels=2, num_classes=2), 2),
        'MANet (S1, 2-band)': (MANet(in_channels=2, num_classes=2), 2),
        'PAN (S1, 2-band)': (PAN(in_channels=2, num_classes=2), 2),
        'FarSeg (S1, 2-band)': (FarSeg(in_channels=2, num_classes=2), 2),
        'PrithviAdapter (S2, 13-band)': (PrithviAdapter(in_channels=13, num_classes=2, depth=2), 13),
        'DOFAAdapter (S1, 2-band)': (DOFAAdapter(in_channels=2, num_classes=2, depth=2), 2)
    }
    
    results = []
    header = f"{'Model':<30} | {'Input Size':<12} | {'Params (M)':<10} | {'Latency (ms)':<12} | {'FPS':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for name, (model, in_ch) in models.items():
        model.to(device)
        params_m = count_parameters(model) / 1e6
        
        for sz in [256, 512]:
            inp = torch.randn(1, in_ch, sz, sz, device=device)
            lat, fps = benchmark_model(model, inp, warmup=2, runs=3)
            row_str = f"{name:<30} | {f'{sz}x{sz}':<12} | {params_m:<10.2f} | {lat:<12.2f} | {fps:<8.2f}"
            print(row_str)
            results.append({
                'model': name,
                'input_size': f"{sz}x{sz}",
                'params_millions': round(params_m, 2),
                'latency_ms': round(lat, 2),
                'fps': round(fps, 2),
                'device': str(device)
            })
            
    print("-" * len(header))
    if out_file:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    return results

if __name__ == '__main__':
    out_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'latency_benchmark.json')
    run_benchmarks(out_path)
