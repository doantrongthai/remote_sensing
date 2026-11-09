import time
import torch
import torch.nn as nn

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def estimate_flops(model, input_shape=(1, 2, 256, 256)):
    total_flops = 0
    def conv_hook(self, input, output):
        nonlocal total_flops
        b, c_out, h_out, w_out = output.shape
        c_in = self.in_channels
        k_h, k_w = self.kernel_size
        flops_per_instance = 2 * c_in * k_h * k_w * c_out * h_out * w_out
        total_flops += flops_per_instance // self.groups
    def linear_hook(self, input, output):
        nonlocal total_flops
        total_flops += 2 * self.in_features * self.out_features
        
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
            
    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_shape, device=device)
    model.eval()
    with torch.no_grad():
        try:
            _ = model(dummy_input)
        except Exception:
            pass
            
    for h in hooks:
        h.remove()
        
    return total_flops

def measure_latency_and_memory(model, input_tensor, warmup=5, runs=20):
    device = input_tensor.device
    model.eval()
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
    mean_lat = sum(latencies) / len(latencies)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    
    if device.type == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem_mb = 0.0
        
    return {
        'latency_ms': round(mean_lat, 2),
        'fps': round(fps, 2),
        'peak_memory_mb': round(peak_mem_mb, 2)
    }

def benchmark_efficiency(model, in_channels=2, height=256, width=256, batch_size=1, device='cpu'):
    d = torch.device(device)
    model.to(d)
    
    total_p, train_p = count_parameters(model)
    inp = torch.randn(batch_size, in_channels, height, width, device=d)
    flops = estimate_flops(model, input_shape=(batch_size, in_channels, height, width))
    lat_mem = measure_latency_and_memory(model, inp, warmup=3, runs=10)
    
    return {
        'params_total': total_p,
        'params_millions': round(total_p / 1e6, 2),
        'flops_giga': round(flops / 1e9, 2),
        'latency_ms': lat_mem['latency_ms'],
        'fps': lat_mem['fps'],
        'peak_memory_mb': lat_mem['peak_memory_mb']
    }
