import os
import sys
import subprocess
import threading
import argparse
import time

def stream_pipe(pipe, prefix):
    for line in iter(pipe.readline, ''):
        sys.stdout.write(f"{prefix} {line}")
        sys.stdout.flush()

def run_command_stream(cmd, prefix):
    proc = subprocess.Popen(
        cmd, shell=True, 
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, bufsize=1
    )
    t = threading.Thread(target=stream_pipe, args=(proc.stdout, prefix))
    t.start()
    t.join()
    proc.wait()
    return proc.returncode

def run_pair_concurrent(cmd0, cmd1, name0, name1):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("=" * 80)
    print(f"DUAL-GPU DISPATCH -> GPU 0: {name0.upper()} | GPU 1: {name1.upper()}")
    print(f"Working directory: {repo_root}")
    print("=" * 80)
    
    p0 = subprocess.Popen(
        cmd0, shell=True, cwd=repo_root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    p1 = subprocess.Popen(
        cmd1, shell=True, cwd=repo_root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    
    t0 = threading.Thread(target=stream_pipe, args=(p0.stdout, f"[GPU 0 | {name0:<14}]"))
    t1 = threading.Thread(target=stream_pipe, args=(p1.stdout, f"[GPU 1 | {name1:<14}]"))
    
    t0.start()
    t1.start()
    
    t0.join()
    t1.join()
    
    rc0 = p0.wait()
    rc1 = p1.wait()
    
    print("=" * 80)
    print(f"COMPLETED DUAL-GPU BATCH -> {name0.upper()}: code {rc0} | {name1.upper()}: code {rc1}")
    print("=" * 80 + "\n")
    return rc0, rc1

def main():
    parser = argparse.ArgumentParser(description="Dual-GPU High-Throughput Remote Sensing Runner")
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/doantrongthai/sen1floods11')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--model0', type=str, default='', help='Model for GPU 0')
    parser.add_argument('--model1', type=str, default='', help='Model for GPU 1')
    parser.add_argument('--all', action='store_true', help='Split all 14 models evenly across GPU 0 (7 models) and GPU 1 (7 models)')
    parser.add_argument('--split_seeds', type=str, default='', help='Model name to split seeds across GPU 0 (seeds 1 2 3) and GPU 1 (seeds 4 5)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--modality', type=str, default='s2')
    parser.add_argument('--bands', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    seeds_str = " ".join(str(s) for s in args.seeds)
    python_exe = sys.executable

    # Mode 1: Split Seeds for 1 model across GPU 0 and GPU 1
    if args.split_seeds:
        m = args.split_seeds
        half = len(args.seeds) // 2 + (len(args.seeds) % 2)
        seeds_g0 = " ".join(str(s) for s in args.seeds[:half])
        seeds_g1 = " ".join(str(s) for s in args.seeds[half:])
        
        cmd0 = (
            f"CUDA_VISIBLE_DEVICES=0 {python_exe} scripts/train.py "
            f"--data_dir {args.data_dir} --save_dir {args.save_dir} --model {m} "
            f"--seeds {seeds_g0} --modality {args.modality} --bands {args.bands} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"
        )
        cmd1 = (
            f"CUDA_VISIBLE_DEVICES=1 {python_exe} scripts/train.py "
            f"--data_dir {args.data_dir} --save_dir {args.save_dir} --model {m} "
            f"--seeds {seeds_g1} --modality {args.modality} --bands {args.bands} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"
        )
        run_pair_concurrent(cmd0, cmd1, f"{m}_seeds{seeds_g0.replace(' ', '')}", f"{m}_seeds{seeds_g1.replace(' ', '')}")
        return

    # Mode 2: Run a pair of models (model0 on GPU 0, model1 on GPU 1)
    if args.model0 and args.model1:
        cmd0 = (
            f"CUDA_VISIBLE_DEVICES=0 {python_exe} scripts/train.py "
            f"--data_dir {args.data_dir} --save_dir {args.save_dir} --model {args.model0} "
            f"--seeds {seeds_str} --modality {args.modality} --bands {args.bands} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"
        )
        cmd1 = (
            f"CUDA_VISIBLE_DEVICES=1 {python_exe} scripts/train.py "
            f"--data_dir {args.data_dir} --save_dir {args.save_dir} --model {args.model1} "
            f"--seeds {seeds_str} --modality {args.modality} --bands {args.bands} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"
        )
        run_pair_concurrent(cmd0, cmd1, args.model0, args.model1)
        return

    # Mode 3: Run all 14 models split 7 on GPU 0 and 7 on GPU 1 concurrently
    if args.all:
        group_0 = [
            'shadowfree_net_v2',
            'unet',
            'deeplabv3plus',
            'hrnet',
            'transunet',
            'farseg',
            'manet'
        ]
        group_1 = [
            'shadowfree_net',
            'procanet',
            'unetplusplus',
            'pspnet',
            'segformer',
            'swin_unet',
            'attentive_unet',
            'pan'
        ]
        
        models_str0 = ",".join(group_0)
        models_str1 = ",".join(group_1)
        
        cmd0 = (
            f"CUDA_VISIBLE_DEVICES=0 {python_exe} scripts/train.py "
            f"--data_dir {args.data_dir} --save_dir {args.save_dir} --model {models_str0} "
            f"--seeds {seeds_str} --modality {args.modality} --bands {args.bands} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"
        )
        cmd1 = (
            f"CUDA_VISIBLE_DEVICES=1 {python_exe} scripts/train.py "
            f"--data_dir {args.data_dir} --save_dir {args.save_dir} --model {models_str1} "
            f"--seeds {seeds_str} --modality {args.modality} --bands {args.bands} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr}"
        )
        
        print("="*80)
        print("LAUNCHING ALL 14 MODELS DUAL-GPU PARALLEL SWEEP (7 MODELS PER GPU)")
        print(f"GPU 0 (7 Models): {group_0}")
        print(f"GPU 1 (7 Models): {group_1}")
        print("="*80)
        
        run_pair_concurrent(cmd0, cmd1, "GPU0_Stream", "GPU1_Stream")
        return

    print("Please specify either --model0 and --model1, or --split_seeds, or --all.")

if __name__ == '__main__':
    main()
