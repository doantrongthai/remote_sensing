# sen1floods11-bench: Reproducible Benchmark Suite for Sen1Floods11

Standardized, scientifically auditable benchmarking framework for satellite flood segmentation on Sen1Floods11 (Sentinel-1 SAR, Sentinel-2 Optical, and Multimodal Fusion).

## 1. Dataset Specification
- Total Hand-Labeled Chips: 446 chips (512x512 pixels, 10m/pixel spatial resolution)
- Official Splits:
  - `flood_train_data.csv`: 252 chips
  - `flood_valid_data.csv`: 89 chips
  - `flood_test_data.csv`: 90 chips
  - `flood_bolivia_data.csv`: 15 chips (Out-of-region holdout)
- Modalities:
  - Sentinel-1: VV, VH (dB backscatter)
  - Sentinel-2: 13 multi-spectral bands
  - Labels: 0 (Non-water), 1 (Water), -1 (No-data / remapped to ignore_index=255)

## 2. Directory Structure
```
sen1floods11-bench/
├── configs/               # Model configuration templates
├── data/
│   ├── download_verify.py # Integrity and split verification script
│   ├── sen1floods11.py    # PyTorch Dataset for Sen1Floods11
│   ├── window_generator.py# 256x256 quadrant & sliding window tiling/stitching
│   └── custom_dataset.py  # Config-driven template loader for user datasets
├── losses/
│   └── losses.py          # DiceLoss, BCEDiceLoss, FocalLoss with ignore_index=255
├── metrics/
│   └── evaluator.py       # Global & per-chip Water IoU, mIoU, F1 Evaluator
├── models/
│   ├── unet.py            # U-Net implementation
│   ├── deeplabv3plus.py   # DeepLabV3+ with ASPP
│   ├── segformer.py       # SegFormer hierarchical transformer
│   ├── attentive_unet.py  # Attentive U-Net for Flood Detection (Yadav et al., Frontiers 2022)
│   ├── manet.py           # Multi-Attention Network (Li et al., IEEE TGRS 2021)
│   ├── pan.py             # Pyramid Attention Network (Li et al., 2018)
│   ├── farseg.py          # Foreground-Aware Relation Network (Zheng et al., CVPR 2020)
│   ├── prithvi_wrapper.py # IBM-NASA Prithvi-100M foundation model adapter
│   └── dofa_wrapper.py    # DOFA dynamic cross-modality adapter
├── registry/
│   ├── papers.csv         # 15 audited primary-source publications
│   └── repos.csv          # Verified official GitHub repositories
├── reports/
│   └── latency_benchmark.json # Hardware latency and throughput measurements
└── scripts/
    ├── sanity_check.py    # Verification suite (splits, stitching, losses, models)
    ├── train.py           # Training entry point
    ├── eval.py            # Evaluation entry point (Test and Bolivia sets)
    └── benchmark_latency.py # Inference latency, FPS, and parameter profiler
```

## 3. Running Sanity Checks
Verify split counts, quadrant stitching, metric math, and model forward passes:
```bash
python scripts/sanity_check.py
```

## 4. Latency and Parameter Profiling
Profile inference time (ms) and throughput (FPS) across models:
```bash
python scripts/benchmark_latency.py
```

## 5. Evaluation Protocol
Evaluate on official test set (90 chips) with quadrant stitching:
```bash
python scripts/eval.py --data_dir /path/to/sen1floods11_hand_labeled --split test --modality s1 --model unet --use_quadrants
```
Evaluate on Bolivia holdout set (15 chips):
```bash
python scripts/eval.py --data_dir /path/to/sen1floods11_hand_labeled --split bolivia --modality s1 --model unet --use_quadrants
```

## 6. Scientific Integrity Notes
- Water IoU formula: TP / (TP + FP + FN), computed strictly over non-ignored pixels (ignore_index=255).
- All reported published metrics are strictly traced to primary paper PDFs (page and table).
- No synthetic or unverified reproduction metrics are reported without local GPU execution.
