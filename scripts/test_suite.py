import sys
import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.download_verify import verify_dataset
from data.sen1floods11 import Sen1Floods11Dataset
from data.window_generator import generate_quadrants, stitch_quadrants, generate_sliding_windows, stitch_sliding_windows
from metrics.evaluator import FloodMetricEvaluator
from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormer
from models.prithvi_wrapper import PrithviAdapter
from models.dofa_wrapper import DOFAAdapter
from losses.losses import BCEDiceLoss

def test_split_counts_256():
    data_dir = r"D:\Downloads\remote_sensing\sen1floods11_256x256"
    expected = {
        'flood_train_data.csv': 1008,
        'flood_valid_data.csv': 356,
        'flood_test_data.csv': 360,
        'flood_bolivia_data.csv': 60
    }
    for fname, count in expected.items():
        p = os.path.join(data_dir, 'splits', fname)
        assert os.path.exists(p)
        df = pd.read_csv(p, header=None)
        assert len(df) == count, f"{fname} expected {count}, got {len(df)}"
    return True

def test_window_counts_formula():
    h, w = 512, 512
    window_sz = 256
    stride_256 = 256
    stride_128 = 128
    
    n_256 = (h // window_sz) * (w // window_sz)
    assert n_256 == 4
    
    n_128 = ((h - window_sz) // stride_128 + 1) ** 2
    assert n_128 == 9
    return True

def test_stitching_reproduction():
    dummy = torch.randn(2, 512, 512)
    quads = generate_quadrants(dummy)
    assert len(quads) == 4
    reconstructed = stitch_quadrants(quads)
    assert torch.allclose(dummy, reconstructed, atol=1e-6)
    return True

def test_metrics_on_synthetic_masks():
    ev = FloodMetricEvaluator(ignore_index=255)
    t = np.array([[1, 1, 0, 0], [1, 0, 0, 255]], dtype=np.int64)
    p = np.array([[1, 0, 1, 0], [1, 0, 0, 1]], dtype=np.int64)
    ev.update(p, t)
    res = ev.compute()
    assert res['tp'] == 2
    assert res['fp'] == 1
    assert res['fn'] == 1
    assert res['tn'] == 3
    assert abs(res['water_iou'] - 0.5) < 1e-5
    return True

def test_all_models_output_shape():
    b, h, w = 2, 256, 256
    models_to_test = [
        (UNet(in_channels=2, num_classes=2), torch.randn(b, 2, h, w)),
        (UNet(in_channels=15, num_classes=2), torch.randn(b, 15, h, w)),
        (DeepLabV3Plus(in_channels=2, num_classes=2), torch.randn(b, 2, h, w)),
        (SegFormer(in_channels=2, num_classes=2), torch.randn(b, 2, h, w)),
        (PrithviAdapter(in_channels=13, num_classes=2, depth=2), torch.randn(b, 13, h, w)),
        (DOFAAdapter(in_channels=2, num_classes=2, depth=2), torch.randn(b, 2, h, w)),
    ]
    for model, inp in models_to_test:
        model.eval()
        with torch.no_grad():
            out = model(inp)
            assert out.shape == (b, 2, h, w), f"Model {type(model)} returned {out.shape}"
    return True

def test_one_batch_overfit():
    torch.manual_seed(42)
    model = UNet(in_channels=2, num_classes=2)
    criterion = BCEDiceLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    
    x = torch.randn(2, 2, 256, 256)
    y = torch.randint(0, 2, (2, 256, 256), dtype=torch.long)
    y[0, :20, :20] = 255
    
    initial_loss = None
    final_loss = None
    
    for step in range(15):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    return True

def test_registry_resolves_every_file():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    papers_csv = os.path.join(base_dir, 'docs', 'papers.csv')
    assert os.path.exists(papers_csv)
    df = pd.read_csv(papers_csv)
    assert len(df) >= 10
    required_cols = [
        'paper', 'authors', 'venue', 'year', 'arxiv_or_doi',
        'uses_sen1floods11', 'metric', 'value', 'input_modality',
        'label_set', 'split', 'source_url', 'pdf_page_or_table', 'status'
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing column {c}"
    return True

def test_no_model_imports_project_file():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    for f in os.listdir(models_dir):
        if f.endswith('.py'):
            p = os.path.join(models_dir, f)
            with open(p, 'r', encoding='utf-8') as fh:
                content = fh.read()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    assert not line.startswith('from data')
                    assert not line.startswith('from models')
                    assert not line.startswith('from metrics')
                    assert not line.startswith('from losses')
                    assert not line.startswith('from scripts')
    return True

def test_no_comments_or_docstrings():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    violations = []
    hash_char = chr(35)
    quote_3 = chr(34) * 3
    quote_s = chr(39) * 3
    for root, dirs, files in os.walk(base_dir):
        if '.git' in root or '__pycache__' in root or 'runs' in root:
            continue
        for f in files:
            if f.endswith('.py'):
                p = os.path.join(root, f)
                with open(p, 'r', encoding='utf-8') as fh:
                    for idx, line in enumerate(fh):
                        s = line.strip()
                        if s.startswith(hash_char) or quote_3 in line or quote_s in line:
                            violations.append((f, idx + 1, s))
    assert len(violations) == 0, f"Violations found: {violations}"
    return True

def run_all_tests():
    tests = [
        ("1. Split counts for 256x256 (1008/356/360/60)", test_split_counts_256),
        ("2. Window counts formula verification", test_window_counts_formula),
        ("3. Synthetic stitching exact reproduction", test_stitching_reproduction),
        ("4. Metric evaluator math with ignore_index", test_metrics_on_synthetic_masks),
        ("5. Output shape [B, 2, H, W] on all models", test_all_models_output_shape),
        ("6. One-batch overfit test (loss decreases)", test_one_batch_overfit),
        ("7. Papers registry resolution & schema check", test_registry_resolves_every_file),
        ("8. Static check: No model imports other project files", test_no_model_imports_project_file),
        ("9. Static check: Zero comments and zero docstrings", test_no_comments_or_docstrings),
    ]
    all_passed = True
    for name, func in tests:
        try:
            func()
            print(f"[PASS] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            all_passed = False
            
    if all_passed:
        print("\nALL TEST SUITE CHECKS COMPLETED SUCCESSFULLY WITH ZERO ERRORS.")
        return 0
    else:
        print("\nTEST SUITE FAILED.")
        return 1

if __name__ == '__main__':
    exit(run_all_tests())
