import sys
import os
import torch
import numpy as np

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
from models.manet import MANet
from models.attentive_unet import AttentiveUNet
from models.pan import PAN
from models.farseg import FarSeg

def test_split_counts(data_path):
    res = verify_dataset(data_path)
    for k, v in res.items():
        if v['status'] != 'MATCH':
            raise AssertionError(f"Dataset verification failed for {k}: {v}")
    return True

def test_quadrant_stitching():
    dummy = torch.randn(2, 512, 512)
    quadrants = generate_quadrants(dummy)
    assert len(quadrants) == 4
    for q in quadrants:
        assert q.shape == (2, 256, 256)
    reconstructed = stitch_quadrants(quadrants)
    diff = torch.max(torch.abs(dummy - reconstructed)).item()
    assert diff < 1e-6
    return True

def test_sliding_window_stitching():
    dummy = torch.ones(2, 512, 512)
    crops, coords = generate_sliding_windows(dummy, window_size=256, stride=128)
    assert len(crops) == 9
    for c in crops:
        assert c.shape == (2, 256, 256)
    stitched = stitch_sliding_windows(crops, coords, output_shape=(2, 512, 512))
    assert stitched.shape == (2, 512, 512)
    assert torch.allclose(stitched, dummy, atol=1e-5)
    return True

def test_evaluator_math():
    evaluator = FloodMetricEvaluator(ignore_index=255)
    
    t = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 255]
    ], dtype=np.int64)
    
    p = np.array([
        [1, 0, 1, 0],
        [1, 0, 0, 1]
    ], dtype=np.int64)
    
    evaluator.update(p, t)
    metrics = evaluator.compute()
    
    assert metrics['tp'] == 2
    assert metrics['fp'] == 1
    assert metrics['fn'] == 1
    assert metrics['tn'] == 3
    assert abs(metrics['water_iou'] - 0.5) < 1e-5
    assert abs(metrics['precision'] - (2.0 / 3.0)) < 1e-5
    assert abs(metrics['recall'] - (2.0 / 3.0)) < 1e-5
    return True

def test_model_forward():
    x_s1 = torch.randn(1, 2, 256, 256)
    x_s2 = torch.randn(1, 13, 256, 256)
    x_s1s2 = torch.randn(1, 15, 256, 256)
    
    m_unet_s1 = UNet(in_channels=2, num_classes=2)
    out_unet_s1 = m_unet_s1(x_s1)
    assert out_unet_s1.shape == (1, 2, 256, 256)
    
    m_unet_s1s2 = UNet(in_channels=15, num_classes=2)
    out_unet_s1s2 = m_unet_s1s2(x_s1s2)
    assert out_unet_s1s2.shape == (1, 2, 256, 256)
    
    m_deeplab = DeepLabV3Plus(in_channels=2, num_classes=2)
    out_deeplab = m_deeplab(x_s1)
    assert out_deeplab.shape == (1, 2, 256, 256)
    
    m_segformer = SegFormer(in_channels=2, num_classes=2)
    out_segformer = m_segformer(x_s1)
    assert out_segformer.shape == (1, 2, 256, 256)
    
    m_prithvi = PrithviAdapter(in_channels=13, num_classes=2, depth=2)
    out_prithvi = m_prithvi(x_s2)
    assert out_prithvi.shape == (1, 2, 256, 256)
    
    m_dofa = DOFAAdapter(in_channels=2, num_classes=2, depth=2)
    out_dofa = m_dofa(x_s1)
    assert out_dofa.shape == (1, 2, 256, 256)
    
    m_attentive = AttentiveUNet(in_channels=2, num_classes=2)
    out_attentive = m_attentive(x_s1)
    assert out_attentive.shape == (1, 2, 256, 256)
    
    m_manet = MANet(in_channels=2, num_classes=2)
    out_manet = m_manet(x_s1)
    assert out_manet.shape == (1, 2, 256, 256)
    
    m_pan = PAN(in_channels=2, num_classes=2)
    out_pan = m_pan(x_s1)
    assert out_pan.shape == (1, 2, 256, 256)
    
    m_farseg = FarSeg(in_channels=2, num_classes=2)
    out_farseg = m_farseg(x_s1)
    assert out_farseg.shape == (1, 2, 256, 256)
    
    return True

def run_all():
    default_data_dir = r'D:\Downloads\remote_sensing\sen1floods11_hand_labeled'
    
    print("1. Testing official split counts and file integrity...")
    test_split_counts(default_data_dir)
    print("   -> PASSED")
    
    print("2. Testing quadrant decomposition and exact stitching...")
    test_quadrant_stitching()
    print("   -> PASSED")
    
    print("3. Testing sliding window overlap and weighted stitching...")
    test_sliding_window_stitching()
    print("   -> PASSED")
    
    print("4. Testing FloodMetricEvaluator math and ignore_index handling...")
    test_evaluator_math()
    print("   -> PASSED")
    
    print("5. Testing forward passes on all benchmark model architectures...")
    test_model_forward()
    print("   -> PASSED")
    
    print("\nALL SANITY CHECKS PASSED SUCCESSFULLY.")

if __name__ == '__main__':
    run_all()
