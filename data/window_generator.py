import torch
import numpy as np

def generate_quadrants(image, label=None):
    c, h, w = image.shape
    mid_h, mid_w = h // 2, w // 2
    
    img_q = [
        image[:, 0:mid_h, 0:mid_w],
        image[:, 0:mid_h, mid_w:w],
        image[:, mid_h:h, 0:mid_w],
        image[:, mid_h:h, mid_w:w]
    ]
    
    if label is None:
        return img_q
        
    lbl_q = [
        label[0:mid_h, 0:mid_w],
        label[0:mid_h, mid_w:w],
        label[mid_h:h, 0:mid_w],
        label[mid_h:h, mid_w:w]
    ]
    return img_q, lbl_q

def stitch_quadrants(quadrant_preds):
    top = torch.cat([quadrant_preds[0], quadrant_preds[1]], dim=-1)
    bottom = torch.cat([quadrant_preds[2], quadrant_preds[3]], dim=-1)
    return torch.cat([top, bottom], dim=-2)

def generate_sliding_windows(image, label=None, window_size=256, stride=128):
    c, h, w = image.shape
    img_crops = []
    lbl_crops = []
    coords = []
    
    y_starts = list(range(0, h - window_size + 1, stride))
    if y_starts[-1] + window_size < h:
        y_starts.append(h - window_size)
        
    x_starts = list(range(0, w - window_size + 1, stride))
    if x_starts[-1] + window_size < w:
        x_starts.append(w - window_size)
        
    for y in y_starts:
        for x in x_starts:
            coords.append((y, y + window_size, x, x + window_size))
            img_crops.append(image[:, y:y+window_size, x:x+window_size])
            if label is not None:
                lbl_crops.append(label[y:y+window_size, x:x+window_size])
                
    if label is None:
        return img_crops, coords
    return img_crops, lbl_crops, coords

def stitch_sliding_windows(preds, coords, output_shape=(2, 512, 512), device='cpu'):
    num_classes, h, w = output_shape
    canvas = torch.zeros((num_classes, h, w), device=device)
    counts = torch.zeros((1, h, w), device=device)
    
    for pred, (y1, y2, x1, x2) in zip(preds, coords):
        canvas[:, y1:y2, x1:x2] += pred.to(device)
        counts[:, y1:y2, x1:x2] += 1.0
        
    counts = torch.clamp(counts, min=1.0)
    return canvas / counts
