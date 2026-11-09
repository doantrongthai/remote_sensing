import torch
import numpy as np

class FloodMetricEvaluator:
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.chip_water_ious = []

    def update(self, preds, targets):
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
            
        if preds.ndim == 4:
            preds = np.argmax(preds, axis=1)
        elif preds.ndim == 3 and targets.ndim == 2:
            preds = np.argmax(preds, axis=0)

        preds = np.atleast_3d(preds)
        targets = np.atleast_3d(targets)
        
        batch_size = preds.shape[0] if preds.shape[0] == targets.shape[0] else 1
        
        for b in range(batch_size):
            p = preds[b].flatten()
            t = targets[b].flatten()
            
            valid = (t != self.ignore_index)
            p_val = p[valid]
            t_val = t[valid]
            
            c_tp = np.sum((p_val == 1) & (t_val == 1))
            c_fp = np.sum((p_val == 1) & (t_val == 0))
            c_fn = np.sum((p_val == 0) & (t_val == 1))
            c_tn = np.sum((p_val == 0) & (t_val == 0))
            
            self.tp += c_tp
            self.fp += c_fp
            self.fn += c_fn
            self.tn += c_tn
            
            denom = c_tp + c_fp + c_fn
            if denom > 0:
                self.chip_water_ious.append(c_tp / denom)

    def compute(self):
        tp, fp, fn, tn = float(self.tp), float(self.fp), float(self.fn), float(self.tn)
        
        water_denom = tp + fp + fn
        water_iou = (tp / water_denom) if water_denom > 0 else 0.0
        
        bg_denom = tn + fp + fn
        bg_iou = (tn / bg_denom) if bg_denom > 0 else 0.0
        
        mean_iou = (water_iou + bg_iou) / 2.0
        
        prec_denom = tp + fp
        precision = (tp / prec_denom) if prec_denom > 0 else 0.0
        
        rec_denom = tp + fn
        recall = (tp / rec_denom) if rec_denom > 0 else 0.0
        
        f1_denom = precision + recall
        f1_score = (2.0 * precision * recall / f1_denom) if f1_denom > 0 else 0.0
        
        mean_chip_water_iou = float(np.mean(self.chip_water_ious)) if len(self.chip_water_ious) > 0 else 0.0
        
        return {
            'water_iou': water_iou,
            'bg_iou': bg_iou,
            'm_iou': mean_iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_chip_water_iou': mean_chip_water_iou,
            'total_valid_pixels': int(tp + fp + fn + tn),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
