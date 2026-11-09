import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

def plot_curves(history, save_dir):
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    epochs = [h['epoch'] for h in history]
    losses = [h['train_loss'] for h in history]
    val_ious = [h['val_water_iou'] for h in history]
    val_mious = [h['val_m_iou'] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(epochs, losses, color='#D55E00', lw=2, label='Train Loss')
    l2 = ax2.plot(epochs, val_ious, color='#0072B2', lw=2, label='Val Water IoU')
    l3 = ax2.plot(epochs, val_mious, color='#009E73', lw=2, ls='--', label='Val mIoU')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='#D55E00', fontsize=12)
    ax2.set_ylabel('IoU', color='#0072B2', fontsize=12)
    
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')
    plt.title('Training Loss & Validation Metrics')
    
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(save_dir, "figures", f"curves.{ext}"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm_dict, save_dir):
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    tp, fp = cm_dict['tp'], cm_dict['fp']
    fn, tn = cm_dict['fn'], cm_dict['tn']
    
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap='Blues')
    plt.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, f"{z:,}", ha='center', va='center', color='black' if z < matrix.max()/2 else 'white')
        
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Background', 'Water'])
    ax.set_yticklabels(['Background', 'Water'])
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(save_dir, "figures", f"confusion_matrix.{ext}"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_chip_iou(chip_ious, save_dir):
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(chip_ious, bins=20, color='#56B4E9', edgecolor='black', alpha=0.8)
    ax.axvline(np.mean(chip_ious), color='#D55E00', linestyle='dashed', linewidth=2, label=f"Mean: {np.mean(chip_ious):.4f}")
    ax.axvline(np.median(chip_ious), color='#009E73', linestyle='dotted', linewidth=2, label=f"Median: {np.median(chip_ious):.4f}")
    plt.xlabel('Water IoU per Chip')
    plt.ylabel('Number of Chips')
    plt.legend()
    plt.title('Per-Chip Water IoU Distribution')
    
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(save_dir, "figures", f"per_chip_iou.{ext}"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_map(image, target, pred, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    h, w = target.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    valid = (target != 255)
    tp = (pred == 1) & (target == 1) & valid
    fp = (pred == 1) & (target == 0) & valid
    fn = (pred == 0) & (target == 1) & valid
    tn = (pred == 0) & (target == 0) & valid
    ig = (target == 255)
    
    error_map[tp] = [0, 158, 115]
    error_map[fp] = [213, 94, 0]
    error_map[fn] = [0, 114, 178]
    error_map[tn] = [30, 30, 30]
    error_map[ig] = [128, 128, 128]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    if image.shape[0] >= 3:
        rgb = np.transpose(image[:3], (1, 2, 0))
    else:
        rgb = np.stack([image[0], image[1], (image[0]+image[1])/2], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-7)
    
    axes[0].imshow(rgb)
    axes[0].set_title("Input Composite")
    axes[0].axis('off')
    
    axes[1].imshow(target, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(pred, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    axes[3].imshow(error_map)
    axes[3].set_title("Error Map (Green=TP, Red=FP, Blue=FN)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_latex_table(results_list, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Benchmark Performance on Sen1Floods11 Dataset (256$\\times$256)}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & Modality & Val Water IoU & Test Water IoU & Bolivia Water IoU \\\\")
    lines.append("\\midrule")
    for r in results_list:
        lines.append(f"{r['model'].upper()} & {r['modality'].upper()} & {r.get('best_val_iou', 0.0):.4f} & {r.get('test_iou', 0.0):.4f} & {r.get('bolivia_iou', 0.0):.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(save_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
