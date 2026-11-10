import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff

def _find_dir(base_dir, target_name):
    """
    Search for target_name within base_dir directly, or case-insensitive, or 1 level deeper.
    """
    if not base_dir or not os.path.exists(base_dir):
        return os.path.join(base_dir or '', target_name)
    for name in [target_name, target_name.lower(), target_name.capitalize()]:
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            return p
    # Check 1 level down
    try:
        for sub in os.listdir(base_dir):
            sub_p = os.path.join(base_dir, sub)
            if os.path.isdir(sub_p):
                for name in [target_name, target_name.lower(), target_name.capitalize()]:
                    p = os.path.join(sub_p, name)
                    if os.path.isdir(p):
                        return p
    except Exception:
        pass
    return os.path.join(base_dir, target_name)

class Sen1Floods11Dataset(Dataset):
    def __init__(self, root_dir=None, split='train', modality='s1', bands=None, crop_size=None, transform=None):
        if root_dir is None:
            # Fallback candidates for Kaggle and local environments
            candidates = [
                '/kaggle/input/datasets/doantrongthai/sen1floods11',
                '/kaggle/input/sen1floods11',
                '/kaggle/working/data',
                r'D:\Downloads\remote_sensing\sen1floods11_256x256',
                r'D:\Downloads\remote_sensing\sen1floods11_hand_labeled',
                './data'
            ]
            root_dir = next((c for c in candidates if os.path.exists(c)), candidates[0])

        self.root_dir = root_dir
        self.split = split
        self.modality = modality.lower()
        self.bands = bands
        self.crop_size = crop_size
        self.transform = transform
        
        # If bands is not explicitly passed, infer from modality
        if self.bands is None:
            if self.modality == 's1':
                self.bands = 2
            elif self.modality in ['s2', 's2_4ch']:
                self.bands = 4  # Default: 4 channels (Red, Green, Blue, NIR)
            else:
                self.bands = 4
        
        split_map = {
            'train': 'flood_train_data.csv',
            'valid': 'flood_valid_data.csv',
            'test': 'flood_test_data.csv',
            'bolivia': 'flood_bolivia_data.csv'
        }
        
        self.label_dir = _find_dir(self.root_dir, 'LabelHand')
        self.s1_dir = _find_dir(self.root_dir, 'S1Hand')
        self.s2_dir = _find_dir(self.root_dir, 'S2Hand')
        self.splits_dir = _find_dir(self.root_dir, 'splits')
        
        csv_file = os.path.join(self.splits_dir, split_map[split])
        if not os.path.exists(csv_file):
            alt_csv = os.path.join(self.root_dir, split_map[split])
            if os.path.exists(alt_csv):
                csv_file = alt_csv
            
        self.samples = []
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        self.samples.append((row[0].strip(), row[1].strip()))
        else:
            print(f"[Warning] CSV split file not found: {csv_file}")

    def __len__(self):
        return len(self.samples)

    def _load_s1(self, filename):
        fname = os.path.basename(filename)
        path = os.path.join(self.s1_dir, fname)
        arr = tiff.imread(path).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.clip(arr, -50.0, 1.0)
        arr = (arr + 50.0) / 51.0
        return arr

    def _load_s2(self, filename):
        fname = os.path.basename(filename).replace('_S1Hand.tif', '_S2Hand.tif')
        path = os.path.join(self.s2_dir, fname)
        arr = tiff.imread(path).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = arr / 10000.0
        arr = np.clip(arr, 0.0, 1.0)
        # Select 4 channels (Red, Green, Blue, NIR) as in Feliren et al. (IEEE GRSL 2025)
        if self.bands == 4 and arr.shape[0] >= 8:
            arr = arr[[3, 2, 1, 7], :, :]  # B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
        return arr

    def _load_label(self, filename):
        fname = os.path.basename(filename)
        path = os.path.join(self.label_dir, fname)
        arr = tiff.imread(path).astype(np.int64)
        arr[arr == -1] = 255
        return arr

    def __getitem__(self, idx):
        s1_file, label_file = self.samples[idx]
        
        if self.modality == 's1':
            img = self._load_s1(s1_file)
        elif self.modality in ['s2', 's2_4ch']:
            img = self._load_s2(s1_file)
        elif self.modality == 's1s2':
            s1 = self._load_s1(s1_file)
            s2 = self._load_s2(s1_file)
            img = np.concatenate([s1, s2], axis=0)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")
            
        label = self._load_label(label_file)

        if self.crop_size is not None:
            _, h, w = img.shape
            ch, cw = (self.crop_size, self.crop_size) if isinstance(self.crop_size, int) else self.crop_size
            if h >= ch and w >= cw:
                if self.split == 'train':
                    # Feliren et al. Section II.D: filter patches with >50% nodata (255)
                    best_top, best_left = 0, 0
                    for _ in range(10):
                        top = np.random.randint(0, h - ch + 1)
                        left = np.random.randint(0, w - cw + 1)
                        sub_lbl = label[top:top+ch, left:left+cw]
                        if np.mean(sub_lbl == 255) < 0.5:
                            best_top, best_left = top, left
                            break
                    img = img[:, best_top:best_top+ch, best_left:best_left+cw]
                    label = label[best_top:best_top+ch, best_left:best_left+cw]
                else:
                    # Center crop for evaluation consistency
                    top = (h - ch) // 2
                    left = (w - cw) // 2
                    img = img[:, top:top+ch, left:left+cw]
                    label = label[top:top+ch, left:left+cw]

        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).long()

        if self.transform is not None:
            img_tensor, label_tensor = self.transform(img_tensor, label_tensor)

        return img_tensor, label_tensor
