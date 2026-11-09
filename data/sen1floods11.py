import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff

class Sen1Floods11Dataset(Dataset):
    def __init__(self, root_dir, split='train', modality='s1', crop_size=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.modality = modality.lower()
        self.crop_size = crop_size
        self.transform = transform
        
        split_map = {
            'train': 'flood_train_data.csv',
            'valid': 'flood_valid_data.csv',
            'test': 'flood_test_data.csv',
            'bolivia': 'flood_bolivia_data.csv'
        }
        
        splits_dir = os.path.join(root_dir, 'splits')
        if not os.path.isdir(splits_dir):
            splits_dir = root_dir
            
        csv_file = os.path.join(splits_dir, split_map[split])
        self.samples = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    self.samples.append((row[0].strip(), row[1].strip()))

    def __len__(self):
        return len(self.samples)

    def _load_s1(self, filename):
        path = os.path.join(self.root_dir, 'S1Hand', filename)
        arr = tiff.imread(path).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = np.clip(arr, -50.0, 1.0)
        arr = (arr + 50.0) / 51.0
        return arr

    def _load_s2(self, filename):
        s2_filename = filename.replace('_S1Hand.tif', '_S2Hand.tif')
        path = os.path.join(self.root_dir, 'S2Hand', s2_filename)
        arr = tiff.imread(path).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        arr = arr / 10000.0
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    def _load_label(self, filename):
        path = os.path.join(self.root_dir, 'LabelHand', filename)
        arr = tiff.imread(path).astype(np.int64)
        arr[arr == -1] = 255
        return arr

    def __getitem__(self, idx):
        s1_file, label_file = self.samples[idx]
        
        if self.modality == 's1':
            img = self._load_s1(s1_file)
        elif self.modality == 's2':
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
            ch, cw = self.crop_size
            if h >= ch and w >= cw:
                top = np.random.randint(0, h - ch + 1)
                left = np.random.randint(0, w - cw + 1)
                img = img[:, top:top+ch, left:left+cw]
                label = label[top:top+ch, left:left+cw]

        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).long()

        if self.transform is not None:
            img_tensor, label_tensor = self.transform(img_tensor, label_tensor)

        return img_tensor, label_tensor
