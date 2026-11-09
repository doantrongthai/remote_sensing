import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff

class CustomFloodDataset(Dataset):
    def __init__(self, config, split='train', transform=None):
        self.config = config
        self.split = split
        self.transform = transform
        
        self.data_dir = config.get('data_dir', '')
        self.image_ext = config.get('image_ext', '.tif')
        self.mask_ext = config.get('mask_ext', '.tif')
        self.in_channels = config.get('in_channels', 3)
        self.ignore_index = config.get('ignore_index', 255)
        self.normalize_mean = config.get('mean', None)
        self.normalize_std = config.get('std', None)
        
        images_dir = os.path.join(self.data_dir, split, 'images')
        if not os.path.isdir(images_dir):
            images_dir = os.path.join(self.data_dir, 'images')
            
        masks_dir = os.path.join(self.data_dir, split, 'masks')
        if not os.path.isdir(masks_dir):
            masks_dir = os.path.join(self.data_dir, 'masks')
            
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, f'*{self.image_ext}')))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, f'*{self.mask_ext}')))

    def __len__(self):
        return len(self.image_paths)

    def _read_image(self, path):
        if path.endswith('.tif') or path.endswith('.tiff'):
            arr = tiff.imread(path).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3 and arr.shape[-1] <= 16:
                arr = np.transpose(arr, (2, 0, 1))
        else:
            pil_img = Image.open(path).convert('RGB')
            arr = np.array(pil_img, dtype=np.float32)
            arr = np.transpose(arr, (2, 0, 1)) / 255.0
            
        return arr

    def _read_mask(self, path):
        if path.endswith('.tif') or path.endswith('.tiff'):
            arr = tiff.imread(path).astype(np.int64)
            if arr.ndim == 3:
                arr = arr[0]
        else:
            arr = np.array(Image.open(path), dtype=np.int64)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
        return arr

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx] if idx < len(self.mask_paths) else None
        
        img = self._read_image(img_path)
        img_tensor = torch.from_numpy(img).float()
        
        if self.normalize_mean is not None and self.normalize_std is not None:
            mean = torch.tensor(self.normalize_mean).view(-1, 1, 1)
            std = torch.tensor(self.normalize_std).view(-1, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
        if mask_path is not None:
            mask = self._read_mask(mask_path)
            mask_tensor = torch.from_numpy(mask).long()
        else:
            mask_tensor = torch.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.long)
            
        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)
            
        return img_tensor, mask_tensor
