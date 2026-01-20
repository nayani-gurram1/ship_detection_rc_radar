import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json

class RadarShipDataset(Dataset):
    def __init__(self, cpi_dir, label_dir, transform=None):
        self.cpi_dir = cpi_dir
        self.label_dir = label_dir
        self.transform = transform

        # List all CPI and label files
        self.cpi_files = sorted([f for f in os.listdir(cpi_dir) if f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])

    def __len__(self):
        return len(self.cpi_files)

    def __getitem__(self, idx):
        # Load CPI
        cpi_path = os.path.join(self.cpi_dir, self.cpi_files[idx])
        cpi = np.load(cpi_path)
        cpi = np.abs(cpi)  # magnitude
        cpi = cpi.astype(np.float32)

        # Convert to 3D (C,H,W) as required by PyTorch (1 channel)
        cpi = np.expand_dims(cpi, axis=0)
        image = torch.from_numpy(cpi)

        # Load labels
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            labels = json.load(f)

        boxes = torch.tensor(labels['bboxes'], dtype=torch.float32)
        # Faster R-CNN expects labels starting from 1
        target = {
            "boxes": boxes,
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target
