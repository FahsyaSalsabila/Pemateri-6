"""Dataset utilities for the deforestation-alert case study."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = [
    "smallholder_agriculture",
    "large_scale_agriculture",
    "road",
    "selective_logging",
    "mining",
]

class PatchDataset(Dataset):
    """Loads an NPZ with X=(N,H,W,C), y=(N,), and converts to NCHW tensors."""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.int64)
        if self.X.ndim != 4:
            raise ValueError("X harus berbentuk (N, tinggi, lebar, kanal).")
        if len(self.X) != len(self.y):
            raise ValueError("Jumlah patch dan label tidak sama.")
        # Standardize per channel. For real data, fit statistics on train set only.
        mean = self.X.mean(axis=(0, 1, 2), keepdims=True)
        std = self.X.std(axis=(0, 1, 2), keepdims=True) + 1e-6
        self.X = (self.X - mean) / std
        self.X = np.transpose(self.X, (0, 3, 1, 2)).copy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)
