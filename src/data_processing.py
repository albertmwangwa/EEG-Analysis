import numpy as np
from scipy.interpolate import griddata

# 10-20 electrode coordinates for interpolation
CHANNEL_POSITIONS = {
    "F3": (-0.5, 0.6),
    "Fz": (0.0, 0.65),
    "F4": (0.5, 0.6),
    "C3": (-0.5, 0.0),
    "C4": (0.5, 0.0),
    "Pz": (0.0, -0.4),
    "O1": (-0.4, -0.8),
    "O2": (0.4, -0.8),
}

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
ELECTRODES = ["F3", "Fz", "F4", "C3", "C4", "Pz", "O1", "O2"]

def get_grid_coordinates(resolution=64):
    """Generates the meshgrid coordinates."""
    return np.mgrid[-1:1:complex(0, resolution), -1:1:complex(0, resolution)]

def create_eeg_image(row, resolution=64):
    """
    Converts a single row of EEG band power data into a 5-channel image.
    
    Args:
        row: A dictionary-like object (e.g., pandas Series) containing keys like 'Delta_F3', 'Theta_Fz', etc.
        resolution: Output image resolution (default 64x64).
        
    Returns:
        np.ndarray: A (5, resolution, resolution) float32 array.
    """
    grid_x, grid_y = get_grid_coordinates(resolution)
    img = np.zeros((5, resolution, resolution))

    for bi, band in enumerate(BANDS):
        # Extract values for the current band across all electrodes
        try:
            values = [row[f"{band}_{ch}"] for ch in ELECTRODES]
        except KeyError as e:
            # Fallback if keys are missing or named differently (e.g. if row is just a dict)
            # This part assumes the input row strictly follows the naming convention
            raise KeyError(f"Missing key in input row: {e}")

        coords = np.array([CHANNEL_POSITIONS[ch] for ch in ELECTRODES])

        # Interpolate
        grid = griddata(coords, values, (grid_x, grid_y), method='cubic', fill_value=0)
        img[bi] = grid

    return img.astype(np.float32)

import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, X, labels=None, transform=None):
        self.X = X
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float)
        
        if self.transform:
            x = self.transform(x)

        if self.labels is None:
            return x

        shot, stress, task = self.labels

        return (
            x,
            torch.tensor(shot[idx], dtype=torch.long),
            torch.tensor(stress[idx], dtype=torch.float),
            torch.tensor(task[idx], dtype=torch.long)
        )
