import numpy as np
import torch
from src.data_processing import EEGDataset
from src.model import MultiTaskResNet

try:
    print("Loading one cache file...")
    data = np.load("cache/clean_serie1_X.npy")
    print(f"Shape: {data.shape}")
    
    print("Creating Dataset...")
    # Dummy labels
    labels = (np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data)))
    ds = EEGDataset(data, labels)
    item = ds[0]
    print("Dataset item shape:", item[0].shape)
    
    print("Instantiating Model...")
    model = MultiTaskResNet()
    print("Model created.")
    
    print("Forward pass...")
    out = model(item[0].unsqueeze(0))
    print("Output shapes:", [o.shape for o in out])
    
    print("VERIFICATION SUCCESSFUL")
except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
