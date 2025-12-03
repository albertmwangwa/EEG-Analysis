import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.model import MultiTaskResNet
from src.data_processing import EEGDataset

# Configuration
CACHE_DIR = "cache"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Safe Augmentations (No Left-Right Flips)
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    # Add noise or other safe augmentations if needed
])

def generate_dummy_labels(n):
    """Generates dummy labels for testing purposes."""
    shot = np.random.randint(0, 2, size=n)          # 0/1 good/bad
    stress = np.random.uniform(0, 1, size=n)        # regression
    task = np.random.randint(0, 3, size=n)          # 3 classes
    return shot, stress, task

def load_cached_data():
    """Loads all .npy files from the cache directory."""
    files = glob.glob(os.path.join(CACHE_DIR, "*_X.npy"))
    if not files:
        raise FileNotFoundError(f"No cache files found in {CACHE_DIR}. Please generate them first.")
    
    data = {}
    labels = {}
    for f in files:
        name = os.path.basename(f).replace("_X.npy", "")
        X = np.load(f)
        data[name] = X
        # In a real scenario, you'd load real labels. Here we generate dummy ones.
        labels[name] = generate_dummy_labels(len(X))
        print(f"Loaded {name}: {X.shape}")
    return data, labels

def get_loso_loaders(data, labels, val_name, batch_size=32):
    X_train = []
    shot_train = []
    stress_train = []
    task_train = []

    X_val = None
    shot_val, stress_val, task_val = None, None, None

    for name, X in data.items():
        s, st, ta = labels[name]
        if name != val_name:
            X_train.append(X)
            shot_train.append(s)
            stress_train.append(st)
            task_train.append(ta)
        else:
            X_val = X
            shot_val, stress_val, task_val = s, st, ta

    if X_val is None:
        raise ValueError(f"Validation set {val_name} not found in data.")

    X_train = np.concatenate(X_train)
    shot_train = np.concatenate(shot_train)
    stress_train = np.concatenate(stress_train)
    task_train = np.concatenate(task_train)

    train_ds = EEGDataset(X_train, (shot_train, stress_train, task_train), transform=train_transforms)
    val_ds   = EEGDataset(X_val, (shot_val, stress_val, task_val), transform=None)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False)

def train_one_fold(val_name, data, labels, epochs=EPOCHS, lr=LR):
    print(f"\n==== LOSO Fold: Validation on {val_name} ====\n")

    train_loader, val_loader = get_loso_loaders(data, labels, val_name, BATCH_SIZE)

    model = MultiTaskResNet().to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, shot, stress, task in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x = x.to(DEVICE)
            shot = shot.to(DEVICE)
            stress = stress.to(DEVICE).float()
            task = task.to(DEVICE)

            pred_shot, pred_stress, pred_task = model(x)

            loss = (
                ce(pred_shot, shot) +
                mse(pred_stress.squeeze(), stress) +
                ce(pred_task, task)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}")

    # Save the model for this fold
    torch.save(model.state_dict(), f"model_{val_name}.pth")
    print(f"Fold complete. Model saved to model_{val_name}.pth\n")
    return model

def main():
    if not os.path.exists(CACHE_DIR):
        print(f"Creating {CACHE_DIR} directory...")
        os.makedirs(CACHE_DIR)
        # Here you might want to call the data generation logic if you had the raw files
        # For now, we assume the user will populate it or has populated it.
        print("WARNING: Cache directory is empty. Please ensure .npy files are present.")

    try:
        data, labels = load_cached_data()
    except FileNotFoundError as e:
        print(e)
        return

    for val_name in data.keys():
        train_one_fold(val_name, data, labels)

if __name__ == "__main__":
    main()
