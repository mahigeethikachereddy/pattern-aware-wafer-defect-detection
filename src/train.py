"""
Stage 2: Model Training Pipeline
==================================
Trains the CNN model on wafer defect data with class weighting
to handle severe class imbalance.

Run: python src/train.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_processed, CLASSES, IMG_SIZE
from src.model import WaferCNN

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CONFIG = {
    "img_size": IMG_SIZE,
    "batch_size": 64,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": len(CLASSES),
    "classes": CLASSES,
    "model_path": "models/wafer_cnn.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "scheduler_step": 7,
    "scheduler_gamma": 0.5,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ------------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------------
class NumpyDataset(Dataset):
    """Wraps numpy arrays for PyTorch DataLoader."""

    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # shape (64, 64)
        label = self.y[idx]
        # Add channel dimension: (64, 64) -> (1, 64, 64)
        img = img.unsqueeze(0) if img.dim() == 2 else img
        if self.transform:
            img = self.transform(img)
        return img, label


# ------------------------------------------------------------------
# Compute class weights for imbalanced dataset
# ------------------------------------------------------------------
def compute_class_weights(y_train):
    """
    Compute inverse frequency class weights.
    weight_c = total_samples / (num_classes * samples_in_class_c)
    """
    class_counts = np.bincount(y_train, minlength=len(CLASSES))
    total = len(y_train)
    weights = total / (len(CLASSES) * class_counts)
    # Normalize to average 1.0
    weights = weights / weights.mean()
    
    print("Class weights (for handling imbalance):")
    for i, (cls, count, weight) in enumerate(zip(CLASSES, class_counts, weights)):
        print(f"  {cls:15s}: {count:6d} samples → weight={weight:.3f}")
    
    return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------------
# Training functions
# ------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        correct += (output.argmax(1) == y_batch).sum().item()
        total += x_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        loss = criterion(output, y_batch)

        total_loss += loss.item() * x_batch.size(0)
        preds = output.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ------------------------------------------------------------------
# Main training pipeline
# ------------------------------------------------------------------
def run_training():
    """Full training pipeline."""
    print("=" * 60)
    print("STAGE 2: Model Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['lr']}")
    print()

    # Load data
    print("[1/6] Loading processed data...")
    X_train, y_train, _ = load_processed("train")
    X_val, y_val, _ = load_processed("val")
    X_test, y_test, _ = load_processed("test")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Create datasets and dataloaders
    print("[2/6] Creating data loaders...")
    train_dataset = NumpyDataset(X_train, y_train)
    val_dataset = NumpyDataset(X_val, y_val)
    test_dataset = NumpyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=False, num_workers=0, pin_memory=False)

    # Create model
    print("[3/6] Creating model...")
    model = WaferCNN(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Compute class weights
    print("[4/6] Computing class weights...")
    class_weights = compute_class_weights(y_train)
    class_weights = class_weights.to(CONFIG["device"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                           step_size=CONFIG["scheduler_step"],
                                           gamma=CONFIG["scheduler_gamma"])

    # Training loop
    print("[5/6] Training model...")
    os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)
    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    for epoch in range(1, CONFIG["epochs"] + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, CONFIG["device"]
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, CONFIG["device"]
        )

        # Step scheduler
        scheduler.step()

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(scheduler.get_last_lr()[0])

        epoch_time = time.time() - start_time
        print(f"  Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} | "
              f"time={epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["model_path"])
            print(f"    ✓ Saved best model (val_acc={val_acc:.4f})")

    # Save training history
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[✓] Best validation accuracy: {best_val_acc:.4f}")
    print(f"[✓] Model saved to {CONFIG['model_path']}")

    # Save metadata
    meta = {
        "config": CONFIG,
        "best_val_acc": best_val_acc,
        "class_names": CLASSES,
        "class_weights": class_weights.cpu().tolist()
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    model.load_state_dict(torch.load(CONFIG["model_path"],
                                      map_location=CONFIG["device"]))
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion, CONFIG["device"]
    )
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Save predictions for evaluation script
    np.save("outputs/y_test_true.npy", y_true)
    np.save("outputs/y_test_pred.npy", y_pred)

    print(f"\n[✓] Training complete. Model ready for evaluation.")
    return model, history


if __name__ == "__main__":
    run_training()
