"""
Stage 3: Model Evaluation
===========================
Computes comprehensive evaluation metrics including
accuracy, precision, recall, F1-score, and confusion matrix.
Also generates Grad-CAM visualizations.

Run: python src/evaluate.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data_loader import load_processed, CLASSES, IMG_SIZE
from src.model import WaferCNN
from src.train import NumpyDataset

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = "models/wafer_cnn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GradCAM:
    """Grad-CAM implementation for model explainability."""

    def __init__(self, model):
        self.model = model
        model.eval()

    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            x: Input tensor (1, 1, 64, 64)
            class_idx: Target class index (default: predicted)
        
        Returns:
            cam: Heatmap (H, W) normalized to [0, 1]
            pred_idx: Predicted class index
            probs: Softmax probabilities
        """
        x = x.clone().requires_grad_(True)
        output = self.model(x)

        if class_idx is None:
            pred_idx = output.argmax(dim=1).item()
        else:
            pred_idx = class_idx

        self.model.zero_grad()
        output[0, pred_idx].backward()

        # Get gradients and activations from the model
        gradients = self.model.gradients[0].cpu().detach().numpy()
        activations = self.model.activations[0].cpu().detach().numpy()

        # Global average pooling of gradients
        weights = gradients.mean(axis=(1, 2))

        # Weighted combination
        cam = np.einsum("c,chw->hw", weights, activations)
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max

        # Resize heatmap to match original image size for visualization
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().detach().numpy()
        return cam, pred_idx, probs


def evaluate_model():
    """Run full evaluation."""
    print("=" * 60)
    print("STAGE 3: Model Evaluation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # Load data
    print("[1/5] Loading data...")
    X_test, y_test, y_test_names = load_processed("test")
    test_dataset = NumpyDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load model
    print("[2/5] Loading model...")
    model = WaferCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Run evaluation
    print("[3/5] Running predictions...")
    all_preds = []
    all_labels = []
    all_probs = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            probs = torch.nn.functional.softmax(output, dim=1)
            preds = output.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    avg_loss = total_loss / len(y_true)

    # Compute metrics
    print("[4/5] Computing metrics...")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Loss:       {avg_loss:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-score:   {f1:.4f}")
    print()

    # Per-class metrics
    print("Per-Class Metrics:")
    report = classification_report(
        y_true, y_pred, target_names=CLASSES, zero_division=0
    )
    print(report)

    # Confusion matrix
    print("[5/5] Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Wafer Defect Classification")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

    # Save results
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "loss": float(avg_loss),
        "per_class": {}
    }

    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
        results["per_class"][cls] = {
            "precision": float(cls_precision),
            "recall": float(cls_recall),
            "f1": float(cls_f1),
            "support": int(cm[i].sum())
        }

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate Grad-CAM visualizations
    print("\nGenerating Grad-CAM visualizations...")
    gradcam = GradCAM(model)

    # Pick 6 diverse samples
    n_samples = min(6, len(X_test))
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        idx = np.random.randint(0, len(X_test))
        img = X_test[idx]
        true_label = CLASSES[y_test[idx]]

        # Prepare input
        x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

        # Generate Grad-CAM
        cam, pred_idx, probs = gradcam(x)
        pred_label = CLASSES[pred_idx]
        confidence = probs[pred_idx] * 100

        # Original image
        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"True: {true_label}", fontsize=10)
        axes[i, 0].axis("off")

        # Original with Grad-CAM overlay
        axes[i, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(cam, cmap="jet", alpha=0.5)
        axes[i, 1].set_title(f"Grad-CAM ({pred_label})", fontsize=10)
        axes[i, 1].axis("off")

        # Probability bar
        axes[i, 2].barh(CLASSES, probs * 100, color=[
            "red" if c == pred_label else "steelblue" for c in CLASSES
        ])
        axes[i, 2].set_title(f"Conf: {confidence:.1f}%", fontsize=10)
        axes[i, 2].set_xlabel("% Probability")
        axes[i, 2].tick_params(axis="y", labelsize=8)
        axes[i, 2].set_xlim(0, 100)

    plt.suptitle("Grad-CAM Localization Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gradcam_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Grad-CAM results saved to {OUTPUT_DIR}/gradcam_results.png")

    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    evaluate_model()
