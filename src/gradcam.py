"""
Grad-CAM Implementation
==========================
Standalone Grad-CAM module for model explainability.
Can be used independently or imported.

Run: python src/gradcam.py  (demo)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import WaferCNN
from src.data_loader import load_processed, CLASSES


def generate_gradcam(model, x, class_idx=None):
    """
    Generate a Grad-CAM heatmap for a single input.
    
    Args:
        model: Trained WaferCNN model
        x: Input tensor (1, 1, 64, 64)
        class_idx: Target class index (default: predicted)
    
    Returns:
        cam: Normalized heatmap (64, 64)
        pred_idx: Predicted class index
        probs: Softmax probabilities array
        class_name: Name of predicted class
    """
    model.eval()
    x = x.clone().requires_grad_(True)

    # Forward pass
    output = model(x)

    # Get predicted class if not specified
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, class_idx].backward()

    # Extract gradients and activations
    gradients = model.gradients[0].cpu().detach().numpy()
    activations = model.activations[0].cpu().detach().numpy()

    # Global average pooling of gradients → channel weights
    weights = gradients.mean(axis=(1, 2))  # (C,)

    # Weighted combination of feature maps
    cam = np.einsum("c,chw->hw", weights, activations)
    cam = np.maximum(cam, 0)  # ReLU for positive contributions only

    # Normalize to [0, 1]
    cam_max = cam.max()
    if cam_max > 1e-8:
        cam = cam / cam_max

    # Softmax probabilities
    probs = F.softmax(output, dim=1)[0].cpu().detach().numpy()
    class_name = CLASSES[class_idx]

    return cam, class_idx, probs, class_name


def visualize_gradcam(model, X, y, y_names, n_samples=6, output_path="outputs/gradcam.png"):
    """
    Create a visualization grid with original, Grad-CAM, and prediction.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_samples = min(n_samples, len(X))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        idx = np.random.randint(0, len(X))
        img = X[idx]
        true_name = y_names[idx]

        x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        cam, pred_idx, probs, pred_name = generate_gradcam(model, x)
        confidence = probs[pred_idx] * 100

        # Original
        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"True: {true_name}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Grad-CAM overlay
        axes[i, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(cam, cmap="jet", alpha=0.5)
        axes[i, 1].set_title(f"Pred: {pred_name} ({confidence:.1f}%)", fontsize=10)
        axes[i, 1].axis("off")

        # Probability bar
        colors = ["#e74c3c" if c == pred_name else "#3498db" for c in CLASSES]
        axes[i, 2].barh(range(len(CLASSES)), probs * 100, color=colors)
        axes[i, 2].set_yticks(range(len(CLASSES)))
        axes[i, 2].set_yticklabels(CLASSES, fontsize=8)
        axes[i, 2].set_xlim(0, 100)
        axes[i, 2].set_xlabel("Probability (%)", fontsize=9)
        axes[i, 2].tick_params(axis="y", labelsize=7)

    plt.suptitle("Grad-CAM: Model Explainability for Wafer Defect Detection",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Grad-CAM visualization saved to {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Grad-CAM Demo")
    print("=" * 60)

    # Load data and model
    X_test, y_test, y_test_names = load_processed("test")
    model = WaferCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load("models/wafer_cnn.pth", map_location="cpu"))
    model.eval()

    print(f"Loaded model and {len(X_test)} test samples")
    print()

    # Generate visualization
    visualize_gradcam(model, X_test, y_test, y_test_names, n_samples=6)

    # Single sample demo
    print("\nSingle sample demo:")
    idx = np.random.randint(0, len(X_test))
    x = torch.from_numpy(X_test[idx]).float().unsqueeze(0).unsqueeze(0)
    cam, pred_idx, probs, pred_name = generate_gradcam(model, x)
    true_name = y_test_names[idx]

    print(f"  True class:    {true_name}")
    print(f"  Predicted:     {pred_name}")
    print(f"  Confidence:    {probs[pred_idx]*100:.1f}%")
    print(f"  Heatmap range: [{cam.min():.3f}, {cam.max():.3f}]")
