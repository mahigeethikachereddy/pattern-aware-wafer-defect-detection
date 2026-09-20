"""
Stage 1b: Exploratory Data Analysis (EDA)
=========================================
Analyzes the wafer map dataset to understand class distributions,
sample wafer patterns, and data characteristics.

Run: python src/eda.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_processed, CLASSES

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
OUTPUT_DIR = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color map for wafer visualization
# 0 = black (no die), 0.5 = gray (good die), 1.0 = white (defect)
CMAP_LEVELS = {0: "black", 0.5: "#808080", 1.0: "#ffffff"}


def load_metadata():
    """Load dataset metadata."""
    with open("data/processed/metadata.json") as f:
        return json.load(f)


def plot_class_distribution(y_train, y_test, y_train_names, y_test_names, meta):
    """Plot the class distribution for train and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (y, names, title) in zip(axes, [
        (y_train, y_train_names, "Training Set"),
        (y_test, y_test_names, "Test Set")
    ]):
        counts = Counter(names)
        class_labels = [c for c in CLASSES]
        class_counts = [counts.get(c, 0) for c in class_labels]

        colors = plt.cm.Set3(np.linspace(0, 1, len(CLASSES)))
        bars = ax.barh(class_labels, class_counts, color=colors)
        ax.set_xlabel("Number of Samples")
        ax.set_title(f"Class Distribution — {title}")
        ax.set_xlabel("Count")

        # Add count labels
        for bar, count in zip(bars, class_counts):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    f"{count:,}", va="center", fontsize=9)

        # Add percentage
        total = sum(class_counts)
        for i, (bar, count) in enumerate(zip(bars, class_counts)):
            pct = 100 * count / total if total > 0 else 0
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    f"({pct:.1f}%)", va="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
    print(f"[✓] Saved class_distribution.png")
    plt.close()


def visualize_sample_wafers(X, y, y_names, n_samples_per_class=2, meta=None):
    """Visualize sample wafer maps from each class."""
    n_classes = len(CLASSES)
    n_cols = n_samples_per_class * 2  # original + zoomed
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(16, 4 * n_classes))

    for class_idx, cls_name in enumerate(CLASSES):
        # Find indices of this class
        idx = np.where(y == class_idx)[0]
        if len(idx) == 0:
            continue

        # Pick a few samples
        selected = np.random.choice(idx, min(n_samples_per_class, len(idx)),
                                    replace=False)

        for col_idx, sample_idx in enumerate(selected):
            img = X[sample_idx]

            # Original view
            ax = axes[class_idx, col_idx * 2]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{cls_name} (sample {sample_idx})", fontsize=10)
            ax.axis("off")

            # Zoomed view (center crop)
            ax2 = axes[class_idx, col_idx * 2 + 1]
            h, w = img.shape
            crop_size = min(h, w) // 2
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            crop = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
            ax2.imshow(crop, cmap="gray", vmin=0, vmax=1)
            ax2.set_title(f"  Zoomed", fontsize=10)
            ax2.axis("off")

    plt.suptitle("Wafer Map Samples by Defect Class", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_wafers.png"), dpi=150, bbox_inches="tight")
    print(f"[✓] Saved sample_wafers.png")
    plt.close()


def plot_wafer_value_distribution(X, y, y_names, n_classes=8):
    """Analyze the pixel value distribution across all wafer maps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall pixel value histogram
    ax = axes[0, 0]
    all_pixels = X.flatten()
    ax.hist(all_pixels, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Pixel Value (0=no die, 0.5=good, 1=defect)")
    ax.set_ylabel("Frequency")
    ax.set_title("Overall Pixel Value Distribution")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=1, color="red", linestyle="--", alpha=0.5)

    # 2. Average wafer per class
    ax = axes[0, 1]
    mean_values = []
    for c in range(n_classes):
        class_imgs = X[y == c]
        if len(class_imgs) > 0:
            mean_values.append(np.mean(class_imgs))
        else:
            mean_values.append(0)
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    bars = ax.bar(CLASSES, mean_values, color=colors)
    ax.set_xlabel("Class")
    ax.set_ylabel("Average Pixel Value")
    ax.set_title("Average Wafer Intensity by Class")
    ax.set_xticklabels(CLASSES)
    ax.tick_params(axis="x", labelrotation=45)
    for bar, val in zip(bars, mean_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=8)

    # 3. Defect density per sample (fraction of defect pixels)
    ax = axes[1, 0]
    defect_densities = []
    for c in range(n_classes):
        class_imgs = X[y == c]
        # Defect = pixel value close to 1
        density = np.mean(class_imgs > 0.8) * 100
        defect_densities.append(density)
    bars = ax.bar(CLASSES, defect_densities, color=colors)
    ax.set_xlabel("Class")
    ax.set_ylabel("Defect Pixel %")
    ax.set_title("Defect Density by Class")
    ax.set_xticklabels(CLASSES)
    ax.tick_params(axis="x", labelrotation=45)
    for bar, val in zip(bars, defect_densities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", fontsize=8)

    # 4. Wafer size / shape analysis
    ax = axes[1, 1]
    wafer_sizes = []
    for i in range(min(1000, len(X))):
        img = X[i]
        # Count non-zero pixels to estimate wafer area
        wafer_pixels = np.sum(img > 0.01)
        wafer_sizes.append(wafer_pixels)
    ax.hist(wafer_sizes, bins=50, color="coral", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Wafer Pixels (non-zero)")
    ax.set_ylabel("Frequency")
    ax.set_title("Wafer Area Distribution")

    plt.suptitle("Dataset Characteristics Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "value_distribution.png"), dpi=150, bbox_inches="tight")
    print(f"[✓] Saved value_distribution.png")
    plt.close()


def plot_heatmap_examples(X, y, y_names, n_samples=3):
    """Create a composite heatmap showing defect locations."""
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3 * n_samples))

    for i in range(n_samples):
        # Pick a random sample
        idx = np.random.randint(0, len(X))
        img = X[idx]
        cls_name = y_names[idx]
        class_idx = CLASSES.index(cls_name)

        # Create a colored version: red for defects, gray for good, black for no die
        colored = np.zeros((*img.shape, 3))
        colored[img < 0.1] = [0, 0, 0]          # no die = black
        colored[(img >= 0.1) & (img < 0.9)] = [0.5, 0.5, 0.5]  # good = gray
        colored[img >= 0.9] = [1, 0, 0]          # defect = red

        # Left: original grayscale, Right: colored heatmap
        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Sample {idx} — {cls_name}", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(colored)
        axes[i, 1].set_title(f"Defect Map (Red = Defect)", fontsize=11)
        axes[i, 1].axis("off")

    plt.suptitle("Defect Localization Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_examples.png"), dpi=150, bbox_inches="tight")
    print(f"[✓] Saved heatmap_examples.png")
    plt.close()


def summary_report(meta, X_train, y_train, y_train_names, X_test, y_test, y_test_names):
    """Print a comprehensive summary report."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS — SUMMARY REPORT")
    print("=" * 60)

    print(f"\n📊 Dataset Overview:")
    print(f"   Total classes: {meta['n_classes']}")
    print(f"   Image size: {meta['img_size']}x{meta['img_size']}")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Test samples: {len(X_test):,}")

    print(f"\n📋 Class Distribution:")
    for c in range(meta['n_classes']):
        cls = CLASSES[c]
        train_count = np.sum(y_train_names == cls)
        test_count = np.sum(y_test_names == cls)
        print(f"   {cls:15s}: train={train_count:6,}, test={test_count:6,}")

    print(f"\n📈 Data Characteristics:")
    print(f"   Pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   Train mean: {np.mean(X_train):.4f}")
    print(f"   Train std:  {np.std(X_train):.4f}")

    # Check for class imbalance
    total_train = len(X_train)
    counts = [np.sum(y_train_names == cls) for cls in CLASSES]
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n⚠️  Class Imbalance: {imbalance_ratio:.1f}x (max/min)")
    if imbalance_ratio > 3:
        print("   Recommendation: Use class weighting or oversampling")

    print("\n" + "=" * 60)
    print("✓ EDA complete. Check outputs/eda/ for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 1b: Exploratory Data Analysis")
    print("=" * 60)

    # Load metadata and data
    meta = load_metadata()
    X_train, y_train, y_train_names = load_processed("train")
    X_val, y_val, y_val_names = load_processed("val")
    X_test, y_test, y_test_names = load_processed("test")

    print(f"\nLoaded data:")
    print(f"  Train: {X_train.shape}, {len(y_train)} samples")
    print(f"  Val:   {X_val.shape}, {len(y_val)} samples")
    print(f"  Test:  {X_test.shape}, {len(y_test)} samples")

    # Generate all EDA visualizations
    print("\nGenerating visualizations...")

    plot_class_distribution(y_train, y_test, y_train_names, y_test_names, meta)
    visualize_sample_wafers(X_train, y_train, y_train_names, n_samples_per_class=2, meta=meta)
    plot_wafer_value_distribution(X_train, y_train, y_train_names, meta['n_classes'])
    plot_heatmap_examples(X_train, y_train, y_train_names, n_samples=4)

    # Print summary
    summary_report(meta, X_train, y_train, y_train_names, X_test, y_test, y_test_names)
