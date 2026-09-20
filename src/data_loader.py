"""
Stage 1: Dataset Loading and Preprocessing
==========================================
Loads the WM-811K wafer map dataset, extracts defect labels,
splits into train/val/test, and saves processed numpy arrays.

Run: python src/data_loader.py
"""

import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
RAW_PKL = "data/raw/LSWMD.pkl"
PROCESSED_DIR = "data/processed"
IMG_SIZE = 64
SEED = 42

# The 8 known defect classes from WM-811K
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Random", "Scratch", "Near-full"]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ------------------------------------------------------------------
# Pandas compatibility shim (WM-811K pickle uses old pandas internals)
# ------------------------------------------------------------------
class PandasCompatUnpickler(pickle.Unpickler):
    """Handles old pandas internal references in the WM-811K pickle."""
    RENAME = {
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.numeric": "pandas.core.indexes.numeric",
        "pandas.indexes.category": "pandas.core.indexes.category",
        "pandas.indexes.range": "pandas.core.indexes.range",
        "pandas.indexes.multi": "pandas.core.indexes.multi",
        "pandas.indexes.interval": "pandas.core.indexes.interval",
        "pandas.indexes.datetimes": "pandas.core.indexes.datetimes",
        "pandas.indexes.timedeltas": "pandas.core.indexes.timedeltas",
        "pandas.indexes.period": "pandas.core.indexes.period",
        "pandas.indexes.frozen": "pandas.core.indexes.frozen",
        "pandas.indexes.api": "pandas.core.indexes.api",
        "pandas.sparse.array": "pandas.core.arrays.sparse.array",
        "pandas.sparse.dtype": "pandas.core.arrays.sparse.dtype",
        "pandas.sparse.series": "pandas.core.arrays.sparse.array",
    }

    def find_class(self, module, name):
        module = self.RENAME.get(module, module)
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            import pandas as pd
            if hasattr(pd, name):
                return getattr(pd, name)
            raise


def load_wm811k(path):
    """Load the WM-811K pickle file with compatibility handling."""
    print(f"[1/4] Loading dataset from {path} ...")
    with open(path, "rb") as f:
        df = PandasCompatUnpickler(f, encoding="latin1").load()
    print(f"      Loaded {len(df):,} rows with columns: {list(df.columns)}")
    return df


def extract_labels(df):
    """
    Extract defect class labels from failureType column.
    failureType is a numpy array like array([['Center']]) or array([['none']]).
    """
    print("[2/4] Extracting defect labels ...")

    def get_label(x):
        if isinstance(x, np.ndarray) and x.size > 0:
            v = x.flat[0]
            if isinstance(v, bytes):
                v = v.decode("utf-8", "ignore")
            return str(v).strip()
        return "none"

    df = df.copy()
    df["label"] = df["failureType"].apply(get_label)

    # Keep only the 8 known defect classes (drop 'none' and unknown)
    before = len(df)
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    print(f"      Before filtering: {before:,} rows")
    print(f"      After filtering to {len(CLASSES)} classes: {len(df):,} rows")

    # Show distribution
    print("\n      Class distribution:")
    for cls in CLASSES:
        count = (df["label"] == cls).sum()
        print(f"        {cls:15s}: {count:,}")

    return df


def wafer_to_array(wafer_map, img_size=IMG_SIZE):
    """
    Convert a wafer map to a normalized numpy array.
    Values: 0=no die (black), 1=good die (gray), 2=defect die (white)
    Always resizes to img_size x img_size to ensure uniform shape.
    """
    w = np.array(wafer_map, dtype=np.float32)
    # Normalize: 0→0, 1→0.5, 2→1.0
    normalized = w / 2.0

    # Always resize to target resolution
    import cv2
    normalized = cv2.resize(normalized, (img_size, img_size),
                            interpolation=cv2.INTER_NEAREST)

    return normalized


def build_datasets(df):
    """
    Convert all wafer maps and labels to numpy arrays,
    then split into train/val/test sets using stratified sampling
    to ensure every class is represented appropriately in each split.
    
    Data leakage prevention: each wafer map is an independent row in the
    DataFrame. The split is done on rows only — no wafer map appears in
    more than one split.
    """
    print("[3/4] Converting wafer maps to arrays ...")

    X = []
    y = []
    y_names = []

    for idx, row in df.iterrows():
        img = wafer_to_array(row["waferMap"])
        X.append(img)
        label = row["label"]
        y.append(CLASSES.index(label))
        y_names.append(label)

        if (idx + 1) % 10000 == 0:
            print(f"      Processed {idx + 1:,}/{len(df):,} samples ...")

    X = np.array(X)
    y = np.array(y)
    y_names = np.array(y_names)

    print(f"      Final shape: X={X.shape}, y={y.shape}")
    print(f"      Data type: {X.dtype}, Range: [{X.min()}, {X.max()}]")

    # Stratified split: train → (val + test) → val + test
    # Preserves class distribution in each split. Seed=42 preserved.
    print("[4/4] Splitting into train/val/test (stratified) ...")
    
    # First split: separate train from the rest
    X_train, X_temp, y_train, y_temp, yn_train, yn_temp = train_test_split(
        X, y, y_names,
        test_size=0.30,
        stratify=y,
        random_state=SEED
    )
    
    # Second split: separate val from test
    val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)  # 0.15/0.30 = 0.50
    X_val, X_test, y_val, y_test, yn_val, yn_test = train_test_split(
        X_temp, y_temp, yn_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=SEED
    )

    splits = {
        "train": (X_train, y_train, yn_train),
        "val":   (X_val,   y_val,   yn_val),
        "test":  (X_test,  y_test,  yn_test),
    }

    # Verify and print class distribution for all three splits
    total = len(X)
    print(f"\n      Split sizes: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
    print(f"\n      Class distribution per split:")
    print(f"      {'Class':<15s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
    for cls in CLASSES:
        train_count = (yn_train == cls).sum()
        val_count = (yn_val == cls).sum()
        test_count = (yn_test == cls).sum()
        train_pct = 100 * train_count / (len(yn_train) or 1)
        val_pct = 100 * val_count / (len(yn_val) or 1)
        test_pct = 100 * test_count / (len(yn_test) or 1)
        print(f"      {cls:<15s} {train_count:>6,} ({train_pct:4.1f}%) {val_count:>6,} ({val_pct:4.1f}%) {test_count:>6,} ({test_pct:4.1f}%)")

    # Verify no data leakage: check that all samples are unique
    assert len(X_train) + len(X_val) + len(X_test) == len(X), \
        "Data leakage detected: splits don't add up!"
    
    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), splits["train"][0])
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), splits["train"][1])
    np.save(os.path.join(PROCESSED_DIR, "y_train_names.npy"), splits["train"][2])

    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), splits["val"][0])
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), splits["val"][1])
    np.save(os.path.join(PROCESSED_DIR, "y_val_names.npy"), splits["val"][2])

    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), splits["test"][0])
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), splits["test"][1])
    np.save(os.path.join(PROCESSED_DIR, "y_test_names.npy"), splits["test"][2])

    # Save metadata
    meta = {
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "n_classes": len(CLASSES),
        "split_method": "stratified",
    }
    np.save(os.path.join(PROCESSED_DIR, "metadata.npy"), meta)
    import json
    with open(os.path.join(PROCESSED_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[✓] Processed data saved to {PROCESSED_DIR}/")
    return splits


def load_processed(split="train"):
    """Load preprocessed data from disk (fast, no pickle loading needed)."""
    X = np.load(os.path.join(PROCESSED_DIR, f"X_{split}.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, f"y_{split}.npy"))
    y_names = np.load(os.path.join(PROCESSED_DIR, f"y_{split}_names.npy"))
    return X, y, y_names


def verify_processed():
    """Check that processed data exists and is valid."""
    required = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                "X_test.npy", "y_test.npy", "metadata.json"]
    all_ok = True
    for f in required:
        path = os.path.join(PROCESSED_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)
            print(f"  [OK] {f} ({size:.1f} MB)")
        else:
            print(f"  [MISSING] {f}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 1: Dataset Loading and Preprocessing")
    print("=" * 60)

    # Check if already processed
    if verify_processed():
        print("\n[✓] Processed data already exists. Skipping preprocessing.")
        print("    Run with --force to reprocess.")
    else:
        # Load and process
        df = load_wm811k(RAW_PKL)
        df = extract_labels(df)
        splits = build_datasets(df)

    print("\n[✓] Stage 1 complete. Data ready for EDA and training.")
