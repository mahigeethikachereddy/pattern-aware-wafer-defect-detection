"""
Standalone prediction script for batch inference on new wafer images.

Run: python src/predict.py
"""

import sys
import os
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import WaferCNN
from src.data_loader import CLASSES, IMG_SIZE

MODEL_PATH = "models/wafer_cnn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_image(image_path, model):
    """
    Predict the defect class of a wafer image.
    
    Args:
        image_path: Path to the image file
        model: Trained WaferCNN model
    
    Returns:
        pred_class: Class name string
        confidence: Confidence percentage
        cam: Grad-CAM heatmap
        probs: All class probabilities
    """
    # Load and preprocess
    img = Image.open(image_path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0, 1)
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx] * 100

    # Generate Grad-CAM
    tensor.requires_grad_(True)
    model.zero_grad()
    output = model(tensor)
    output[0, pred_idx].backward()

    gradients = model.gradients[0].cpu().detach().numpy()
    activations = model.activations[0].cpu().detach().numpy()
    weights = gradients.mean(axis=(1, 2))
    cam = np.einsum("c,chw->hw", weights, activations)
    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max > 1e-8:
        cam = cam / cam_max

    return CLASSES[pred_idx], confidence, cam, probs


if __name__ == "__main__":
    print("=" * 60)
    print("Wafer Defect Prediction")
    print("=" * 60)

    model = WaferCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print()

    # Find sample images
    test_dir = "data/processed"
    if os.path.exists(MODEL_PATH):
        # Demo with a test sample
        X_test = np.load(f"{test_dir}/X_test.npy")
        y_test_names = np.load(f"{test_dir}/y_test_names.npy")

        idx = np.random.randint(0, len(X_test))
        print(f"Demo prediction on sample {idx}:")
        print(f"  True class: {y_test_names[idx]}")

        # Create tensor from numpy array
        tensor = torch.from_numpy(X_test[idx]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[pred_idx] * 100

        print(f"  Predicted:  {CLASSES[pred_idx]}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  All probabilities: {dict(zip(CLASSES, (probs * 100).round(1)))}")
    else:
        print("No model found. Train first with: python src/train.py")
