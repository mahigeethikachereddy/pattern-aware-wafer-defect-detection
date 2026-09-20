"""
Stage 4: Streamlit Web Application
======================================
Upload a wafer map image to get:
- Predicted defect class
- Confidence score
- Wafer visualization
- Grad-CAM heatmap showing suspicious regions

Run: streamlit run src/app.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from src.model import WaferCNN
from src.data_loader import CLASSES, IMG_SIZE
from src.chatbot import get_chatbot_response

# ------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Wafer Defect Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/wafer_cnn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_trained_model():
    """Load the trained model."""
    model = WaferCNN(num_classes=len(CLASSES)).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    else:
        return None


def preprocess_upload(uploaded_file):
    """Preprocess an uploaded image to match model training preprocessing.
    
    Training preprocessing: wafer_map (values {0, 1, 2}) / 2.0 → {0, 0.5, 1.0}
    Upload preprocessing:   grayscale image → threshold to {0, 1, 2} → / 2.0 → {0, 0.5, 1.0}
    
    Thresholds: < 64 → 0 (no die), 64-191 → 1 (good die), ≥ 192 → 2 (defect)
    """
    img = Image.open(uploaded_file).convert("L")  # Grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    
    # Map pixel values to {0, 1, 2} to match WM-811K wafer map format
    # Training: wafer_map ∈ {0, 1, 2}, then divided by 2.0
    arr = np.where(arr < 64, 0, np.where(arr >= 192, 2, 1)).astype(np.float32)
    
    # Exactly match training: divide by 2.0
    arr = arr / 2.0  # Values in {0, 0.5, 1.0}
    
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
    return tensor, arr


def generate_gradcam(model, x):
    """Generate Grad-CAM heatmap."""
    model.eval()
    x = x.clone().requires_grad_(True)
    output = model(x)
    pred_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_idx].backward()

    gradients = model.gradients[0].cpu().detach().numpy()
    activations = model.activations[0].cpu().detach().numpy()
    weights = gradients.mean(axis=(1, 2))
    cam = np.einsum("c,chw->hw", weights, activations)
    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max > 1e-8:
        cam = cam / cam_max
    
    # Resize heatmap to match original image size (64x64)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    probs = F.softmax(output, dim=1)[0].cpu().detach().numpy()
    return cam, pred_idx, probs


# ------------------------------------------------------------------
# Main App
# ------------------------------------------------------------------
def main():
    # Sidebar
    st.sidebar.title("🔬 Wafer Defect Inspector")
    st.sidebar.markdown("---")
    st.sidebar.write("**Upload a wafer map image to classify defect patterns.**")
    st.sidebar.write("Supported formats: PNG, JPG, JPEG")

    # Model loading status
    model = load_trained_model()
    if model is None:
        st.error(
            f"⚠️ Model not found at `{MODEL_PATH}`.\n\n"
            f"Please train the model first by running:\n"
            f"`python src/train.py`"
        )
        st.info("💡 **Need help?** Check the README.md for instructions.")
        return

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant",
             "content": "Welcome! I'm the **WaferDefect Assistant**. "
                        "Use the Inspector tab to upload and analyze a wafer map, "
                        "then ask me about the prediction."}
        ]

    # Tabs
    tab1, tab2 = st.tabs(["🔬 Inspector", "🤖 WaferDefect Assistant"])

    # ===================== TAB 1: INSPECTOR =====================
    with tab1:
        # Header
        st.title("🔬 Semiconductor Wafer Defect Inspector")
        st.markdown("""
        **AI-powered wafer map analysis with explainable predictions.**

        Upload a wafer map image to classify the defect pattern and see which regions
        contributed to the prediction via Grad-CAM heatmap.

        > ⚠️ **Research Prototype**: This tool analyzes publicly available wafer map
        > data (WM-811K dataset). It does NOT predict future manufacturing defects or
        > control any real semiconductor manufacturing process.
        """)

        # Class info
        st.sidebar.markdown("---")
        st.sidebar.subheader("Defect Classes")
        for i, cls in enumerate(CLASSES):
            st.sidebar.markdown(f"**{i}. {cls}**")

        # Upload area
        uploaded_file = st.file_uploader(
            "📤 Upload a wafer map image",
            type=["png", "jpg", "jpeg"],
            help="Upload a grayscale wafer map image (PNG or JPG)"
        )

        if uploaded_file is not None:
            # Preprocess
            tensor, original_arr = preprocess_upload(uploaded_file)

            # Run prediction
            with st.spinner("🔍 Analyzing wafer map..."):
                tensor = tensor.to(DEVICE)
                with torch.no_grad():
                    output = model(tensor)
                    probs = F.softmax(output, dim=1)[0].cpu().numpy()
                    pred_idx = output.argmax(dim=1).item()
                    confidence = probs[pred_idx] * 100

            # Generate Grad-CAM
                cam, cam_pred_idx, cam_probs = generate_gradcam(model, tensor)

            # Display results
            st.markdown("---")

            # Title for results
            st.subheader("📊 Results")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Prediction")
                st.metric("Defect Class", CLASSES[pred_idx])
                st.metric("Confidence", f"{confidence:.1f}%")

                # Progress bar for confidence
                st.progress(min(confidence / 100, 1.0))

                # Probability breakdown
                st.markdown("#### Class Probabilities")
                prob_data = {cls: float(p * 100) for cls, p in zip(CLASSES, probs)}
                st.bar_chart(prob_data)

                # Anomaly score
                anomaly = probs.max() - probs.mean()
                st.markdown(f"**Anomaly Score**: {anomaly:.3f}")

            with col2:
                st.markdown("### Grad-CAM Heatmap")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(original_arr, cmap="gray", vmin=0, vmax=1)
                ax.imshow(cam, cmap="jet", alpha=0.5)
                ax.set_title(f"Predicted: {CLASSES[pred_idx]} | Confidence: {confidence:.1f}%",
                             fontsize=12, fontweight="bold")
                ax.axis("off")
                st.pyplot(fig)

            # Side-by-side comparison
            st.markdown("### 📸 Visual Comparison")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Original Wafer**")
                fig1, ax1 = plt.subplots(figsize=(5, 5))
                ax1.imshow(original_arr, cmap="gray", vmin=0, vmax=1)
                ax1.set_title("Input Wafer Map", fontsize=11)
                ax1.axis("off")
                st.pyplot(fig1)

            with col2:
                st.markdown("**Defect Map**")
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                # Color code: black=no die, gray=good, red=defect
                colored = np.zeros((*original_arr.shape, 3))
                colored[original_arr < 0.1] = [0, 0, 0]
                colored[(original_arr >= 0.1) & (original_arr < 0.9)] = [0.5, 0.5, 0.5]
                colored[original_arr >= 0.9] = [1, 0, 0]
                ax2.imshow(colored)
                ax2.set_title("Defect Visualization")
                ax2.axis("off")
                st.pyplot(fig2)

            with col3:
                st.markdown("**Grad-CAM Attention**")
                fig3, ax3 = plt.subplots(figsize=(5, 5))
                ax3.imshow(original_arr, cmap="gray", vmin=0, vmax=1)
                ax3.imshow(cam, cmap="jet", alpha=0.6)
                ax3.set_title("Model Focus Areas")
                ax3.axis("off")
                st.pyplot(fig3)

            # Detailed metrics table
            st.markdown("### 📋 Detailed Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Predicted**: {CLASSES[pred_idx]}")
                st.markdown(f"**Confidence**: {confidence:.2f}%")
                st.markdown(f"**Device**: {DEVICE}")
            with col2:
                st.markdown(f"**Image Size**: {IMG_SIZE}×{IMG_SIZE}")
                st.markdown(f"**Input Range**: [{original_arr.min():.3f}, {original_arr.max():.3f}]")
                st.markdown(f"**Anomaly Score**: {anomaly:.4f}")

    # ===================== TAB 2: CHATBOT =====================
    with tab2:
        st.markdown("### 🤖 WaferDefect Assistant")
        st.markdown(
            "Ask me about the WM-811K dataset, defect classes, CNN architecture, "
            "Grad-CAM, model evaluation, or the project. Upload a wafer image first "
            "to enable prediction-specific questions."
        )

        # Display chat messages
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_query = st.chat_input("Ask about wafer defect detection...")

        if user_query:
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Build prediction state from current upload if available
            prediction_state = None
            if uploaded_file is not None:
                prediction_state = {
                    "pred_class": CLASSES[pred_idx],
                    "confidence": confidence,
                    "probs": probs,
                    "cam": True,
                    "anomaly": anomaly
                }

            # Generate response
            response = get_chatbot_response(user_query, prediction_state)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)


if __name__ == "__main__":
    main()
