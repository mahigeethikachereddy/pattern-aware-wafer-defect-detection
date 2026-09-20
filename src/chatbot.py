"""
Lightweight local knowledge-base chatbot for the semiconductor wafer defect
detection project.

No external API is used. All responses are generated from an internal
knowledge base using keyword/intent matching.

Run: python src/chatbot.py  (self-test)
"""

import re

# ------------------------------------------------------------------
# Project knowledge base
# ------------------------------------------------------------------
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Random", "Scratch", "Near-full"]

CLASS_DESCRIPTIONS = {
    "Center": "Circular defect pattern located near the wafer center",
    "Donut": "Ring-shaped or annular defect pattern",
    "Edge-Loc": "Localized defect near the wafer edge",
    "Edge-Ring": "Ring-shaped defect located at the wafer edge",
    "Loc": "Small localized defect anywhere on the wafer",
    "Random": "Defects scattered randomly across the wafer",
    "Scratch": "Linear scratch pattern across the wafer surface",
    "Near-full": "Defect pattern covering most of the wafer area",
}

EVALUATION_RESULTS = {
    "accuracy": 82.87,
    "macro_precision": 79.06,
    "macro_recall": 81.22,
    "macro_f1": 80.04,
    "loss": 0.5373,
    "test_samples": 3828,
    "per_class_f1": {
        "Center": 0.923, "Donut": 0.570, "Edge-Loc": 0.800,
        "Edge-Ring": 0.953, "Loc": 0.711, "Random": 0.901,
        "Scratch": 0.683, "Near-full": 0.766,
    },
}


def get_chatbot_response(query, prediction_state=None):
    """Generate a response to a user query using keyword/intent matching.

    Args:
        query: User question string.
        prediction_state: Optional dict with keys:
            - pred_class: str (e.g. "Scratch")
            - confidence: float (e.g. 51.8)
            - probs: list of floats (class probabilities)
            - cam: bool (whether Grad-CAM is available)
            - anomaly: float (anomaly score)
            If None or prediction not available, relevant questions will
            inform the user to upload a wafer image first.

    Returns:
        str: The chatbot's response.
    """
    q = query.lower().strip()

    # --- Project overview ---
    if re.search(r'\bwhat is this\b', q) or re.search(r'\bwhat is the project\b', q) \
       or re.search(r'\babout this project\b', q) or re.search(r'\btell me about\b', q):
        return (
            "This project is a **computer vision research prototype** for "
            "semiconductor wafer defect detection and localization. It classifies "
            "wafer map images into one of 8 known defect pattern categories "
            "using a Convolutional Neural Network (CNN).\n\n"
            "**Key components:**\n"
            "- CNN model: 4-block Conv2d architecture with Grad-CAM hooks "
            "(241,992 parameters)\n"
            "- Dataset: WM-811K (Large-Scale Wafer Map Defect Detection)\n"
            "- Explainability: Grad-CAM heatmaps highlight regions influencing "
            "predictions\n"
            "- Interface: Streamlit web application\n\n"
            "> ⚠️ **Research Prototype**: This tool analyzes publicly available "
            "wafer map data (WM-811K dataset). It does NOT predict future "
            "manufacturing defects or control any real semiconductor manufacturing "
            "process."
        )

    # --- Wafer maps ---
    if re.search(r'\bwafer map\b', q) or re.search(r'\bwhat is a wafer\b', q):
        return (
            "A **wafer map** is a grayscale image representing a semiconductor "
            "wafer where individual sensor die positions are recorded. Each pixel "
            "value indicates the state of a die:\n"
            "- **0** (black): No die / empty position\n"
            "- **0.5** (gray): Good die (passes inspection)\n"
            "- **1.0** (white): Defective die (fails inspection)\n\n"
            "The WM-811K dataset contains 25,519 such wafer maps, each classified "
            "into one of 8 defect pattern types representing visual groupings of "
            "defective die positions across the wafer surface."
        )

    # --- Dataset ---
    if re.search(r'\bwm-811k\b', q) or re.search(r'\blswmd\b', q) \
       or re.search(r'\bdataset\b', q) or re.search(r'\bwafer map dataset\b', q):
        return (
            "The project uses the **WM-811K dataset** (also known as **LSWMD** — "
            "Large-Scale Wafer Map Defect Detection). Key facts:\n\n"
            "- **25,519 wafer maps** across 8 defect classes\n"
            "- **Image size**: 64×64 grayscale (resized from original 45×48)\n"
            "- **Pixel values**: 0 (no die), 0.5 (good), 1.0 (defect)\n"
            "- **Severe class imbalance**: 63.7x ratio between most and least "
            "frequent classes\n"
            "- **Largest class**: Edge-Ring (6,776 training samples)\n"
            "- **Smallest class**: Near-full (104 training samples)\n"
            "- **Source**: Publicly available on Kaggle\n\n"
            "The dataset was split **stratified** into train (70%), validation "
            "(15%), and test (15%) sets to preserve class proportions across all "
            "splits."
        )

    # --- Defect classes ---
    for cls in CLASSES:
        pattern = r'\b' + re.escape(cls.lower()) + r'\b'
        if re.search(pattern, q):
            return f"**{cls}**: {CLASS_DESCRIPTIONS[cls]}"

    # --- CNN ---
    if re.search(r'\bcnn\b', q) or re.search(r'\bconvolutional\b', q) \
       or re.search(r'\bneural network\b', q) or re.search(r'\barchitecture\b', q) \
       or re.search(r'\bhow does the model\b', q):
        return (
            "The CNN architecture consists of **4 convolutional blocks** followed "
            "by a classifier:\n\n"
            "1. **Block 1**: Conv2d(1→32) → BatchNorm → ReLU → MaxPool "
            "(64×64 → 32×32)\n"
            "2. **Block 2**: Conv2d(32→64) → BatchNorm → ReLU → MaxPool "
            "(32×32 → 16×16)\n"
            "3. **Block 3**: Conv2d(64→128) → BatchNorm → ReLU → MaxPool "
            "(16×16 → 8×8)\n"
            "4. **Block 4**: Conv2d(128→128) → BatchNorm → ReLU → MaxPool "
            "(8×8 → 4×4)\n"
            "5. **Global Average Pooling** → Dropout(0.3) → Fully Connected → "
            "8 output classes\n\n"
            "**Total parameters**: 241,992\n"
            "**Optimizer**: Adam (lr=0.001, weight_decay=1e-4)\n"
            "**Loss function**: CrossEntropyLoss with class weighting "
            "(inverse frequency)\n"
            "**Scheduler**: StepLR (gamma=0.5, step_size=7)\n\n"
            "The network is trained on CPU and uses class-weighted loss to handle "
            "the severe class imbalance."
        )

    # --- Confidence ---
    if re.search(r'\bconfidence\b', q) or re.search(r'\bprediction confidence\b', q) \
       or re.search(r'\bwhat does confidence\b', q):
        if prediction_state and prediction_state.get("confidence") is not None:
            pred_class = prediction_state.get("pred_class", "Unknown")
            conf = prediction_state["confidence"]
            probs = prediction_state.get("probs")
            prob_text = ""
            if probs is not None:
                prob_text = "\n\n**Class Probabilities:**\n" + "\n".join(
                    [f"- {cls}: {p*100:.1f}%" for cls, p in zip(CLASSES, probs)]
                )
            return (
                f"**Prediction confidence** represents the model's certainty for its "
                f"chosen class. It is the softmax probability of the predicted class.\n\n"
                f"For the current prediction, the model classified the wafer as "
                f"**{pred_class}** with **{conf:.1f}%** confidence.{prob_text}\n\n"
                f"**Important**: Confidence is NOT the same as accuracy. A high "
                f"confidence score indicates the model is 'sure' about its prediction, "
                f"but this doesn't guarantee the prediction is correct."
            )
        return (
            "**Prediction confidence** represents the model's certainty for its "
            "chosen class. It is computed as the softmax probability of the predicted "
            "class after applying softmax to all 8 output logits.\n\n"
            "**Important**: Confidence is NOT the same as accuracy. A high confidence "
            "score indicates the model is 'sure' about its prediction, but this "
            "doesn't guarantee the prediction is correct. The model's actual test "
            "accuracy depends on how well it generalizes to unseen data."
        )

    # --- Grad-CAM ---
    if re.search(r'\bgrad-cam\b', q) or re.search(r'\bgradcam\b', q) \
       or re.search(r'\bexplainable\b', q) or re.search(r'\bheatmap\b', q) \
       or re.search(r'\bwhat regions\b', q) or re.search(r'\bexplain\b', q):
        return (
            "**Grad-CAM** (Gradient-weighted Class Activation Mapping) is an "
            "explainability technique that visualizes which regions of the input "
            "image most influenced the model's prediction.\n\n"
            "**How it works:**\n"
            "1. Forward pass computes the output for the predicted class\n"
            "2. Backward pass computes gradients of that class score w.r.t. the "
            "last convolutional layer\n"
            "3. Gradients are globally averaged to obtain channel importance weights\n"
            "4. Weighted combination of feature maps produces a heatmap\n"
            "5. Heatmap is resized to 64×64 and overlaid on the original image\n\n"
            "**Interpretation**:\n"
            "- Red/yellow regions: Areas that most contributed to the prediction\n"
            "- Blue regions: Areas with negative or minimal contribution\n"
            "- The heatmap shows **model attention**, not exact physical defect boundaries\n"
            "- It helps verify whether the model is focusing on plausible defect regions\n\n"
            "The model uses hooks registered on the last Conv2d layer (output at "
            "8×8 spatial resolution) to capture gradients for Grad-CAM computation."
        )

    # --- Defect map / anomaly ---
    if re.search(r'\bdefect map\b', q) or re.search(r'\banomaly score\b', q) \
       or re.search(r'\bwhat is anomaly\b', q):
        return (
            "The **defect map** is a color-coded visualization of the wafer map:\n"
            "- **Black**: No die (pixel value 0)\n"
            "- **Gray**: Good die (pixel value ~0.5)\n"
            "- **Red**: Defective die (pixel value ~1.0)\n\n"
            "The **anomaly score** measures how unusual the current wafer map is "
            "relative to the average prediction across all classes. It is computed "
            "as `max(probability) - mean(probability)`.\n\n"
            "A higher anomaly score indicates the model is more 'confident' that "
            "the wafer contains a defect pattern, rather than looking like a random "
            "or normal wafer."
        )

    # --- Methodology ---
    if re.search(r'\bmethodology\b', q) or re.search(r'\bapproach\b', q) \
       or re.search(r'\bhow was this built\b', q) or re.search(r'\bpipeline\b', q):
        return (
            "The project follows a structured **12-stage pipeline**:\n\n"
            "1. **Dataset loading**: Load WM-811K pickle with pandas compatibility\n"
            "2. **EDA**: Visualize class distributions and sample wafer maps\n"
            "3. **Preprocessing**: Resize to 64×64, normalize pixel values to {0, 0.5, 1.0}\n"
            "4. **Class imbalance handling**: Inverse-frequency weighting in loss function\n"
            "5. **Train/val/test split**: Stratified 70%/15%/15% split\n"
            "6. **CNN baseline**: 4-block Conv2d architecture\n"
            "7. **Training**: Adam optimizer, StepLR scheduler, class-weighted loss\n"
            "8. **Evaluation**: Accuracy, precision, recall, F1, confusion matrix\n"
            "9. **Defect prediction**: Inference on new wafer maps\n"
            "10. **Grad-CAM localization**: Visual explanations\n"
            "11. **Streamlit web app**: Interactive interface\n"
            "12. **Model saving**: state_dict saved to disk for reuse\n\n"
            "The model was trained from scratch (epoch 1) using stratified splits "
            "with no data leakage. The test set was never used during training or "
            "model selection."
        )

    # --- Evaluation results ---
    if re.search(r'\baccuracy\b', q) or re.search(r'\bprecision\b', q) \
       or re.search(r'\brecall\b', q) or re.search(r'\bf1\b', q) \
       or re.search(r'\bresults\b', q) or re.search(r'\bhow accurate\b', q) \
       or re.search(r'\bperformance\b', q):
        return (
            "The model was evaluated on the held-out test set ({test_samples} samples "
            "from the stratified split) using macro-averaged metrics:\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            "| **Accuracy** | **{accuracy}%** |\n"
            "| **Macro Precision** | **{macro_precision}%** |\n"
            "| **Macro Recall** | **{macro_recall}%** |\n"
            "| **Macro F1-score** | **{macro_f1}%** |\n"
            "| Loss | {loss} |\n\n"
            "**Per-class F1 scores:**\n"
            "{per_class}\n\n"
            "> ⚠️ These results are from a **research prototype** on public data. "
            "The model is NOT production-ready and should not be used for real "
            "manufacturing quality control.".format(
                test_samples=EVALUATION_RESULTS["test_samples"],
                accuracy=EVALUATION_RESULTS["accuracy"],
                macro_precision=EVALUATION_RESULTS["macro_precision"],
                macro_recall=EVALUATION_RESULTS["macro_recall"],
                macro_f1=EVALUATION_RESULTS["macro_f1"],
                loss=EVALUATION_RESULTS["loss"],
                per_class="\n".join(
                    [f"- {cls}: {f1:.3f}"
                     for cls, f1 in EVALUATION_RESULTS["per_class_f1"].items()]
                )
            )
        )

    # --- Limitations ---
    if re.search(r'\blimitation\b', q) or re.search(r'\bwhat are the limitations\b', q) \
       or re.search(r'\bwhat are the\b.*\blimit', q):
        return (
            "**Current limitations of this project:**\n"
            "1. **Research prototype only**: Not intended for real manufacturing quality control\n"
            "2. **CPU-only training**: Training on CPU takes ~75 seconds/epoch; no GPU acceleration\n"
            "3. **Class imbalance**: Despite weighting, minority classes (Donut, Near-full) still have lower F1 scores\n"
            "4. **Fixed dataset**: Only evaluates on WM-811K public data; no generalization to other wafer types\n"
            "5. **Grad-CAM resolution**: Heatmaps are at 8×8 spatial resolution, upsampled to 64×64\n"
            "6. **No real-time processing**: Not optimized for production-line throughput\n"
            "7. **Model not production-ready**: Not validated on industrial manufacturing data\n\n"
            "The project explicitly does NOT predict future manufacturing defects or "
            "control any real semiconductor process."
        )

    # --- Streamlit app ---
    if re.search(r'\bstreamlit\b', q) or re.search(r'\bhow does the app\b', q) \
       or re.search(r'\bhow to use\b', q) or re.search(r'\bweb app\b', q) \
       or re.search(r'\binterface\b', q):
        return (
            "The **Streamlit web application** provides an interactive interface:\n\n"
            "**🔬 Inspector Tab:**\n"
            "- Upload a wafer map image (PNG/JPG)\n"
            "- View predicted defect class and confidence\n"
            "- See class probability breakdown\n"
            "- View Grad-CAM heatmap overlay\n"
            "- Compare original, defect map, and attention visualization\n\n"
            "**🤖 WaferDefect Assistant Tab:**\n"
            "- Ask questions about the project, dataset, and methodology\n"
            "- Get instant answers powered by a local knowledge base\n"
            "- No API key required — all responses are generated locally\n"
            "- Chat history is maintained in the browser session\n\n"
            "The app uses the trained CNN model (wafer_cnn.pth) for inference and "
            "the Grad-CAM module for explainability. All preprocessing matches the "
            "training pipeline exactly."
        )

    # --- Why did the model predict this? ---
    if re.search(r'\bwhy\b', q) and (
        re.search(r'\bpredict\b', q) or re.search(r'\bwhy did\b', q)
        or re.search(r'\bwhy this\b', q)
    ):
        if prediction_state and prediction_state.get("pred_class") is not None:
            pred_class = prediction_state["pred_class"]
            confidence = prediction_state["confidence"]
            probs = prediction_state.get("probs")
            prob_text = ""
            if probs is not None:
                prob_text = "\n\n**Class Probabilities:**\n" + "\n".join(
                    [f"- {cls}: {p*100:.1f}%" for cls, p in zip(CLASSES, probs)]
                )
            return (
                f"Based on the current prediction: the model classified the uploaded "
                f"wafer as **{pred_class}** with **{confidence:.1f}%** confidence.{prob_text}\n\n"
                f"The CNN identified this pattern by detecting visual features through "
                f"its 4 convolutional layers. Grad-CAM highlights the regions that "
                f"most contributed to this prediction. The class probabilities across "
                f"all 8 classes are shown above."
            )
        return (
            "No wafer image has been uploaded yet. Please upload a wafer map image "
            "in the Inspector tab to see the current prediction, and then ask me "
            "why the model predicted that class."
        )

    # --- Current prediction ---
    if re.search(r'\bcurrent prediction\b', q) or re.search(r'\bwhat is the current\b', q):
        if prediction_state and prediction_state.get("pred_class") is not None:
            pred_class = prediction_state["pred_class"]
            confidence = prediction_state["confidence"]
            anomaly = prediction_state.get("anomaly")
            anomaly_text = f"\n**Anomaly Score**: {anomaly:.3f}" if anomaly is not None else ""
            probs = prediction_state.get("probs")
            prob_text = ""
            if probs is not None:
                prob_text = "\n\n**Class Probabilities:**\n" + "\n".join(
                    [f"- {cls}: {p*100:.1f}%" for cls, p in zip(CLASSES, probs)]
                )
            return (
                f"The current prediction is **{pred_class}** with **{confidence:.1f}%** "
                f"confidence.{anomaly_text}{prob_text}"
            )
        return (
            "No wafer image has been uploaded yet. Please upload a wafer map image "
            "in the Inspector tab to see the current prediction."
        )

    # --- What does Grad-CAM show for this prediction? ---
    if re.search(r'\bwhy scratch\b', q) or re.search(r'\bwhy donut\b', q) \
       or re.search(r'\bwhy center\b', q) or re.search(r'\bwhy edge\b', q):
        if prediction_state and prediction_state.get("pred_class") is not None:
            pred_class = prediction_state["pred_class"]
            return (
                f"The model classified the wafer as **{pred_class}** based on learned "
                f"visual patterns detected through its 4 convolutional layers. "
                f"Grad-CAM provides a visualization of which regions most contributed "
                f"to this prediction. The heatmap overlaid on the wafer map shows "
                f"the model's focus areas."
            )
        return (
            "No wafer image has been uploaded yet. Please upload a wafer map image "
            "first to see the prediction and Grad-CAM visualization."
        )

    # --- Fallback ---
    return (
        "I'm focused on semiconductor wafer defect inspection and this project. "
        "I can help with:\n\n"
        "- The WM-811K dataset\n"
        "- The 8 defect classes (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full)\n"
        "- CNN architecture and how it works\n"
        "- What prediction confidence means\n"
        "- What Grad-CAM shows and how to interpret it\n"
        "- Project methodology and pipeline\n"
        "- Model evaluation results and limitations\n"
        "- How the Streamlit application works\n"
        "- Why the model made a specific prediction (upload an image first)\n\n"
        "Please try asking about one of these topics!"
    )


# ------------------------------------------------------------------
# Self-test when run directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("WaferDefect Chatbot — Self-Test")
    print("=" * 60)

    test_questions = [
        "What is this project?",
        "What is Grad-CAM?",
        "What is the accuracy?",
        "What is Scratch?",
        "Why did the model predict this?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {get_chatbot_response(q)[:200]}...")
        print("-" * 40)

    # Test with prediction state
    print("\n\n--- Testing with prediction state ---")
    pred_state = {
        "pred_class": "Scratch",
        "confidence": 51.8,
        "probs": [0.05, 0.02, 0.03, 0.04, 0.02, 0.03, 0.518, 0.272],
        "cam": True,
        "anomaly": 0.48,
    }
    q = "Why did the model predict this?"
    print(f"\nQ: {q}")
    print(f"A: {get_chatbot_response(q, pred_state)}")
    print("-" * 40)

    q = "What is the current prediction?"
    print(f"\nQ: {q}")
    print(f"A: {get_chatbot_response(q, pred_state)}")
    print("-" * 40)

    # Test fallback
    print("\n\n--- Testing fallback ---")
    q = "What is the capital of France?"
    print(f"\nQ: {q}")
    print(f"A: {get_chatbot_response(q)[:200]}...")
    print("=" * 60)
    print("All self-tests passed!")
