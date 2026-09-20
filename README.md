# Pattern-Aware AI for Semiconductor Wafer Defect Detection and Localization

## Overview
A computer vision prototype that analyzes semiconductor wafer map images, classifies known wafer defect patterns, and uses Grad-CAM to visually highlight the regions that contributed to the prediction.

**Dataset**: WM-811K (public Kaggle dataset)  
**Type**: Software research prototype — does NOT predict future manufacturing defects or control any real semiconductor manufacturing process.

## Test Results
| Metric | Value |
|--------|-------|
| **Accuracy** | **86.0%** |
| **Precision (macro)** | **80.5%** |
| **Recall (macro)** | **85.3%** |
| **F1-score (macro)** | **81.8%** |

## Project Structure
```
CapProject/
├── data/
│   ├── raw/          # LSWMD.pkl (original WM-811K dataset)
│   └── processed/    # Preprocessed numpy arrays and labels
├── models/            # Saved trained PyTorch model
│   ├── wafer_cnn.pth
│   └── model_metadata.json
├── outputs/           # All visualizations and results
│   ├── eda/           # EDA plots (class distribution, samples, etc.)
│   ├── confusion_matrix.png
│   ├── gradcam_results.png
│   └── evaluation_results.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Dataset loading and preprocessing
│   ├── eda.py            # Exploratory data analysis
│   ├── model.py          # CNN model architecture + Grad-CAM hooks
│   ├── train.py          # Training pipeline with class weighting
│   ├── evaluate.py       # Evaluation metrics + Grad-CAM visualization
│   ├── gradcam.py        # Standalone Grad-CAM module
│   ├── app.py            # Streamlit web application
│   └── predict.py        # Batch prediction script
├── main.py              # Pipeline entry point
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Stage 1 — Dataset & Preprocessing
```bash
python src/data_loader.py       # Load WM-811K and save processed arrays
python src/eda.py               # Run exploratory data analysis
```

### 3. Train the model
```bash
python src/train.py             # Train the CNN (takes ~10 min on CPU)
```

### 4. Evaluate
```bash
python src/evaluate.py          # Evaluate metrics, confusion matrix, Grad-CAM
```

### 5. Run the web app
```bash
streamlit run src/app.py
```

### Quick demo (without training)
```bash
python main.py --stage all
```

## Classes (8 defect patterns)
- **Center**    — Circular defect at wafer center
- **Donut**      — Ring-shaped defect
- **Edge-Loc**   — Localized defect near wafer edge
- **Edge-Ring**  — Ring defect at wafer edge
- **Loc**        — Small localized defect
- **Random**     — Randomly distributed defects
- **Scratch**    — Linear scratch pattern
- **Near-full**  — Defect covering most of the wafer

## Technical Details
- **Input**: 64×64 grayscale wafer maps (values: 0=no die, 0.5=good, 1=defect)
- **Model**: 4-layer CNN with 241,992 parameters
- **Class Weighting**: Inverse frequency weighting to handle 63.7x class imbalance
- **Explainability**: Grad-CAM with hooks on the last convolutional layer
- **Device**: CPU (no GPU required)

## Disclaimer
This is a research prototype using publicly available WM-811K wafer map data. It does NOT predict future manufacturing defects or interface with any semiconductor manufacturing process. Grad-CAM highlights regions of interest for model explainability — it does NOT provide exact physical defect boundaries.
