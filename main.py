"""
Pattern-Aware AI for Semiconductor Wafer Defect Detection and Localization
===========================================================================
Entry point for the complete project pipeline.

Usage:
    python main.py --stage eda          # Dataset inspection & EDA
    python main.py --stage train        # Train the CNN model
    python main.py --stage evaluate     # Evaluate the model
    python main.py --stage gradcam      # Generate Grad-CAM visualizations
    python main.py --stage app          # Launch Streamlit web application
    python main.py --stage all          # Run all stages sequentially

Run: streamlit run src/app.py
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Wafer Defect Detection — Pipeline Entry Point"
    )
    parser.add_argument(
        "--stage",
        choices=["eda", "train", "evaluate", "gradcam", "app", "all"],
        default="all",
        help="Which stage to run"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Pattern-Aware AI for Semiconductor Wafer Defect Detection")
    print("=" * 60)

    if args.stage in ["eda", "all"]:
        print("\n[Stage 1] Dataset Loading & EDA")
        print("-" * 40)
        os.system("python3 src/data_loader.py")
        os.system("python3 src/eda.py")

    if args.stage in ["train", "all"]:
        print("\n[Stage 2] Model Training")
        print("-" * 40)
        os.system("python3 src/train.py")

    if args.stage in ["evaluate", "all"]:
        print("\n[Stage 3] Model Evaluation")
        print("-" * 40)
        os.system("python3 src/evaluate.py")

    if args.stage in ["gradcam", "all"]:
        print("\n[Stage 3b] Grad-CAM Visualization")
        print("-" * 40)
        os.system("python3 src/gradcam.py")

    if args.stage in ["app", "all"]:
        print("\n[Stage 4] Launching Streamlit App")
        print("-" * 40)
        print("Run this in a separate terminal:")
        print("  streamlit run src/app.py")
        print()
        os.system("streamlit run src/app.py")


if __name__ == "__main__":
    main()
