"""
Full Pipeline: Download missing videos, process them, then train the model.
Runs download_and_build_kathak_dataset.py followed by train_kathak_model.py
"""

import sys
import os

# Ensure backend is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("  PHASE 1: Downloading & Processing Kathak Dataset Videos")
print("=" * 70)

from app.pipeline.download_and_build_kathak_dataset import run_pipeline
run_pipeline()

print("\n" + "=" * 70)
print("  PHASE 2: Training Kathak LSTM Model on Full Dataset")
print("=" * 70 + "\n")

from app.pipeline.train_kathak_model import train_kathak_model
train_kathak_model()

print("\nFull pipeline complete!")
