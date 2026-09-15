"""
Phase 2 Only: Train the Kathak LSTM model on the already-processed dataset.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("  Training Kathak LSTM Model on Full Dataset (1.85M samples)")
print("=" * 70 + "\n")

from app.pipeline.train_kathak_model import train_kathak_model
train_kathak_model()

print("\nTraining complete!")
