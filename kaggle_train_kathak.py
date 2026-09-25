# Kaggle Kathak Model Training Pipeline
# This notebook handles dataset extraction/loading, model architecture setup, per-epoch loops with dynamic progress tracking, evaluation, and saving model checkpoints.

import os
import sys
import glob
import json
import gc
import time
import zipfile
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

# ==========================================
# 1. ENVIRONMENT & DATASET SETUP
# ==========================================
print("🚀 Initializing Kaggle Kathak Training Pipeline...")

# Autodetect Kaggle dataset directory or local directory
KAGGLE_INPUT_DIR = "/kaggle/input"
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
DATASET_DIR = os.path.join(WORKING_DIR, "kathak_dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed_samples")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Unzip or locate dataset if uploaded as zip on Kaggle
zip_files = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "**/*.zip"), recursive=True) + glob.glob("*.zip")
if zip_files:
    print(f"📦 Found dataset zip archive: {zip_files[0]}. Unzipping...")
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print("✅ Unzipped dataset successfully.")
else:
    # Look for existing processed_samples in input directory
    found_samples = glob.glob(os.path.join(KAGGLE_INPUT_DIR, "**/processed_samples/*.npy"), recursive=True)
    if found_samples:
        PROCESSED_DIR = os.path.dirname(found_samples[0])
        print(f"📁 Found existing dataset at: {PROCESSED_DIR}")

sample_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".npy")]
print(f"📊 Total `.npy` sample files available: {len(sample_files):,}")

# ==========================================
# 2. MODEL ARCHITECTURE (BiLSTM + Self-Attention)
# ==========================================
class SelfAttention(nn.Module):
    """Learned self-attention pooling over temporal LSTM outputs."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights.squeeze(-1)

class KathakBiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Self-Attention for Kathak dance classification.
    Input shape: (batch_size, 30_frames, 50_features)
    """
    def __init__(self, input_size=50, hidden_size=128, num_layers=2,
                 num_classes=40, dropout=0.3, fc_dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = SelfAttention(hidden_size * 2)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_w = self.attention(lstm_out)
        context = self.layer_norm(context)
        out = self.gelu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ==========================================
# 3. LAZY DATASET & STATS COMPUTATION
# ==========================================
class LazyNpyDataset(Dataset):
    def __init__(self, file_paths, labels, mean=None, std=None):
        self.file_paths = file_paths
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx]).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if self.mean is not None and self.std is not None:
            data = (data - self.mean) / self.std
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(data), self.labels[idx]

def compute_streaming_stats(file_paths, sample_size=100000):
    indices = random.sample(range(len(file_paths)), min(sample_size, len(file_paths)))
    running_sum = np.zeros(50, dtype=np.float64)
    running_sq_sum = np.zeros(50, dtype=np.float64)
    count = 0
    for i, idx in enumerate(indices):
        data = np.load(file_paths[idx]).astype(np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        running_sum += data.sum(axis=0)
        running_sq_sum += (data ** 2).sum(axis=0)
        count += data.shape[0]
    mean = running_sum / count
    var = np.maximum(running_sq_sum / count - mean ** 2, 1e-5)
    std = np.sqrt(var)
    std = np.where(std < 1e-3, 1.0, std)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    return mean.astype(np.float32).reshape(1, 50), std.astype(np.float32).reshape(1, 50)

# ==========================================
# 4. PREPARE DATASET SPLIT & DATALOADERS
# ==========================================
labels = sorted(list(set([f.split("_sample_")[0] for f in sample_files])))
label_to_id = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

all_paths = [os.path.join(PROCESSED_DIR, f) for f in sample_files]
all_labels = [label_to_id[f.split("_sample_")[0]] for f in sample_files]

combined = list(zip(all_paths, all_labels))
random.seed(42)
random.shuffle(combined)
split_idx = int(len(combined) * 0.8)

train_items = combined[:split_idx]
val_items = combined[split_idx:]

train_paths = [p for p, _ in train_items]
train_labels = torch.tensor([l for _, l in train_items], dtype=torch.long)
val_paths = [p for p, _ in val_items]
val_labels = torch.tensor([l for _, l in val_items], dtype=torch.long)

print(f"🔹 Dataset Split: {len(train_paths):,} Train | {len(val_paths):,} Validation ({num_classes} Classes)")

print("⚡ Computing normalization statistics...")
mean, std = compute_streaming_stats(train_paths, sample_size=min(100000, len(train_paths)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1024 if torch.cuda.is_available() else 128
num_workers = 2 if torch.cuda.is_available() else 0
pin_mem = torch.cuda.is_available()

train_dataset = LazyNpyDataset(train_paths, train_labels, mean=mean, std=std)
val_dataset = LazyNpyDataset(val_paths, val_labels, mean=mean, std=std)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

# ==========================================
# 5. TRAINING LOOP (Loop for Each Epoch)
# ==========================================
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1
GRAD_CLIP_MAX_NORM = 1.0
EARLY_STOP_PATIENCE = 10

model = KathakBiLSTMAttention(input_size=50, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

best_val_acc = 0.0
best_val_loss = float("inf")
epochs_no_improve = 0
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "kathak_lstm_trained_model.pt")

print("\n" + "=" * 80)
print(f"🔥 STARTING TRAINING LOOP FOR {NUM_EPOCHS} EPOCHS ON {device}")
print("=" * 80)

training_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    
    # --- TRAINING PHASE ---
    model.train()
    total_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == batch_y).sum().item()
        train_total += batch_y.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_acc = (train_correct / train_total) * 100.0

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)

    avg_val_loss = val_loss / max(len(val_loader), 1)
    val_acc = (val_correct / val_total) * 100.0
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    epoch_time = time.time() - epoch_start
    print(f"📍 [Epoch {epoch:02d}/{NUM_EPOCHS:02d}] "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

    # Save Best Model Checkpoint
    improved = False
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   🏆 Best model saved! New Val Acc: {val_acc:.2f}%")
        improved = True

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        improved = True

    if improved:
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"⚠️ Early stopping triggered at epoch {epoch}!")
            break

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Save Class Metadata JSON
class_meta = {"label_to_id": label_to_id, "num_classes": num_classes}
with open(os.path.join(WORKING_DIR, "model_classes.json"), "w") as f:
    json.dump(class_meta, f, indent=2)

print("\n" + "=" * 80)
print(f"🎉 TRAINING COMPLETE | Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"📁 Trained Model Saved to: {MODEL_SAVE_PATH}")
print(f"📁 Class Metadata Saved to: {os.path.join(WORKING_DIR, 'model_classes.json')}")
print("=" * 80)
