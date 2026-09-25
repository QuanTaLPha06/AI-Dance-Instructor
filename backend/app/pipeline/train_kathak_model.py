"""
Kathak AI Extreme Perfection Model Training Engine
GPU-accelerated with Mixed Precision (AMP fp16), 3-Layer Deep Residual BiLSTM + 4-Head Multi-Head Self-Attention,
Exponential Moving Average (EMA) weight tracking, and Biomechanical Pose Data Augmentation.
"""

import os
import sys
import glob
import json
import gc
import copy
import time
import zipfile
import random
from datetime import datetime

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.amp import GradScaler, autocast
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DATASET_DIR = r"d:\aidance\kathak_dataset"
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed_samples")
MODEL_SAVE_DIR = DATASET_DIR
STATUS_FILE = r"d:\aidance\TRAINING_STATUS.md"

# ─────────────────────────────────────────────────────────────
# 1. EXTREME PERFECTION CONFIGURATION
# ─────────────────────────────────────────────────────────────

class ExtremeConfig:
    NUM_EPOCHS = 80             # Total training epochs
    WARMUP_EPOCHS = 3           # Linear warmup epochs
    BATCH_SIZE = 1024 if HAS_TORCH and torch.cuda.is_available() else 128
    LEARNING_RATE = 0.0015      # Initial peak learning rate
    WEIGHT_DECAY = 1e-3         # AdamW weight decay
    LABEL_SMOOTHING = 0.05      # Label smoothing
    GRAD_CLIP_MAX_NORM = 1.0    # Gradient clipping max norm
    EARLY_STOP_PATIENCE = 15    # Patience for early stopping
    EMA_DECAY = 0.999           # Exponential Moving Average decay
    
    # Model Architecture
    INPUT_SIZE = 50             # 50D biomechanical pose feature space
    SEQUENCE_LENGTH = 30        # 30 frames per temporal sequence
    PROJECTION_DIM = 128        # Input feature projection dimension
    HIDDEN_SIZE = 256           # LSTM hidden units (512 bidirectional)
    NUM_LAYERS = 3              # 3-Layer Deep BiLSTM
    ATTENTION_HEADS = 4         # Multi-Head Attention heads
    LSTM_DROPOUT = 0.25         # Inter-layer LSTM dropout
    FC_DROPOUT_1 = 0.3          # First FC layer dropout
    FC_DROPOUT_2 = 0.2          # Second FC layer dropout
    
    # Biomechanical Data Augmentation
    AUGMENT_GAUSSIAN_STD = 0.005 # Joint coordinate jittering standard deviation
    AUGMENT_MASK_FRAMES = 3      # Max sequence frames to randomly mask out
    
    SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE: Deep Residual BiLSTM + Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────

if HAS_TORCH:
    class MultiHeadSelfAttentionPooling(nn.Module):
        """Multi-Head Self-Attention pooling over sequence time dimensions."""
        def __init__(self, embed_dim=512, num_heads=4):
            super().__init__()
            self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            self.layer_norm = nn.LayerNorm(embed_dim)
            self.query_proj = nn.Linear(embed_dim, 1, bias=False)

        def forward(self, x):
            # x: (batch, seq_len, embed_dim)
            attn_out, _ = self.mha(x, x, x)
            x = self.layer_norm(x + attn_out)
            
            # Soft attention pooling over sequence length
            attn_weights = F.softmax(self.query_proj(x), dim=1) # (batch, seq_len, 1)
            pooled = torch.sum(attn_weights * x, dim=1)         # (batch, embed_dim)
            return pooled

    class KathakDeepBiLSTMAttention(nn.Module):
        """
        State-of-the-Art Deep Residual BiLSTM with Multi-Head Self-Attention.
        """
        def __init__(self, input_size=50, projection_dim=128, hidden_size=256,
                     num_layers=3, num_classes=40, num_heads=4,
                     lstm_dropout=0.25, fc_dropout_1=0.3, fc_dropout_2=0.2):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU()
            )
            
            self.lstm = nn.LSTM(
                input_size=projection_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=lstm_dropout if num_layers > 1 else 0.0
            )
            
            self.attention = MultiHeadSelfAttentionPooling(embed_dim=hidden_size * 2, num_heads=num_heads)
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(fc_dropout_1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(fc_dropout_2),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            proj = self.input_proj(x)                  # (batch, 30, 128)
            lstm_out, _ = self.lstm(proj)               # (batch, 30, 512)
            pooled = self.attention(lstm_out)           # (batch, 512)
            logits = self.classifier(pooled)            # (batch, num_classes)
            return logits

    # Backward compatibility aliases
    KathakBiLSTMAttention = KathakDeepBiLSTMAttention
    PyTorchKathakLSTM = KathakDeepBiLSTMAttention

    class EMAHelper:
        """Exponential Moving Average (EMA) weight tracker for hyper-stable inference."""
        def __init__(self, model, decay=0.999):
            self.model = model
            self.decay = decay
            self.shadow = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

        def update(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

        def apply_shadow(self, target_model=None):
            m = target_model if target_model is not None else self.model
            for name, param in m.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data.copy_(self.shadow[name])

    class LazyAugmentedNpyDataset(Dataset):
        """Memory-efficient PyTorch Dataset loading .npy files with on-the-fly biomechanical pose augmentation."""
        def __init__(self, file_paths, labels, mean=None, std=None, is_train=True, config=ExtremeConfig):
            self.file_paths = file_paths
            self.labels = labels
            self.mean = mean
            self.std = std
            self.is_train = is_train
            self.config = config

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            data = np.load(self.file_paths[idx]).astype(np.float32)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            if self.is_train:
                if self.config.AUGMENT_GAUSSIAN_STD > 0:
                    noise = np.random.normal(0, self.config.AUGMENT_GAUSSIAN_STD, size=data.shape).astype(np.float32)
                    data += noise

                if self.config.AUGMENT_MASK_FRAMES > 0 and random.random() < 0.4:
                    num_masks = random.randint(1, self.config.AUGMENT_MASK_FRAMES)
                    mask_indices = random.sample(range(data.shape[0]), num_masks)
                    data[mask_indices, :] = 0.0

                scale = np.random.uniform(0.98, 1.02)
                data *= scale

            if self.mean is not None and self.std is not None:
                data = (data - self.mean) / self.std
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            return torch.from_numpy(data), self.labels[idx]

    LazyNpyDataset = LazyAugmentedNpyDataset

# ─────────────────────────────────────────────────────────────
# 3. STREAMING STATS & TRAINING DASHBOARD
# ─────────────────────────────────────────────────────────────

def compute_streaming_stats(file_paths, sample_size=100000):
    if not file_paths:
        return np.zeros((1, 50), dtype=np.float32), np.ones((1, 50), dtype=np.float32)

    sample_size = min(sample_size, len(file_paths))
    indices = random.sample(range(len(file_paths)), sample_size)
    running_sum = np.zeros(50, dtype=np.float64)
    running_sq_sum = np.zeros(50, dtype=np.float64)
    count = 0

    print(f"⚡ Computing streaming normalization stats over {sample_size:,} random samples...")
    for i, idx in enumerate(indices):
        try:
            data = np.load(file_paths[idx]).astype(np.float64)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            running_sum += data.sum(axis=0)
            running_sq_sum += (data ** 2).sum(axis=0)
            count += data.shape[0]
        except Exception:
            continue

    if count == 0:
        return np.zeros((1, 50), dtype=np.float32), np.ones((1, 50), dtype=np.float32)

    mean = running_sum / count
    var = np.maximum(running_sq_sum / count - mean ** 2, 1e-5)
    std = np.sqrt(var)
    std = np.where(std < 1e-3, 1.0, std)
    
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    
    return mean.astype(np.float32).reshape(1, 50), std.astype(np.float32).reshape(1, 50)


def run_training_pipeline():
    if not HAS_TORCH:
        print("❌ Error: PyTorch is required to run the training pipeline.")
        return

    set_seed(ExtremeConfig.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Kathak Extreme Perfection Training Pipeline on {device}...")

    # Discover sample files
    sample_files = [os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.endswith(".npy")] if os.path.exists(PROCESSED_DIR) else []
    if not sample_files:
        print(f"❌ Error: No `.npy` sample files found in {PROCESSED_DIR}")
        return

    print(f"📊 Found {len(sample_files):,} sample files.")

    labels = sorted(list(set([os.path.basename(f).split("_sample_")[0] for f in sample_files])))
    label_to_id = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    all_labels = [label_to_id[os.path.basename(f).split("_sample_")[0]] for f in sample_files]
    combined = list(zip(sample_files, all_labels))
    random.shuffle(combined)

    split_idx = int(len(combined) * 0.8)
    train_items = combined[:split_idx]
    val_items = combined[split_idx:]

    train_paths = [p for p, _ in train_items]
    train_labels = torch.tensor([l for _, l in train_items], dtype=torch.long)
    val_paths = [p for p, _ in val_items]
    val_labels = torch.tensor([l for _, l in val_items], dtype=torch.long)

    mean, std = compute_streaming_stats(train_paths, sample_size=100000)

    num_workers = 2 if torch.cuda.is_available() else 0
    pin_mem = torch.cuda.is_available()

    train_dataset = LazyAugmentedNpyDataset(train_paths, train_labels, mean=mean, std=std, is_train=True, config=ExtremeConfig)
    val_dataset = LazyAugmentedNpyDataset(val_paths, val_labels, mean=mean, std=std, is_train=False, config=ExtremeConfig)

    train_loader = DataLoader(train_dataset, batch_size=ExtremeConfig.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=ExtremeConfig.BATCH_SIZE * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    model = KathakDeepBiLSTMAttention(num_classes=num_classes).to(device)
    ema = EMAHelper(model, decay=ExtremeConfig.EMA_DECAY)

    criterion = nn.CrossEntropyLoss(label_smoothing=ExtremeConfig.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ExtremeConfig.LEARNING_RATE, weight_decay=ExtremeConfig.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    best_ema_acc = 0.0
    BEST_PATH = os.path.join(MODEL_SAVE_DIR, "kathak_model_best.pt")
    EMA_PATH = os.path.join(MODEL_SAVE_DIR, "kathak_model_ema.pt")
    META_PATH = os.path.join(MODEL_SAVE_DIR, "model_classes.json")

    print("\n" + "=" * 80)
    print(f"🔥 STARTING LOCAL EXTREME TRAINING FOR {ExtremeConfig.NUM_EPOCHS} EPOCHS")
    print("=" * 80)

    for epoch in range(1, ExtremeConfig.NUM_EPOCHS + 1):
        epoch_start = time.time()

        if epoch <= ExtremeConfig.WARMUP_EPOCHS:
            warmup_lr = ExtremeConfig.LEARNING_RATE * (epoch / ExtremeConfig.WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        current_lr = optimizer.param_groups[0]['lr']

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), ExtremeConfig.GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            ema.update()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        train_acc = (train_correct / max(train_total, 1)) * 100.0

        # Val Main
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
        val_acc = (val_correct / max(val_total, 1)) * 100.0

        # Val EMA
        ema_model = copy.deepcopy(model)
        ema.apply_shadow(ema_model)
        ema_model.eval()
        ema_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = ema_model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                ema_correct += (preds == batch_y).sum().item()
        ema_acc = (ema_correct / max(val_total, 1)) * 100.0
        del ema_model

        if epoch > ExtremeConfig.WARMUP_EPOCHS:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"📍 [Epoch {epoch:02d}/{ExtremeConfig.NUM_EPOCHS:02d}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"EMA Acc: {ema_acc:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_PATH)

        if ema_acc > best_ema_acc:
            best_ema_acc = ema_acc
            torch.save(ema.shadow, EMA_PATH)

    # Save Class Metadata
    class_meta = {"label_to_id": label_to_id, "num_classes": num_classes}
    with open(META_PATH, "w") as f:
        json.dump(class_meta, f, indent=2)

    print("\n" + "=" * 80)
    print(f"🎉 LOCAL TRAINING COMPLETE | Best Main Acc: {best_val_acc:.2f}% | Best EMA Acc: {best_ema_acc:.2f}%")
    print(f"📁 Saved Checkpoints: {BEST_PATH} & {EMA_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    run_training_pipeline()
