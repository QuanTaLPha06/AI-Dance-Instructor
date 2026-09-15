"""
Kathak LSTM Model Training Script on Ingested Datasets
Memory-efficient: uses disk-backed lazy loading for large datasets.
"""

import os
import json
import numpy as np
import random
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

from app.pipeline.kathak_ai_engine import build_kathak_lstm_model

DATASET_DIR = r"d:\aidance\kathak_dataset"
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed_samples")
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "kathak_lstm_trained_model")

if HAS_TORCH:
    class PyTorchKathakLSTM(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
            self.fc1 = nn.Linear(64, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(32, num_classes)

        def forward(self, x):
            out, (h_n, _) = self.lstm(x)
            out = self.relu(self.fc1(out[:, -1, :]))
            out = self.dropout(out)
            out = self.fc2(out)
            return out

    class LazyNpyDataset(Dataset):
        """Memory-efficient dataset that loads .npy files from disk on demand."""
        def __init__(self, file_paths, labels, mean=None, std=None):
            self.file_paths = file_paths
            self.labels = labels
            self.mean = mean  # shape (1, 50) or None
            self.std = std    # shape (1, 50) or None

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            data = np.load(self.file_paths[idx]).astype(np.float32)  # (30, 50)
            if self.mean is not None and self.std is not None:
                data = (data - self.mean) / self.std
            return torch.from_numpy(data), self.labels[idx]
else:
    PyTorchKathakLSTM = None


def compute_streaming_stats(file_paths, sample_size=50000):
    """Compute mean/std from a random subset to avoid loading everything."""
    indices = random.sample(range(len(file_paths)), min(sample_size, len(file_paths)))
    running_sum = np.zeros(50, dtype=np.float64)
    running_sq_sum = np.zeros(50, dtype=np.float64)
    count = 0
    for idx in indices:
        data = np.load(file_paths[idx]).astype(np.float64)  # (30, 50)
        running_sum += data.sum(axis=0)
        running_sq_sum += (data ** 2).sum(axis=0)
        count += data.shape[0]
    mean = running_sum / count
    std = np.sqrt(running_sq_sum / count - mean ** 2) + 1e-6
    return mean.astype(np.float32).reshape(1, 50), std.astype(np.float32).reshape(1, 50)


def train_kathak_model():
    print(f"Loading processed dataset samples from {PROCESSED_DIR}...")
    sample_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".npy")]

    if not sample_files:
        print("No processed samples found in dataset directory!")
        return

    labels = list(set([f.split("_sample_")[0] for f in sample_files]))
    label_to_id = {label: i for i, label in enumerate(sorted(labels))}

    print(f"Found {len(sample_files)} total samples across {len(labels)} categories:")
    for l, i in label_to_id.items():
        print(f" - Class {i}: {l}")

    # Build file paths and label arrays
    all_paths = []
    all_labels = []
    for fname in sample_files:
        fpath = os.path.join(PROCESSED_DIR, fname)
        label_str = fname.split("_sample_")[0]
        all_paths.append(fpath)
        all_labels.append(label_to_id[label_str])

    # Shuffle and split into train/val (80/20)
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

    print(f"Dataset split: Train={len(train_paths)}, Validation={len(val_paths)}")

    if HAS_TF and not HAS_TORCH:
        print("\nTensorFlow training not supported for large datasets. Install PyTorch.")
    elif HAS_TORCH:
        print("\nComputing normalization statistics from sample subset...")
        mean, std = compute_streaming_stats(train_paths, sample_size=50000)
        print(f"Normalization computed (sampled 50K files). Mean range: [{mean.min():.4f}, {mean.max():.4f}]")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"\n🚀 CUDA ACCELERATION ENABLED:")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Dedicated VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            pin_mem = True
            batch_size = 512
        else:
            print("\n⚠️ CUDA not detected in PyTorch. Running on CPU.")
            pin_mem = False
            batch_size = 128

        print("\nTraining Kathak LSTM Model using PyTorch (disk-backed lazy loading)...")
        NUM_EPOCHS = 40
        BATCH_SIZE = batch_size

        train_dataset = LazyNpyDataset(train_paths, train_labels, mean=mean, std=std)
        val_dataset = LazyNpyDataset(val_paths, val_labels, mean=mean, std=std)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=pin_mem)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                                num_workers=0, pin_memory=pin_mem)

        model = PyTorchKathakLSTM(num_classes=len(labels)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        id_to_label = {v: k for k, v in label_to_id.items()}
        best_val_acc = 0.0
        total_batches = len(train_loader)

        print(f"\n{'='*80}")
        print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10} | {'LR':>10}")
        print(f"{'='*80}")

        import time

        for epoch in range(NUM_EPOCHS):
            # --- Training ---
            model.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            start_time = time.time()

            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_y.size(0)

                # Periodic batch-level progress update every 250 batches
                if (batch_idx + 1) % 250 == 0 or (batch_idx + 1) == total_batches:
                    elapsed = time.time() - start_time
                    batches_done = batch_idx + 1
                    pct = (batches_done / total_batches) * 100.0
                    rate = batches_done / max(elapsed, 0.01)
                    eta_sec = (total_batches - batches_done) / max(rate, 0.01)
                    current_loss = total_loss / batches_done
                    current_acc = (train_correct / train_total) * 100.0
                    print(f"  [Epoch {epoch+1:02d}/{NUM_EPOCHS} | Batch {batches_done:>5}/{total_batches} ({pct:>5.1f}%)] "
                          f"Loss: {current_loss:.4f} | Acc: {current_acc:.2f}% | "
                          f"Elapsed: {int(elapsed//60)}m{int(elapsed%60):02d}s | "
                          f"ETA: {int(eta_sec//60)}m{int(eta_sec%60):02d}s")

            avg_train_loss = total_loss / len(train_loader)
            train_acc = (train_correct / train_total) * 100.0

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)
            avg_val_loss = val_loss / max(len(val_loader), 1)
            val_acc = (val_correct / val_total) * 100.0

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)

            print(f"\n🏁 EPOCH {epoch+1} SUMMARY: "
                  f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}% | LR={current_lr:.6f}\n")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH + ".pt")
                print(f"  ⭐ Saved new best checkpoint ({val_acc:.2f}%) to {MODEL_SAVE_PATH}.pt")

        print(f"{'='*80}")
        print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Saved best PyTorch model to {MODEL_SAVE_PATH}.pt")

        # --- Per-class accuracy breakdown ---
        model.load_state_dict(torch.load(MODEL_SAVE_PATH + ".pt", weights_only=True, map_location=device))
        model.eval()
        all_preds = []
        all_labels_list = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels_list.extend(batch_y.numpy())
        all_preds = np.array(all_preds)
        all_labels_arr = np.array(all_labels_list)

        print(f"\n{'='*60}")
        print(f"  PER-CLASS ACCURACY BREAKDOWN (Validation Set)")
        print(f"{'='*60}")
        print(f"  {'Class':<35} {'Correct':>8} {'Total':>7} {'Acc':>8}")
        print(f"  {'-'*58}")
        for class_id in sorted(id_to_label.keys()):
            mask = all_labels_arr == class_id
            cls_total = mask.sum()
            if cls_total == 0:
                continue
            cls_correct = (all_preds[mask] == class_id).sum()
            cls_acc = (cls_correct / cls_total) * 100.0
            print(f"  {id_to_label[class_id]:<35} {cls_correct:>8} {cls_total:>7} {cls_acc:>7.2f}%")
        overall_correct = (all_preds == all_labels_arr).sum()
        print(f"  {'-'*58}")
        print(f"  {'OVERALL':<35} {overall_correct:>8} {len(all_labels_arr):>7} {(overall_correct/len(all_labels_arr))*100:>7.2f}%")
        print(f"{'='*60}\n")
    else:
        print("Neither TensorFlow nor PyTorch installed. Training simulated and metrics exported.")

    class_meta = {"label_to_id": label_to_id, "num_classes": len(labels)}
    with open(os.path.join(DATASET_DIR, "model_classes.json"), "w") as f:
        json.dump(class_meta, f, indent=2)

if __name__ == "__main__":
    train_kathak_model()
