# 🩰 Kathak Dance AI — Kaggle Dataset & Model Training Guide

Welcome! This folder contains everything required to upload the **Kathak Dance AI Pose Feature Dataset** to Kaggle and train the **BiLSTM + Self-Attention** model using Kaggle GPUs.

---

## 📁 Folder Contents

- `dataset-metadata.json`: Kaggle Dataset metadata specification file.
- `model_classes.json`: Dictionary mapping all **40 Kathak dance classes** (0 to 39).
- `dataset_summary.json`: Detailed specifications (1,855,392 samples, sequence length 30, 50D feature space).
- `package_dataset_zip.py`: Helper script to compress `.npy` samples into fast-upload `.zip` archives.
- `kathak_kaggle_training.ipynb`: Complete, ready-to-run Jupyter notebook for Kaggle.

---

## 🚀 Step 1: Prepare the Dataset for Kaggle Upload

### Option A: Package into `.zip` Archive (Recommended for Kaggle Upload)
Run the packaging script from your terminal:

```bash
python kathak_kaggle_dataset/package_dataset_zip.py
```

This creates `kathak_processed_samples.zip` inside this folder containing the `.npy` sample files.

---

## 📤 Step 2: Upload to Kaggle

### Method 1: Upload via Kaggle Web UI (Easy)
1. Log in to [Kaggle](https://www.kaggle.com).
2. Click **Create** `+` -> **New Dataset**.
3. Drag & drop the `kathak_kaggle_dataset` folder (or `kathak_processed_samples.zip`, `dataset-metadata.json`, and `model_classes.json`).
4. Enter dataset title: `Kathak Dance AI BiLSTM Model Dataset`.
5. Click **Create**.

### Method 2: Upload via Kaggle CLI
If you have the Kaggle CLI installed (`pip install kaggle`):

```bash
kaggle datasets create -p kathak_kaggle_dataset
```

---

## ⚡ Step 3: Run Training on Kaggle Notebook

1. Go to Kaggle -> **Create** `+` -> **New Notebook**.
2. Click **File** -> **Import Notebook** and select `kathak_kaggle_training.ipynb`.
3. In the right panel, click **+ Add Data** and search for your uploaded dataset (`Kathak Dance AI BiLSTM Model Dataset`).
4. Under **Notebook Settings**, enable **GPU Accelerator** (GPU T4 x2 or P100/V100).
5. Click **Run All**!

---

## 📊 Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Model** | PyTorch BiLSTM + Self-Attention |
| **Input Shape** | `(Batch, 30 frames, 50 features)` |
| **Classes** | 40 Kathak Movement & Technique Classes |
| **Precision** | PyTorch Mixed Precision (AMP `fp16`) |
| **Optimizer** | AdamW with Cosine Annealing Restarts |
| **Checkpoint** | `kathak_lstm_trained_model.pt` & `model_classes.json` |

---
