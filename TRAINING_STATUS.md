# 🩰 Kathak AI Model Training — Live Dashboard

> **Real-Time Automated Training Status Tracker**  
> *Last Updated:* **2026-09-15 21:44:00 (Local Time)**

---

## 📊 Quick Status Overview

| Metric | Current Value |
| :--- | :--- |
| **Status** | 🟢 **ACTIVE / TRAINING IN PROGRESS** |
| **Model Type** | PyTorch 2-Layer Kathak LSTM |
| **Feature Space** | 50D Normalized Biomechanical Distance Features |
| **Temporal Window** | 30 Frames per Movement Sequence |
| **Dataset Volume** | **1,855,392 Total Samples** (~10.59 GB) |
| **Classes** | **40 Kathak Movement & Technique Classes** |
| **Train / Val Split** | 1,484,313 Train (80%) / 371,079 Val (20%) |
| **Current Epoch** | **Epoch 01 / 40** |
| **Batch Progress** | **4,750 / 11,597 batches (41.0%)** |
| **Running Loss** | `3.0506` (declining steadily from initial `3.1415`) |
| **Running Accuracy** | `15.91%` (well above random baseline of 2.5%) |
| **Elapsed Time** | `16m 33s` |
| **Estimated Epoch ETA** | `~23m 52s` |
| **Learning Rate** | `0.002000` (Adam + ReduceLROnPlateau) |

---

## 📈 Real-Time Progress Bar

`[████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 41.0% (Epoch 1/40)`

---

## 📜 Recent Batch Telemetry

| Epoch | Batch | % Complete | Running Loss | Running Acc | Elapsed | ETA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 250 / 11,597 | 2.2% | 3.1415 | 14.86% | 0m 50s | ~37m 54s |
| 1 | 500 / 11,597 | 4.3% | 3.0984 | 15.33% | 1m 35s | ~35m 18s |
| 1 | 1,000 / 11,597 | 8.6% | 3.0780 | 15.60% | 3m 22s | ~35m 45s |
| 1 | 2,000 / 11,597 | 17.2% | 3.0615 | 15.86% | 7m 21s | ~35m 19s |
| 1 | 3,000 / 11,597 | 25.9% | 3.0556 | 15.90% | 10m 48s | ~30m 58s |
| 1 | 4,000 / 11,597 | 34.5% | 3.0518 | 15.91% | 13m 57s | ~26m 31s |
| 1 | 4,500 / 11,597 | 38.8% | 3.0515 | 15.90% | 15m 33s | ~24m 32s |
| 1 | **4,750 / 11,597** | **41.0%** | **3.0506** | **15.91%** | **16m 33s** | **~23m 52s** |

---

## 💾 Checkpoint & Artifacts

- **Model Destination**: `d:\aidance\kathak_dataset\kathak_lstm_trained_model.pt`
- **Class Mappings**: `d:\aidance\kathak_dataset\model_classes.json`
- **GitHub Sync**: Clean and synchronized with `origin/main`

*This file will be updated dynamically as training progresses.*
