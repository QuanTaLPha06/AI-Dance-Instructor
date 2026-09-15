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
| **Batch Progress** | **4,000 / 11,597 batches (34.5%)** |
| **Running Loss** | `3.0518` (declining steadily from initial `3.1415`) |
| **Running Accuracy** | `15.91%` (well above random baseline of 2.5%) |
| **Elapsed Time** | `13m 57s` |
| **Estimated Epoch ETA** | `~26m 31s` |
| **Learning Rate** | `0.002000` (Adam + ReduceLROnPlateau) |

---

## 📈 Real-Time Progress Bar

`[██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░] 34.5% (Epoch 1/40)`

---

## 📜 Recent Batch Telemetry

| Epoch | Batch | % Complete | Running Loss | Running Acc | Elapsed | ETA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 250 / 11,597 | 2.2% | 3.1415 | 14.86% | 0m 50s | ~37m 54s |
| 1 | 500 / 11,597 | 4.3% | 3.0984 | 15.33% | 1m 35s | ~35m 18s |
| 1 | 1,000 / 11,597 | 8.6% | 3.0780 | 15.60% | 3m 22s | ~35m 45s |
| 1 | 1,500 / 11,597 | 12.9% | 3.0691 | 15.71% | 5m 12s | ~35m 05s |
| 1 | 2,000 / 11,597 | 17.2% | 3.0615 | 15.86% | 7m 21s | ~35m 19s |
| 1 | 2,500 / 11,597 | 21.6% | 3.0573 | 15.88% | 9m 11s | ~33m 26s |
| 1 | 3,000 / 11,597 | 25.9% | 3.0556 | 15.90% | 10m 48s | ~30m 58s |
| 1 | 3,500 / 11,597 | 30.2% | 3.0537 | 15.91% | 12m 31s | ~28m 59s |
| 1 | **4,000 / 11,597** | **34.5%** | **3.0518** | **15.91%** | **13m 57s** | **~26m 31s** |

---

## 💾 Checkpoint & Artifacts

- **Model Destination**: `d:\aidance\kathak_dataset\kathak_lstm_trained_model.pt`
- **Class Mappings**: `d:\aidance\kathak_dataset\model_classes.json`
- **GitHub Sync**: Clean and synchronized with `origin/main`

*This file will be updated dynamically as training progresses.*
