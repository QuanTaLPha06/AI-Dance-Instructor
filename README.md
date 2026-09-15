# 🩰 AI Dance Instructor (Aidance)

> **AI-Powered Real-Time Dance Assessment, Kinematic Alignment & Multimodal Feedback System**

AI Dance Instructor is an advanced, full-stack AI platform designed to evaluate dance performances against reference demonstrations. It supports both **Western Freestyle / Pop** and traditional **Indian Classical (Kathak / Bharatanatyam)** dance styles using 3D pose extraction, Procrustes spatial normalization, Soft-DTW temporal alignment, joint kinematics, and audio rhythm synchronization.

---

## 📁 Repository Organization

Below is the repository structure outlining what is present and where:

```
.
├── backend/                               # FastAPI Accuracy Core Engine & REST API
│   ├── app/
│   │   ├── main.py                        # FastAPI entry point & API endpoints
│   │   ├── tasks.py                       # Background processing tasks
│   │   ├── core/
│   │   │   └── genre_configs.py          # Weighting profiles (Western vs. Indian Classical)
│   │   └── pipeline/
│   │       ├── pose_extractor.py          # MediaPipe / YOLO 3D landmark extraction
│   │       ├── procrustes_normalizer.py   # Spatial Procrustes 3D scale, rotation & translation invariance
│   │       ├── dtw_aligner.py             # Soft-DTW temporal trajectory alignment
│   │       ├── joint_analytics.py         # Kinematic joint angle calculation (e.g., Araimandi 90° bend)
│   │       ├── audio_analytics.py         # Librosa beat tracking & rhythm analytics
│   │       ├── audio_processor.py         # Audio extraction & spectral processing
│   │       ├── scoring_engine.py          # Multi-genre weighted scoring engine
│   │       ├── kathak_ai_engine.py        # Kathak BiLSTM sequence classification & mudra evaluation
│   │       ├── kathak_multimodal_extractor.py # Multimodal pose + hand + audio extraction
│   │       ├── download_and_build_kathak_dataset.py # YouTube video dataset pipeline script
│   │       ├── train_kathak_model.py      # LSTM/BiLSTM model training pipeline
│   │       └── wham_adapter.py            # WHAM 3D mesh adapter integration
│   ├── tests/                             # Pytest test suite
│   │   ├── test_accuracy_core.py          # 3D Procrustes, DTW, and Scoring tests
│   │   └── test_kathak_ai.py             # Kathak model and classification tests
│   └── requirements.txt                   # Python backend dependencies
│
├── frontend/                              # Next.js 16 Client PWA Web Application
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                   # Main dashboard, web camera recorder & score visualizer
│   │   │   ├── layout.tsx                 # App layout configuration
│   │   │   └── globals.css                # Styling & glassmorphism UI theme
│   │   └── hooks/
│   │       └── useMediaPipe.ts            # Client-side MediaPipe landmark detection hook
│   ├── package.json                       # Next.js frontend dependencies
│   └── tsconfig.json                      # TypeScript configuration
│
├── kathak_dataset/                        # Kathak Dataset & Trained AI Models
│   ├── kathak_bilstm_trained.pt           # PyTorch trained BiLSTM model weights
│   ├── kathak_lstm_trained_model.pt       # PyTorch trained LSTM baseline model weights
│   ├── model_classes.json                 # Class mappings (Tatkar, Hastak Mudras, CCRT, etc.)
│   └── processed_samples/                 # Extracted frame sequences & landmark features
│
├── yolo26n-pose.pt                        # YOLO pose detection model weights
├── pose_landmarker_lite.task              # MediaPipe Pose Landmarker task bundle
├── hand_landmarker.task                   # MediaPipe Hand Landmarker task bundle
└── README.md                              # Project documentation
```

---

## ⚡ Core Features & Capabilities

1. **3D Spatial Procrustes Normalization**: Achieves scale, rotation, and translation invariance to accurately compare dancers regardless of camera distance or angle.
2. **Soft-DTW Temporal Alignment**: Dynamically warps time series trajectories to compare movements performed at different tempos or speeds.
3. **Kinematic Joint Angle Analytics**: Monitors precise angular movements (such as knee bend angles for *Araimandi* and arm positions for *Hastak* mudras).
4. **Multimodal Audio Rhythm Sync**: Analyzes audio beat alignments using Librosa to calculate rhythm scores.
5. **Kathak AI Engine**: Trained LSTM/BiLSTM deep learning model to classify Kathak dance movements (*Tatkar*, *Hastak*, etc.).
6. **Real-time Client Interface**: Next.js 16 PWA interface with MediaPipe Web Vision API overlay for webcam feed recording and feedback.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ & npm

---

### 2. Backend Setup & Run

```bash
# Navigate to backend
cd backend

# Create virtual environment (if not created)
python -m venv ../venv

# Activate virtual environment
# Windows:
..\venv\Scripts\activate
# Linux/macOS:
source ../venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Pytest test suite to verify installation
python -m pytest

# Start FastAPI server
uvicorn app.main:app --reload --port 8000
```

FastAPI server runs at `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

---

### 3. Frontend Setup & Run

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Run Next.js development server
npm run dev
```

Frontend application runs at `http://localhost:3000`.

---

## 🧪 Testing

To execute full unit and integration test coverage:

```bash
cd backend
..\venv\Scripts\python.exe -m pytest
```

---

## 🤝 Remote Repository

- **GitHub Repository**: [QuanTaLPha06/AI-Dance-Instructor](https://github.com/QuanTaLPha06/AI-Dance-Instructor.git)
