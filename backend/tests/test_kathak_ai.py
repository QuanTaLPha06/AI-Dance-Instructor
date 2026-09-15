"""
Tests for Kathak AI Evaluation and Multimodal Pipeline
"""

import pytest
import numpy as np
from app.pipeline.kathak_ai_engine import (
    KathakDataProcessor, 
    build_kathak_lstm_model, 
    KathakScorer
)
from app.core.genre_configs import get_genre_profile

def test_scale_normalization_and_feature_extraction():
    # Simulate 50 frames of 33 keypoints
    fake_landmarks = np.random.rand(50, 33, 3).astype(np.float32)
    
    # 1. Scale normalization
    norm_frame = KathakDataProcessor.normalize_frame(fake_landmarks[0])
    assert norm_frame.shape == (33, 3)
    
    # 2. Extract 50D feature matrix
    feats = KathakDataProcessor.extract_50d_features(fake_landmarks)
    assert feats.shape == (50, 50)
    
    # 3. Resample to 30 temporal frames
    windowed = KathakDataProcessor.resample_temporal_window(feats, target_frames=30)
    assert windowed.shape == (30, 50)
    
    # 4. Full LSTM input shape
    lstm_input = KathakDataProcessor.process_video_to_lstm_input(fake_landmarks)
    assert lstm_input.shape == (1, 30, 50)

def test_dynamic_time_warping_alignment():
    # Test DTW alignment between variable speed sequences
    seq_student = np.random.rand(45, 50).astype(np.float32)
    seq_expert = np.random.rand(30, 50).astype(np.float32)
    
    aligned_seq, distance = KathakDataProcessor.align_sequences_dtw(seq_student, seq_expert)
    assert aligned_seq.shape == (30, 50)
    assert isinstance(distance, float)

def test_kathak_lstm_model_building():
    model = build_kathak_lstm_model(num_classes=8)
    assert len(model.layers) == 5
    
    # Test forward pass with dummy input (batch_size=2, timesteps=30, features=50)
    dummy_x = np.random.randn(2, 30, 50).astype(np.float32)
    preds = model.predict(dummy_x, verbose=0)
    assert preds.shape == (2, 8)
    np.testing.assert_almost_equal(np.sum(preds, axis=-1), [1.0, 1.0], decimal=4)

def test_kathak_scoring_and_joint_angle_equations():
    # Confidence-to-Score mapping test
    score_low = KathakScorer.map_confidence_to_score(0.2)
    score_high = KathakScorer.map_confidence_to_score(0.95)
    assert score_low == 2.8
    assert score_high == 9.55 or score_high == 9.6 or abs(score_high - 9.55) < 0.1
    
    # Joint angle deviation evaluation
    expert_pose = np.ones((30, 33, 3)) * 0.5
    student_pose = np.ones((30, 33, 3)) * 0.5
    
    # Create deliberate deviation in left elbow (idx 13)
    student_pose[:, 13, 0] += 0.3
    
    corrections = KathakScorer.evaluate_joint_trajectories(student_pose, expert_pose)
    assert isinstance(corrections, list)
    if len(corrections) > 0:
        assert "joint" in corrections[0]
        assert "recommendation" in corrections[0]

def test_kathak_genre_profile():
    profile = get_genre_profile("kathak")
    assert profile.name.startswith("Kathak")
    assert profile.active_mudra_detection is True
    assert profile.joint_tolerance_degrees == 8.0

def test_audio_processor_and_librosa_integration():
    from app.pipeline.audio_processor import KathakAudioProcessor
    
    # Test rhythm synchronization logic
    pose_peaks = np.array([15, 45, 75, 105])  # Frame indices at 30 fps -> times 0.5s, 1.5s, 2.5s, 3.5s
    beat_times = np.array([0.5, 1.5, 2.5, 3.5])
    
    res = KathakAudioProcessor.calculate_rhythm_dance_synchronization(pose_peaks, beat_times, fps=30.0)
    assert res["rhythm_sync_score"] == 10.0
    assert res["mean_latency_ms"] == 0.0

def test_wham_3d_adapter_integration():
    from app.pipeline.wham_adapter import WHAMMotionAdapter
    adapter = WHAMMotionAdapter()
    
    fake_landmarks = np.random.rand(30, 33, 3).astype(np.float32)
    trajectory = adapter.extract_world_grounded_trajectory(fake_landmarks)
    assert trajectory["wham_enabled"] is True
    assert trajectory["world_grounded"] is True

