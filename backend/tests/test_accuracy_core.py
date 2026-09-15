"""
Pytest Test Suite for Phase 1 Accuracy Core
Tests:
1. 3D Procrustes Normalization scale, translation, and rotation invariance
2. 3D Kinematic Joint Angle calculations (Araimandi 90° knee bend test)
3. Soft-DTW temporal alignment and warp path generation
4. DanceScoringEngine output validity for both Western and Indian Classical profiles
"""

import pytest
import numpy as np
from app.pipeline.procrustes_normalizer import ProcrustesNormalizer
from app.pipeline.joint_analytics import JointAnalyticsEngine
from app.pipeline.dtw_aligner import DTWAlignerEngine
from app.pipeline.scoring_engine import DanceScoringEngine
from app.core.genre_configs import get_genre_profile

def test_procrustes_invariance():
    # Generate random 3D points
    np.random.seed(42)
    pts1 = np.random.rand(33, 3) * 10.0
    
    # Translate, scale, and rotate pts1 to create pts2
    translation = np.array([5.0, -3.0, 12.0])
    scale = 2.5
    theta = np.radians(45)
    rotation_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    pts2 = (pts1 @ rotation_z.T) * scale + translation
    
    # Run Procrustes 3D Alignment
    disparity, norm1, norm2 = ProcrustesNormalizer.align_poses_3d(pts1, pts2)
    
    # Disparity should be nearly 0 after Procrustes normalization
    assert disparity < 1e-4, f"Procrustes failed to achieve rotation/scale/translation invariance: disparity={disparity}"

def test_joint_angle_calculation():
    # Test perpendicular vectors yielding 90 degrees
    a = np.array([0.0, 1.0, 0.0]) # Hip
    b = np.array([0.0, 0.0, 0.0]) # Knee
    c = np.array([1.0, 0.0, 0.0]) # Ankle
    
    angle = JointAnalyticsEngine.calculate_triplet_angle(a, b, c)
    assert abs(angle - 90.0) < 1e-4, f"Expected 90 degrees, got {angle}"

def test_dtw_alignment():
    # Create reference sine wave and phase-shifted attempt sine wave
    t = np.linspace(0, 2 * np.pi, 100)
    ref_signal = np.sin(t)[:, np.newaxis]
    att_signal = np.sin(t - 0.5)[:, np.newaxis] # Phase lag
    
    dtw_dist, path, warped_att = DTWAlignerEngine.align_joint_trajectories(ref_signal, att_signal)
    
    assert len(path) >= 100
    assert dtw_dist >= 0.0
    assert warped_att.shape == ref_signal.shape

def test_scoring_engine_profiles():
    np.random.seed(42)
    T = 50
    # Dummy pose sequence shape (T, 33, 3)
    ref_seq = np.random.rand(T, 33, 3)
    att_seq = ref_seq + np.random.normal(0, 0.05, size=(T, 33, 3))
    
    res_western = DanceScoringEngine.evaluate_performance(ref_seq, att_seq, genre_key="western_freestyle")
    res_indian = DanceScoringEngine.evaluate_performance(ref_seq, att_seq, genre_key="indian_classical")
    
    assert 0.0 <= res_western.overall_score <= 100.0
    assert 0.0 <= res_indian.overall_score <= 100.0
    assert res_western.genre_applied == "Western Freestyle / Pop / Hip-Hop"
    assert res_indian.genre_applied == "Indian Classical (Bharatanatyam / Kathak)"
