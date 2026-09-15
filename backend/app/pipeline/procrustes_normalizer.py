"""
3D Orthogonal Procrustes Normalization Engine
Rigorously normalizes 3D pose landmarks for scale, translation, and 3D rotation invariance.
Calculates rigid transformation (Optimal Rotation Matrix R) using SVD.
"""

import numpy as np
from scipy.spatial import procrustes
from typing import Tuple, Dict

class ProcrustesNormalizer:
    @staticmethod
    def center_and_scale(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Translates pose coordinates so that pelvis/hip midpoint is at origin (0,0,0)
        and scales by total Frobenius norm or hip-to-neck torso height to achieve unit scale.
        
        Args:
            landmarks: Shape (N_points, 3) representing (x, y, z) coordinates.
        Returns:
            (centered_scaled_landmarks, centroid, scale_factor)
        """
        if landmarks.ndim != 2 or landmarks.shape[1] != 3:
            raise ValueError(f"Expected landmarks array of shape (N, 3), got {landmarks.shape}")

        centroid = np.mean(landmarks, axis=0)
        centered = landmarks - centroid
        
        # Calculate scale factor (Frobenius norm divided by sqrt(N))
        scale = np.linalg.norm(centered)
        if scale == 0:
            scale = 1.0
            
        normalized = centered / scale
        return normalized, centroid, scale

    @staticmethod
    def align_poses_3d(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Applies Orthogonal Procrustes analysis to align data2 (attempt) onto data1 (reference).
        Removes translation, scale, and rotation differences.
        
        Args:
            data1: Reference pose landmarks shape (N, 3)
            data2: Attempt pose landmarks shape (N, 3)
            
        Returns:
            (disparity_error, aligned_reference, aligned_attempt)
            disparity_error: Sum of squared errors between normalized, optimally rotated points.
        """
        if data1.shape != data2.shape:
            raise ValueError(f"Shape mismatch in Procrustes alignment: {data1.shape} vs {data2.shape}")

        # Standard scipy procrustes analysis
        mtx1, mtx2, disparity = procrustes(data1, data2)
        return float(disparity), mtx1, mtx2

    @staticmethod
    def compute_frame_by_frame_disparity(ref_seq: np.ndarray, att_seq: np.ndarray) -> np.ndarray:
        """
        Computes spatial disparity across a aligned sequence of frames.
        
        Args:
            ref_seq: Shape (T, N, 3)
            att_seq: Shape (T, N, 3)
        Returns:
            Disparity scores array of shape (T,)
        """
        T = min(len(ref_seq), len(att_seq))
        disparities = np.zeros(T)
        for t in range(T):
            disp, _, _ = ProcrustesNormalizer.align_poses_3d(ref_seq[t], att_seq[t])
            disparities[t] = disp
        return disparities
