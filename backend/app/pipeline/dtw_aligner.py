"""
Soft-DTW & FastDTW Temporal Sequence Alignment Engine
Aligns attempt dance recording trajectory with reference routine trajectory.
Separates tempo drift (timing lag/lead) from true spatial/posture execution error.
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from typing import Tuple, List, Dict, Any

class DTWAlignerEngine:
    @staticmethod
    def align_joint_trajectories(
        ref_matrix: np.ndarray, att_matrix: np.ndarray
    ) -> Tuple[float, List[Tuple[int, int]], np.ndarray]:
        """
        Performs Dynamic Time Warping alignment between reference and attempt joint trajectories.
        
        Args:
            ref_matrix: Shape (T_ref, D) where D is number of joint feature channels
            att_matrix: Shape (T_att, D)
            
        Returns:
            (dtw_distance, warp_path, warped_attempt_matrix)
            warp_path: List of (index_ref, index_att) matching pairs
        """
        # Ensure 2D shape (T, D)
        if ref_matrix.ndim == 1:
            ref_matrix = ref_matrix[:, np.newaxis]
        if att_matrix.ndim == 1:
            att_matrix = att_matrix[:, np.newaxis]

        # Compute FastDTW path using Euclidean distance
        distance, path = fastdtw(ref_matrix, att_matrix, dist=euclidean)

        # Build warped attempt matrix aligned to reference timeline length T_ref
        T_ref = len(ref_matrix)
        D = ref_matrix.shape[1]
        warped_attempt = np.zeros((T_ref, D))
        
        # Aggregate matched attempt frames for each reference frame
        ref_to_att_map: Dict[int, List[int]] = {}
        for r_idx, a_idx in path:
            ref_to_att_map.setdefault(r_idx, []).append(a_idx)

        for r_idx in range(T_ref):
            if r_idx in ref_to_att_map:
                att_indices = ref_to_att_map[r_idx]
                warped_attempt[r_idx] = np.mean(att_matrix[att_indices], axis=0)
            else:
                # Fallback to nearest previous frame
                warped_attempt[r_idx] = att_matrix[min(r_idx, len(att_matrix) - 1)]

        return float(distance), path, warped_attempt

    @staticmethod
    def calculate_temporal_phase_lag(path: List[Tuple[int, int]], fps: float = 30.0) -> np.ndarray:
        """
        Calculates time lag (in seconds) between reference and attempt across the performance timeline.
        Positive lag = attempt is behind reference (too slow)
        Negative lag = attempt is ahead of reference (too fast)
        """
        if not path:
            return np.array([])
            
        ref_indices = np.array([p[0] for p in path])
        att_indices = np.array([p[1] for p in path])
        
        # Lag in frames and seconds
        frame_lags = att_indices - ref_indices
        time_lags_sec = frame_lags / fps
        return time_lags_sec
