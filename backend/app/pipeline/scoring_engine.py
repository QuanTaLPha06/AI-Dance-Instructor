"""
Genre-Aware Accuracy Scoring & Feedback Engine
Combines Procrustes 3D posture match, 3D Kinematic Joint Angles, and Soft-DTW temporal alignment.
Generates overall accuracy scores (0-100%) and timestamped actionable coaching cues.
"""

import numpy as np
from typing import Dict, Any, List
from app.core.genre_configs import GenreProfile, get_genre_profile
from app.pipeline.procrustes_normalizer import ProcrustesNormalizer
from app.pipeline.joint_analytics import JointAnalyticsEngine
from app.pipeline.dtw_aligner import DTWAlignerEngine

class PoseAnalysisResult:
    def __init__(
        self,
        overall_score: float,
        spatial_score: float,
        joint_score: float,
        rhythm_score: float,
        genre_applied: str,
        feedback_cues: List[Dict[str, Any]],
        joint_angle_deviations: Dict[str, float]
    ):
        self.overall_score = round(overall_score, 2)
        self.spatial_score = round(spatial_score, 2)
        self.joint_score = round(joint_score, 2)
        self.rhythm_score = round(rhythm_score, 2)
        self.genre_applied = genre_applied
        self.feedback_cues = feedback_cues
        self.joint_angle_deviations = {k: round(v, 2) for k, v in joint_angle_deviations.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "spatial_score": self.spatial_score,
            "joint_score": self.joint_score,
            "rhythm_score": self.rhythm_score,
            "genre_applied": self.genre_applied,
            "feedback_cues": self.feedback_cues,
            "joint_angle_deviations": self.joint_angle_deviations
        }

class DanceScoringEngine:
    @classmethod
    def evaluate_performance(
        cls,
        ref_sequence: np.ndarray,
        att_sequence: np.ndarray,
        genre_key: str = "western_freestyle",
        fps: float = 30.0
    ) -> PoseAnalysisResult:
        """
        Evaluates attempt 3D pose sequence against reference 3D pose sequence.
        
        Args:
            ref_sequence: Shape (T_ref, 33, 3) reference 3D coordinates
            att_sequence: Shape (T_att, 33, 3) attempt 3D coordinates
            genre_key: 'western_freestyle' or 'indian_classical'
            fps: Video frames per second
        """
        profile: GenreProfile = get_genre_profile(genre_key)
        
        # 1. Compute 3D Joint Trajectories for both sequences
        ref_angles_dict = JointAnalyticsEngine.compute_sequence_joint_angles(ref_sequence)
        att_angles_dict = JointAnalyticsEngine.compute_sequence_joint_angles(att_sequence)

        # Convert dict to feature array (T, N_joints)
        joint_names = profile.key_joints if profile.key_joints else list(ref_angles_dict.keys())
        ref_matrix = np.column_stack([ref_angles_dict[j] for j in joint_names if j in ref_angles_dict])
        att_matrix = np.column_stack([att_angles_dict[j] for j in joint_names if j in att_angles_dict])

        # 2. Soft-DTW Temporal Alignment
        dtw_dist, path, warped_att_matrix = DTWAlignerEngine.align_joint_trajectories(ref_matrix, att_matrix)
        time_lags = DTWAlignerEngine.calculate_temporal_phase_lag(path, fps=fps)

        # Rhythm / DTW score calculation (Exponential decay based on mean DTW error)
        mean_dtw_err = dtw_dist / (len(ref_matrix) + 1e-8)
        rhythm_score = float(np.clip(100.0 * np.exp(-0.05 * mean_dtw_err), 0.0, 100.0))

        # 3. Kinematic Joint Angles Match (mean absolute angular difference on warped timeline)
        angular_diffs = np.abs(ref_matrix - warped_att_matrix)
        mean_joint_deviations = {
            joint_names[idx]: float(np.mean(angular_diffs[:, idx]))
            for idx in range(len(joint_names))
        }

        # Joint score with genre tolerance threshold
        joint_errors = np.array(list(mean_joint_deviations.values()))
        exceeded_errors = np.maximum(0.0, joint_errors - profile.joint_tolerance_degrees)
        joint_score = float(np.clip(100.0 - (np.mean(exceeded_errors) * 3.0), 0.0, 100.0))

        # 4. Procrustes Spatial Pose Alignment across aligned frames
        T_ref = len(ref_sequence)
        spatial_disparities = []
        for r_idx, a_idx in path:
            if a_idx < len(att_sequence) and r_idx < len(ref_sequence):
                disp, _, _ = ProcrustesNormalizer.align_poses_3d(ref_sequence[r_idx], att_sequence[a_idx])
                spatial_disparities.append(disp)

        mean_spatial_disp = np.mean(spatial_disparities) if spatial_disparities else 0.5
        spatial_score = float(np.clip(100.0 * (1.0 - mean_spatial_disp * profile.keyframe_strictness), 0.0, 100.0))

        # 5. Weighted Overall Score
        overall_score = (
            profile.weight_form_spatial * spatial_score +
            profile.weight_joint_angles * joint_score +
            profile.weight_temporal_rhythm * rhythm_score
        )

        # 6. Generate Timestamped Actionable Coaching Feedback Cues
        feedback_cues = cls._generate_feedback_cues(
            angular_diffs, joint_names, time_lags, profile, fps
        )

        return PoseAnalysisResult(
            overall_score=overall_score,
            spatial_score=spatial_score,
            joint_score=joint_score,
            rhythm_score=rhythm_score,
            genre_applied=profile.name,
            feedback_cues=feedback_cues,
            joint_angle_deviations=mean_joint_deviations
        )

    @staticmethod
    def _generate_feedback_cues(
        angular_diffs: np.ndarray,
        joint_names: List[str],
        time_lags: np.ndarray,
        profile: GenreProfile,
        fps: float
    ) -> List[Dict[str, Any]]:
        cues = []
        T = len(angular_diffs)
        
        # Check every 1 second chunk (fps frames) for largest joint errors
        chunk_size = int(fps)
        for chunk_idx, start_f in enumerate(range(0, T, chunk_size)):
            end_f = min(start_f + chunk_size, T)
            chunk_diffs = angular_diffs[start_f:end_f]
            if len(chunk_diffs) == 0:
                continue

            max_err_joint_idx = int(np.argmax(np.mean(chunk_diffs, axis=0)))
            max_err = float(np.mean(chunk_diffs[:, max_err_joint_idx]))
            joint_name = joint_names[max_err_joint_idx]

            timestamp_sec = round(start_f / fps, 1)

            if max_err > profile.joint_tolerance_degrees:
                cue_type = "posture_correction"
                if "knee" in joint_name and "indian" in profile.name.lower():
                    msg = f"Bend your knees deeper for Araimandi posture ({round(max_err, 1)}° deviation)."
                elif "elbow" in joint_name:
                    msg = f"Adjust elbow angle to match reference extension ({round(max_err, 1)}° deviation)."
                elif "hip" in joint_name or "spine" in joint_name:
                    msg = f"Keep your core and torso upright ({round(max_err, 1)}° tilt)."
                else:
                    msg = f"Form deviation detected in {joint_name.replace('_', ' ')} ({round(max_err, 1)}° off)."

                cues.append({
                    "timestamp": timestamp_sec,
                    "type": cue_type,
                    "joint": joint_name,
                    "error_degrees": round(max_err, 1),
                    "message": msg
                })

        return cues[:5]  # Top 5 feedback cues
