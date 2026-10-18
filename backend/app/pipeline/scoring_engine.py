"""
Genre-Aware Accuracy Scoring & Feedback Engine (Integrated with Nitrya-AI Architecture)
Combines Procrustes 3D posture match, 3D Kinematic Joint Angles, Soft-DTW temporal alignment,
Beat Alignment Scoring (BAS), and Quality Confidence Gating.
Automatically identifies the worst-deviating keyframe to feed into the Qwen Visual Pose Corrector.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from app.core.genre_configs import GenreProfile, get_genre_profile
from app.pipeline.procrustes_normalizer import ProcrustesNormalizer
from app.pipeline.joint_analytics import JointAnalyticsEngine
from app.pipeline.dtw_aligner import DTWAlignerEngine
from app.pipeline.quality_gate import pose_quality_report
from app.pipeline.beat_tracker import extract_kinematic_beats, compute_beat_alignment_score


class PoseAnalysisResult:
    def __init__(
        self,
        overall_score: float,
        spatial_score: float,
        joint_score: float,
        rhythm_score: float,
        genre_applied: str,
        feedback_cues: List[Dict[str, Any]],
        joint_angle_deviations: Dict[str, float],
        beat_alignment: Optional[Dict[str, Any]] = None,
        quality_gate: Optional[Dict[str, Any]] = None,
        worst_keyframe: Optional[Dict[str, Any]] = None
    ):
        self.overall_score = round(overall_score, 2)
        self.spatial_score = round(spatial_score, 2)
        self.joint_score = round(joint_score, 2)
        self.rhythm_score = round(rhythm_score, 2)
        self.genre_applied = genre_applied
        self.feedback_cues = feedback_cues
        self.joint_angle_deviations = {k: round(v, 2) for k, v in joint_angle_deviations.items()}
        self.beat_alignment = beat_alignment or {}
        self.quality_gate = quality_gate or {}
        self.worst_keyframe = worst_keyframe or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "spatial_score": self.spatial_score,
            "joint_score": self.joint_score,
            "rhythm_score": self.rhythm_score,
            "genre_applied": self.genre_applied,
            "feedback_cues": self.feedback_cues,
            "joint_angle_deviations": self.joint_angle_deviations,
            "beat_alignment": self.beat_alignment,
            "quality_gate": self.quality_gate,
            "worst_keyframe": self.worst_keyframe
        }


class DanceScoringEngine:
    @classmethod
    def evaluate_performance(
        cls,
        ref_sequence: np.ndarray,
        att_sequence: np.ndarray,
        genre_key: str = "western_freestyle",
        fps: float = 30.0,
        musical_beats: Optional[np.ndarray] = None
    ) -> PoseAnalysisResult:
        """
        Evaluates attempt 3D pose sequence against reference 3D pose sequence.
        
        Args:
            ref_sequence: Shape (T_ref, 33, 3) reference 3D coordinates
            att_sequence: Shape (T_att, 33, 3) attempt 3D coordinates
            genre_key: 'western_freestyle', 'indian_classical', 'kathak', etc.
            fps: Video frames per second
            musical_beats: Optional musical beat timestamps for BAS scoring
        """
        profile: GenreProfile = get_genre_profile(genre_key)

        # 1. Pose Quality Confidence Gating
        quality_report = pose_quality_report([lm for lm in att_sequence])

        # 2. Compute 3D Joint Trajectories for both sequences
        ref_angles_dict = JointAnalyticsEngine.compute_sequence_joint_angles(ref_sequence)
        att_angles_dict = JointAnalyticsEngine.compute_sequence_joint_angles(att_sequence)

        joint_names = profile.key_joints if profile.key_joints else list(ref_angles_dict.keys())
        ref_matrix = np.column_stack([ref_angles_dict[j] for j in joint_names if j in ref_angles_dict])
        att_matrix = np.column_stack([att_angles_dict[j] for j in joint_names if j in att_angles_dict])

        # 3. Soft-DTW Temporal Alignment
        dtw_dist, path, warped_att_matrix = DTWAlignerEngine.align_joint_trajectories(ref_matrix, att_matrix)
        time_lags = DTWAlignerEngine.calculate_temporal_phase_lag(path, fps=fps)

        mean_dtw_err = dtw_dist / (len(ref_matrix) + 1e-8)
        dtw_rhythm_score = float(np.clip(100.0 * np.exp(-0.05 * mean_dtw_err), 0.0, 100.0))

        # 4. Beat Alignment Score (BAS from Nitrya-AI)
        bas_data = {}
        if musical_beats is not None and len(musical_beats) > 0:
            kinematic_beats = extract_kinematic_beats([lm for lm in att_sequence], fps=fps)
            bas_data = compute_beat_alignment_score(musical_beats, kinematic_beats, tolerance_seconds=0.15)
            # Combine DTW tempo tracking with physical beat hits
            rhythm_score = 0.5 * dtw_rhythm_score + 0.5 * bas_data["rhythm_score_pct"]
        else:
            rhythm_score = dtw_rhythm_score

        # 5. Kinematic Joint Angles Match & Deviations
        angular_diffs = np.abs(ref_matrix - warped_att_matrix)
        mean_joint_deviations = {
            joint_names[idx]: float(np.mean(angular_diffs[:, idx]))
            for idx in range(len(joint_names))
        }

        joint_errors = np.array(list(mean_joint_deviations.values()))
        exceeded_errors = np.maximum(0.0, joint_errors - profile.joint_tolerance_degrees)
        joint_score = float(np.clip(100.0 - (np.mean(exceeded_errors) * 3.0), 0.0, 100.0))

        # 6. Procrustes Spatial Pose Alignment
        spatial_disparities = []
        for r_idx, a_idx in path:
            if a_idx < len(att_sequence) and r_idx < len(ref_sequence):
                disp, _, _ = ProcrustesNormalizer.align_poses_3d(ref_sequence[r_idx], att_sequence[a_idx])
                spatial_disparities.append(disp)

        mean_spatial_disp = np.mean(spatial_disparities) if spatial_disparities else 0.5
        spatial_score = float(np.clip(100.0 * (1.0 - mean_spatial_disp * profile.keyframe_strictness), 0.0, 100.0))

        # 7. Weighted Overall Score
        overall_score = (
            profile.weight_form_spatial * spatial_score +
            profile.weight_joint_angles * joint_score +
            profile.weight_temporal_rhythm * rhythm_score
        )

        # 8. Actionable Feedback Cues
        feedback_cues = cls._generate_feedback_cues(
            angular_diffs, joint_names, time_lags, profile, fps
        )

        # 9. Identify Worst-Deviating Keyframe (for Qwen Visual Correction Handoff)
        worst_keyframe = cls._find_worst_keyframe(angular_diffs, joint_names, path, fps, profile)

        return PoseAnalysisResult(
            overall_score=overall_score,
            spatial_score=spatial_score,
            joint_score=joint_score,
            rhythm_score=rhythm_score,
            genre_applied=profile.name,
            feedback_cues=feedback_cues,
            joint_angle_deviations=mean_joint_deviations,
            beat_alignment=bas_data,
            quality_gate=quality_report,
            worst_keyframe=worst_keyframe
        )

    @staticmethod
    def _find_worst_keyframe(
        angular_diffs: np.ndarray,
        joint_names: List[str],
        path: List[tuple],
        fps: float,
        profile: GenreProfile
    ) -> Dict[str, Any]:
        """Finds the single frame with the highest anatomical joint error for Qwen correction."""
        if len(angular_diffs) == 0:
            return {}

        frame_errors = np.mean(angular_diffs, axis=1)
        worst_idx = int(np.argmax(frame_errors))
        worst_err = float(frame_errors[worst_idx])

        # Get matched user frame index
        matched_att_frame = path[worst_idx][1] if worst_idx < len(path) else worst_idx
        worst_joint_idx = int(np.argmax(angular_diffs[worst_idx]))
        worst_joint = joint_names[worst_joint_idx]
        joint_err = float(angular_diffs[worst_idx, worst_joint_idx])

        timestamp = round(matched_att_frame / max(fps, 1.0), 2)

        prompt_suggestion = (
            f"Modify the dancer at timestamp {timestamp}s: correct the {worst_joint.replace('_', ' ')} "
            f"by {joint_err:.1f} degrees into proper {profile.name} alignment. Keep identity and costume intact."
        )

        return {
            "attempt_frame_index": int(matched_att_frame),
            "timestamp_seconds": timestamp,
            "worst_joint": worst_joint,
            "angular_error_degrees": round(joint_err, 1),
            "overall_frame_error": round(worst_err, 1),
            "suggested_qwen_prompt": prompt_suggestion
        }

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
                readable = joint_name.replace("_", " ").title()
                if "knee" in joint_name and "indian" in profile.name.lower():
                    msg = f"Bend your knees deeper for Aramandi posture ({round(max_err, 1)}° off)."
                elif "elbow" in joint_name:
                    msg = f"Adjust {readable} extension to match reference line ({round(max_err, 1)}° off)."
                elif "hip" in joint_name or "spine" in joint_name:
                    msg = f"Keep your core aligned and spine erect ({round(max_err, 1)}° off)."
                else:
                    msg = f"Posture deviation in {readable} ({round(max_err, 1)}° off)."

                cues.append({
                    "timestamp": timestamp_sec,
                    "type": cue_type,
                    "joint": joint_name,
                    "error_degrees": round(max_err, 1),
                    "message": msg
                })

        return cues[:5]
