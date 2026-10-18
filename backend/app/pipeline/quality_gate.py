"""
Pose Quality & Confidence Gating Module (Adapted from Nitrya-AI)
Evaluates whether a video's detected pose sequence is of sufficient quality and completeness
before computing scores, preventing erroneous results caused by occlusions, bad lighting,
or dancers stepping out of frame.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def pose_quality_report(
    pose_sequence: List[Optional[np.ndarray]],
    min_valid_frame_ratio: float = 0.70
) -> Dict[str, Any]:
    """
    Evaluates pose sequence quality and returns diagnostic metrics.
    
    Args:
        pose_sequence: List of per-frame pose landmarks (Shape (N, 3) or None for missed frames)
        min_valid_frame_ratio: Minimum ratio of valid frames required to pass gate (default 70%)
    
    Returns:
        Dict with total_frames, valid_frames, valid_frame_ratio, and passes_confidence_gate.
    """
    total_frames = len(pose_sequence)
    if total_frames == 0:
        return {
            "total_frames": 0,
            "valid_frames": 0,
            "valid_frame_ratio": 0.0,
            "passes_confidence_gate": False,
            "message": "Empty video: No frames received for analysis."
        }

    valid_frames = sum(1 for frame in pose_sequence if frame is not None and np.any(frame != 0))
    valid_ratio = valid_frames / total_frames

    passes = valid_ratio >= min_valid_frame_ratio

    if passes:
        message = f"High-confidence capture: {valid_ratio:.1%} frames tracked cleanly."
    else:
        message = (
            f"Low detection confidence: Only {valid_ratio:.1%} of frames tracked "
            f"(required ≥{min_valid_frame_ratio:.0%}). Please ensure dancer is fully in frame, "
            "properly lit, and feet/hands are visible."
        )

    return {
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "valid_frame_ratio": round(valid_ratio, 3),
        "passes_confidence_gate": passes,
        "message": message
    }


def require_pose_quality(
    pose_sequence: List[Optional[np.ndarray]],
    min_valid_frame_ratio: float = 0.70
) -> Dict[str, Any]:
    """
    Strict validation that raises ValueError if pose detection confidence is insufficient.
    """
    report = pose_quality_report(pose_sequence, min_valid_frame_ratio)
    if not report["passes_confidence_gate"]:
        raise ValueError(report["message"])
    return report
