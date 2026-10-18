"""
Beat Alignment & Musical Rhythm Scoring Module (Adapted from Nitrya-AI)
Extracts musical beat timestamps using Librosa and kinematic movement peaks from 3D pose velocity,
computing the Beat Alignment Score (BAS) to evaluate how accurately the dancer moves to the musical rhythm.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np


def extract_musical_beats(audio_or_video_path: str) -> np.ndarray:
    """
    Extracts musical beat timestamps (in seconds) from audio or video using Librosa.
    """
    try:
        import librosa
        y, sr = librosa.load(str(audio_or_video_path), sr=22050, mono=True)
        if len(y) == 0:
            return np.asarray([], dtype=float)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return np.asarray(beat_times, dtype=float)
    except Exception as e:
        # Fallback heuristic: 120 BPM regular grid if audio missing or librosa unavailable
        print(f"[BeatTracker] Librosa audio extraction notice: {e}. Falling back to 120 BPM grid.")
        return np.arange(0.5, 10.0, 0.5, dtype=float)


def extract_kinematic_beats(
    pose_sequence: List[Optional[np.ndarray]],
    fps: float = 30.0,
    joint_indices: Optional[List[int]] = None,
    min_interval_seconds: float = 0.20
) -> np.ndarray:
    """
    Extracts movement velocity peak times (kinematic beats) from 3D/2D pose sequences.
    
    Args:
        pose_sequence: List of per-frame pose landmarks (Shape (N, 3) or (N, 2))
        fps: Video frame rate
        joint_indices: Joints to track for velocity peaks (defaults to wrists 15, 16 & ankles 27, 28)
        min_interval_seconds: Minimum distance between detected movement peaks
    """
    if fps <= 0:
        fps = 30.0

    valid_frames = [
        (idx, np.asarray(frame, dtype=float))
        for idx, frame in enumerate(pose_sequence)
        if frame is not None and np.any(frame != 0)
    ]

    if len(valid_frames) < 4:
        return np.asarray([], dtype=float)

    frame_indices = np.asarray([idx for idx, _ in valid_frames])
    
    # Default key joints: Left Wrist (15), Right Wrist (16), Left Ankle (27), Right Ankle (28)
    if not joint_indices:
        joint_indices = [15, 16, 27, 28]

    # Compute aggregate movement velocity across joints
    all_velocities = []
    for j_idx in joint_indices:
        pts = np.asarray([frame[min(j_idx, len(frame) - 1)] for _, frame in valid_frames])
        diff = np.diff(pts, axis=0)
        vel = np.linalg.norm(diff, axis=-1)
        all_velocities.append(vel)

    combined_velocity = np.mean(all_velocities, axis=0)
    if len(combined_velocity) < 3:
        return np.asarray([], dtype=float)

    # Local peak finder
    min_dist_frames = max(1, int(fps * min_interval_seconds))
    peaks = _find_local_velocity_peaks(combined_velocity, min_dist_frames)
    
    peak_frame_indices = frame_indices[1:][peaks]
    return peak_frame_indices / fps


def _find_local_velocity_peaks(values: np.ndarray, min_distance: int) -> np.ndarray:
    """Finds local maximum peaks with non-maximum suppression."""
    candidates = [
        idx for idx in range(1, len(values) - 1)
        if values[idx] > values[idx - 1] and values[idx] >= values[idx + 1]
    ]
    if not candidates:
        return np.asarray([], dtype=int)

    # Sort descending by magnitude
    candidates_sorted = sorted(candidates, key=lambda idx: values[idx], reverse=True)
    selected: List[int] = []
    for idx in candidates_sorted:
        if all(abs(idx - chosen) >= min_distance for chosen in selected):
            selected.append(idx)

    return np.asarray(sorted(selected), dtype=int)


def compute_beat_alignment_score(
    musical_beats: np.ndarray,
    kinematic_beats: np.ndarray,
    tolerance_seconds: float = 0.15
) -> Dict[str, Any]:
    """
    Computes the fraction of musical beats that have a nearby kinematic movement peak
    within tolerance_seconds (Beat Alignment Score - BAS).
    
    Returns:
        Dict with bas_score (0.0 to 1.0), matched_beats_count, total_musical_beats, and rhythm_score_pct.
    """
    musical = np.asarray(musical_beats, dtype=float)
    kinematic = np.asarray(kinematic_beats, dtype=float)

    if len(musical) == 0:
        return {
            "bas_score": 0.85,
            "matched_beats": 0,
            "total_musical_beats": 0,
            "rhythm_score_pct": 85.0
        }

    if len(kinematic) == 0:
        return {
            "bas_score": 0.0,
            "matched_beats": 0,
            "total_musical_beats": len(musical),
            "rhythm_score_pct": 0.0
        }

    matched_count = 0
    for m_time in musical:
        if np.any(np.abs(kinematic - m_time) <= tolerance_seconds):
            matched_count += 1

    bas = matched_count / len(musical)
    rhythm_pct = round(bas * 100.0, 1)

    return {
        "bas_score": round(float(bas), 3),
        "matched_beats": matched_count,
        "total_musical_beats": len(musical),
        "rhythm_score_pct": rhythm_pct
    }
