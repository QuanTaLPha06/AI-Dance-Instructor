"""
Audio Processing & Beat Sync Module using Librosa for Kathak Rhythm/Taal Analysis
"""

import numpy as np
from typing import Dict, Any, Tuple

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class KathakAudioProcessor:
    """Extracts audio tempo, beat timestamps, onset features, and Taal rhythm sync."""

    @staticmethod
    def extract_audio_rhythm_features(audio_path: str, sr: int = 22050) -> Dict[str, Any]:
        """
        Analyzes dance music/bol audio track to extract BPM, beat frames, onset envelope, and tempo stability.
        """
        if not HAS_LIBROSA:
            return {
                "bpm": 120.0,
                "beat_times": [0.0, 0.5, 1.0, 1.5, 2.0],
                "onset_env_mean": 0.5,
                "status": "librosa_not_installed"
            }

        y, sr = librosa.load(audio_path, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        onset_env = librosa.onnset.onset_strength(y=y, sr=sr) if hasattr(librosa, 'onnset') else librosa.onset.onset_strength(y=y, sr=sr)

        if isinstance(tempo, np.ndarray):
            bpm_val = float(tempo[0])
        else:
            bpm_val = float(tempo)

        return {
            "bpm": round(bpm_val, 2),
            "beat_times": beat_times.tolist(),
            "beat_count": len(beat_times),
            "duration": float(len(y) / sr),
            "onset_env_mean": float(np.mean(onset_env)),
            "status": "success"
        }

    @staticmethod
    def calculate_rhythm_dance_synchronization(
        pose_velocity_peaks: np.ndarray, 
        beat_times: np.ndarray,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Calculates rhythm alignment error between footwork/hand velocity peaks and musical beats.
        """
        if len(beat_times) == 0 or len(pose_velocity_peaks) == 0:
            return {"rhythm_sync_score": 5.0, "mean_latency_ms": 0.0}

        peak_times = pose_velocity_peaks / fps
        latencies = []

        for p_time in peak_times:
            closest_beat = beat_times[np.argmin(np.abs(beat_times - p_time))]
            latency_ms = abs(p_time - closest_beat) * 1000.0
            latencies.append(latency_ms)

        mean_latency = float(np.mean(latencies))
        sync_score = max(1.0, min(10.0, 10.0 - (mean_latency / 20.0)))

        return {
            "rhythm_sync_score": round(sync_score, 1),
            "mean_latency_ms": round(mean_latency, 1),
            "evaluated_beats": len(latencies)
        }
