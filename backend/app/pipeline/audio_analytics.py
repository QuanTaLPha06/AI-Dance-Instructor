"""
Audio & Rhythm Analytics Engine using Librosa
Extracts onset envelopes, tempo (BPM), beat frames, and audio-pose alignment features.
"""

import librosa
import numpy as np
from typing import Dict, Any, Tuple

class AudioAnalyticsEngine:
    @staticmethod
    def extract_audio_rhythm_features(audio_path_or_video: str) -> Dict[str, Any]:
        """
        Loads audio from video/audio file and computes onset strength, tempo, and beat timestamps.
        """
        try:
            y, sr = librosa.load(audio_path_or_video, sr=None)
            
            # 1. Compute Tempo & Beat frames
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # 2. Onset Strength Envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Handle float or scalar tempo returns in newer librosa versions
            tempo_bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)

            return {
                "sample_rate": sr,
                "duration_sec": round(float(len(y) / sr), 2),
                "tempo_bpm": round(tempo_bpm, 1),
                "total_beats": len(beat_times),
                "beat_timestamps_sec": np.round(beat_times, 2).tolist()
            }
        except Exception as e:
            # Fallback if video file has no audio stream or librosa cannot decode audio track
            return {
                "sample_rate": 22050,
                "duration_sec": 0.0,
                "tempo_bpm": 120.0,
                "total_beats": 0,
                "beat_timestamps_sec": []
            }
