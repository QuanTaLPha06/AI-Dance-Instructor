"""
Reference Profile & Pose Cache Manager (Adapted from Nitrya-AI)
Manages reference dancer profiles, caches extracted 3D/2D pose sequences and audio beat grids
to disk (.npz format) so reference routines can be reused instantaneously without re-extraction.
"""

import os
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

CACHE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "cached_profiles")))
INDEX_FILE = CACHE_DIR / "profiles_index.json"


class ProfileCacheManager:
    @classmethod
    def _ensure_dir(cls):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not INDEX_FILE.exists():
            INDEX_FILE.write_text("{}", encoding="utf-8")

    @classmethod
    def _read_index(cls) -> Dict[str, Any]:
        cls._ensure_dir()
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @classmethod
    def _write_index(cls, data: Dict[str, Any]):
        cls._ensure_dir()
        INDEX_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def create_or_update_profile(
        cls,
        dancer_name: str,
        genre: str,
        video_path: str,
        profile_id: Optional[str] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Processes a reference video, extracts its 3D pose landmarks and audio beats once,
        and caches them to disk for instantaneous future evaluations.
        """
        cls._ensure_dir()
        pid = profile_id or str(uuid.uuid4())[:8]

        from app.pipeline.pose_extractor import PoseExtractor
        from app.pipeline.beat_tracker import extract_musical_beats
        from app.pipeline.quality_gate import pose_quality_report

        # 1. Extract 3D Pose Sequence
        extractor = PoseExtractor()
        landmarks_3d, fps, duration = extractor.process_video(video_path)
        landmarks_array = np.asarray(landmarks_3d, dtype=np.float32)

        # 2. Extract Musical Beats
        musical_beats = extract_musical_beats(video_path)

        # 3. Quality Validation
        quality = pose_quality_report([lm for lm in landmarks_array])

        # 4. Save Cache File (.npz)
        cache_file = CACHE_DIR / f"{pid}_pose.npz"
        np.savez_compressed(
            str(cache_file),
            landmarks_3d=landmarks_array,
            musical_beats=musical_beats,
            fps=np.array([fps], dtype=np.float32),
            duration=np.array([duration], dtype=np.float32)
        )

        # 5. Update Index
        index = cls._read_index()
        profile_meta = {
            "id": pid,
            "dancer_name": dancer_name,
            "genre": genre,
            "notes": notes,
            "fps": round(float(fps), 2),
            "duration_sec": round(float(duration), 2),
            "total_frames": len(landmarks_array),
            "valid_frame_ratio": quality["valid_frame_ratio"],
            "cache_file": str(cache_file),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        index[pid] = profile_meta
        cls._write_index(index)

        print(f"[ProfileCache] Saved reference profile {pid} ({dancer_name} - {genre}) with {len(landmarks_array)} frames.")
        return profile_meta

    @classmethod
    def get_profile(cls, profile_id: str) -> Optional[Dict[str, Any]]:
        index = cls._read_index()
        return index.get(profile_id)

    @classmethod
    def list_profiles(cls) -> List[Dict[str, Any]]:
        index = cls._read_index()
        return list(index.values())

    @classmethod
    def load_cached_pose_sequence(cls, profile_id: str) -> Dict[str, Any]:
        """
        Loads the cached numpy pose array and musical beats in milliseconds.
        """
        profile = cls.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id!r} not found in cache.")

        cache_file = Path(profile["cache_file"])
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file {cache_file} is missing.")

        data = np.load(str(cache_file))
        return {
            "profile": profile,
            "landmarks_3d": data["landmarks_3d"],
            "musical_beats": data["musical_beats"],
            "fps": float(data["fps"][0]),
            "duration": float(data["duration"][0])
        }
