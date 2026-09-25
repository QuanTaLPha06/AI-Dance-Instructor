"""
FastAPI Main Server Application for AI Dance Guidance System
Exposes REST API endpoints for pose comparison, 3D alignment, joint analytics,
and Qwen Image-to-Image visual pose correction.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import tempfile
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional, List, Dict, Any

from app.core.genre_configs import GENRE_PROFILES, get_genre_profile
from app.pipeline.pose_extractor import PoseExtractor
from app.pipeline.scoring_engine import DanceScoringEngine

# Import Qwen Pose Corrector Engine
try:
    from app.pipeline.qwen_pose_corrector import QwenPoseCorrectorEngine
    HAS_QWEN_ENGINE = True
except ImportError:
    HAS_QWEN_ENGINE = False
    QwenPoseCorrectorEngine = None

app = FastAPI(
    title="AI Dance Instructor Backend",
    description="Accuracy Core API: 3D Pose Extraction, Spatial Procrustes Normalization, Joint Kinematics, Soft-DTW Temporal Alignment & Qwen Visual Pose Correction",
    version="1.0.0"
)

# CORS middleware for Next.js PWA client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for analysis jobs
ANALYSIS_JOBS = {}

# Global instance of Qwen Pose Corrector Engine
qwen_corrector_instance = None

def get_qwen_corrector() -> Optional[Any]:
    global qwen_corrector_instance
    if qwen_corrector_instance is None and HAS_QWEN_ENGINE:
        hf_token = os.environ.get("HF_TOKEN", "")
        qwen_corrector_instance = QwenPoseCorrectorEngine(hf_token=hf_token, use_local_diffusers=False)
    return qwen_corrector_instance


class GenreInfoResponse(BaseModel):
    key: str
    name: str
    description: str
    weight_form_spatial: float
    weight_joint_angles: float
    weight_temporal_rhythm: float


class VisualCorrectionRequest(BaseModel):
    genre: Optional[str] = "indian_classical"
    target_angles: Optional[Dict[str, float]] = None


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AI Dance Instructor Backend",
        "qwen_engine_available": HAS_QWEN_ENGINE
    }

@app.get("/api/v1/genres", response_model=list[GenreInfoResponse])
def list_genres():
    res = []
    for k, v in GENRE_PROFILES.items():
        res.append(GenreInfoResponse(
            key=k,
            name=v.name,
            description=v.description,
            weight_form_spatial=v.weight_form_spatial,
            weight_joint_angles=v.weight_joint_angles,
            weight_temporal_rhythm=v.weight_temporal_rhythm
        ))
    return res

@app.post("/api/v1/analyze/videos")
async def analyze_dance_videos(
    reference_video: UploadFile = File(...),
    attempt_video: UploadFile = File(...),
    genre: str = Form("western_freestyle")
):
    """
    Receives reference video & dancer attempt video, runs pose extraction,
    Procrustes normalization, 3D kinematic joint angle analytics, and Soft-DTW alignment.
    """
    job_id = str(uuid.uuid4())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, f"ref_{reference_video.filename}")
        att_path = os.path.join(tmpdir, f"att_{attempt_video.filename}")

        with open(ref_path, "wb") as f_ref:
            shutil.copyfileobj(reference_video.file, f_ref)

        with open(att_path, "wb") as f_att:
            shutil.copyfileobj(attempt_video.file, f_att)

        try:
            # 1. Extract 3D Pose Sequences
            extractor = PoseExtractor()
            ref_landmarks, ref_fps, _ = extractor.process_video(ref_path)
            att_landmarks, att_fps, _ = extractor.process_video(att_path)

            # 2. Evaluate Scoring Engine
            result = DanceScoringEngine.evaluate_performance(
                ref_sequence=ref_landmarks,
                att_sequence=att_landmarks,
                genre_key=genre,
                fps=att_fps
            )

            # 3. Librosa Audio Rhythm Analytics
            try:
                from app.pipeline.audio_analytics import AudioAnalyticsEngine
                audio_features = AudioAnalyticsEngine.extract_audio_rhythm_features(att_path)
            except Exception:
                audio_features = {"beat_count": 0, "tempo_bpm": 120.0}

            res_dict = result.to_dict()
            res_dict["audio_rhythm_analytics"] = audio_features

            response_payload = {
                "job_id": job_id,
                "status": "completed",
                "result": res_dict
            }
            ANALYSIS_JOBS[job_id] = response_payload
            return response_payload

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/v1/pose/visual-correction")
async def generate_visual_pose_correction(
    frame_image: UploadFile = File(...),
    genre: str = Form("indian_classical")
):
    """
    Generates AI Image-to-Image target pose correction using Qwen/Qwen-Image-Edit-2511,
    YOLO Pose keypoints, and MediaPipe 3D joint analytics.
    """
    corrector = get_qwen_corrector()
    if corrector is None:
        raise HTTPException(status_code=503, detail="Qwen Pose Corrector Engine unavailable.")

    try:
        contents = await frame_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr_frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

        # Run Qwen + YOLO + MediaPipe Pose Corrector Pipeline
        results = corrector.analyze_and_correct_frame(bgr_frame)

        # Convert corrected PIL image to Base64 data URL
        corrected_pil = results["corrected_image_pil"]
        buf = io.BytesIO()
        corrected_pil.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        corrected_data_url = f"data:image/png;base64,{b64_str}"

        return {
            "status": "success",
            "genre": genre,
            "joint_angles": results["joint_angles"],
            "coaching_cues": results["coaching_cues"],
            "qwen_prompt": results["qwen_prompt"],
            "generation_status": results["generation_status"],
            "corrected_image_url": corrected_data_url,
            "yolo_bbox": results["yolo_bbox"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual pose correction failed: {str(e)}")


@app.get("/api/v1/analyze/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return ANALYSIS_JOBS[job_id]
