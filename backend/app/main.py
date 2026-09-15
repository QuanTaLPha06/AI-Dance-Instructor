"""
FastAPI Main Server Application for AI Dance Guidance System
Exposes Phase 1 Accuracy Core REST API endpoints for pose comparison, alignment, and analysis.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import tempfile
import os
import uuid
import numpy as np

from app.core.genre_configs import GENRE_PROFILES, get_genre_profile
from app.pipeline.pose_extractor import PoseExtractor
from app.pipeline.scoring_engine import DanceScoringEngine

app = FastAPI(
    title="AI Dance Instructor Backend",
    description="Accuracy Core API: 3D Pose Extraction, Spatial Procrustes Normalization, Joint Kinematics, Soft-DTW Temporal Alignment",
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

# In-memory storage for analysis results (Phase 1)
ANALYSIS_JOBS = {}

class GenreInfoResponse(BaseModel):
    key: str
    name: str
    description: str
    weight_form_spatial: float
    weight_joint_angles: float
    weight_temporal_rhythm: float

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Dance Instructor Accuracy Core"}

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
    
    # Save uploaded temporary files
    with tempfile.NamedTemporaryDirectory() as tmpdir:
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
            from app.pipeline.audio_analytics import AudioAnalyticsEngine
            audio_features = AudioAnalyticsEngine.extract_audio_rhythm_features(att_path)

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

@app.get("/api/v1/analyze/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return ANALYSIS_JOBS[job_id]
