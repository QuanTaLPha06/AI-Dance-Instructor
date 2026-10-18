"""
FastAPI Main Server Application for AI Dance Guidance System (Integrated with Nitrya-AI)
Exposes REST API endpoints for:
1. Reference Dancer Profiles & Pose Caching (/api/v1/profiles)
2. Asynchronous Attempt Evaluation & Progress Tracking (/api/v1/attempts)
3. Direct Video Evaluation with Soft-DTW & Beat Alignment (/api/v1/analyze/videos)
4. Multimodal Qwen Image-to-Image Pose Correction (/api/v1/pose/visual-correction)
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
import time
from typing import Optional, List, Dict, Any

from app.core.genre_configs import GENRE_PROFILES, get_genre_profile
from app.pipeline.pose_extractor import PoseExtractor
from app.pipeline.scoring_engine import DanceScoringEngine
from app.pipeline.beat_tracker import extract_musical_beats
from app.pipeline.profile_cache import ProfileCacheManager
from app.pipeline.quality_gate import pose_quality_report

# Import Qwen Pose Corrector Engine
try:
    from app.pipeline.qwen_pose_corrector import QwenPoseCorrectorEngine
    HAS_QWEN_ENGINE = True
except ImportError:
    HAS_QWEN_ENGINE = False
    QwenPoseCorrectorEngine = None

app = FastAPI(
    title="AiDance Instructor Core API",
    description="Accuracy Core API: 3D Pose Extraction, Spatial Procrustes Normalization, Joint Kinematics, Soft-DTW Temporal Alignment, Beat Alignment Scoring & Qwen Visual Pose Correction",
    version="2.0.0"
)

# CORS middleware for Next.js client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for analysis jobs & progress tracking
ANALYSIS_JOBS: Dict[str, Dict[str, Any]] = {}

# Global instance of Qwen Pose Corrector Engine
qwen_corrector_instance = None

def get_qwen_corrector(hf_token: Optional[str] = None, use_local_gpu: bool = False) -> Optional[Any]:
    global qwen_corrector_instance
    if HAS_QWEN_ENGINE:
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if qwen_corrector_instance is None or token:
            qwen_corrector_instance = QwenPoseCorrectorEngine(hf_token=token, use_local_diffusers=use_local_gpu)
    return qwen_corrector_instance


class GenreInfoResponse(BaseModel):
    key: str
    name: str
    description: str
    weight_form_spatial: float
    weight_joint_angles: float
    weight_temporal_rhythm: float


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AiDance Instructor Backend (Integrated with Nitrya-AI)",
        "qwen_engine_available": HAS_QWEN_ENGINE,
        "profiles_cached": len(ProfileCacheManager.list_profiles()),
        "active_jobs": len(ANALYSIS_JOBS)
    }


@app.get("/api/v1/genres", response_model=List[GenreInfoResponse])
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


# ==============================================================================
# 1. REFERENCE PROFILES & POSE CACHING (Nitrya-AI Architecture)
# ==============================================================================

@app.post("/api/v1/profiles")
async def create_profile(
    dancer_name: str = Form(...),
    genre: str = Form("kathak"),
    notes: str = Form(""),
    video: UploadFile = File(...)
):
    """
    Registers a master dancer reference routine: extracts 3D pose landmarks and
    musical beats once, caching them to disk (.npz) for instantaneous comparisons.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = os.path.splitext(video.filename or "ref.mp4")[1] or ".mp4"
        ref_path = os.path.join(tmpdir, f"ref_upload{suffix}")

        with open(ref_path, "wb") as f_ref:
            shutil.copyfileobj(video.file, f_ref)

        try:
            profile_meta = ProfileCacheManager.create_or_update_profile(
                dancer_name=dancer_name,
                genre=genre,
                video_path=ref_path,
                notes=notes
            )
            return {
                "status": "ready",
                "message": f"Profile created and cached for {dancer_name}.",
                "profile": profile_meta
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed caching profile: {str(e)}")


@app.get("/api/v1/profiles")
def list_profiles():
    """Lists all cached dancer reference profiles."""
    return ProfileCacheManager.list_profiles()


@app.get("/api/v1/profiles/{profile_id}")
def get_profile_details(profile_id: str):
    """Gets details for a single reference profile."""
    profile = ProfileCacheManager.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


# ==============================================================================
# 2. ASYNC ATTEMPT EVALUATION & PROGRESS TRACKING (Nitrya-AI Workflow)
# ==============================================================================

def _run_async_attempt_analysis(
    job_id: str,
    att_video_path: str,
    profile_id: Optional[str] = None,
    ref_video_path: Optional[str] = None,
    genre: str = "kathak",
    run_qwen_correction: bool = False,
    hf_token: Optional[str] = None
):
    """Background worker task executing staged dance evaluation."""
    try:
        t0 = time.perf_counter()

        # Stage 1: Load Reference Pose Sequence (Cached or Fresh)
        ANALYSIS_JOBS[job_id] = {"status": "processing", "stage": "loading_reference", "pct": 10}

        ref_sequence = None
        ref_fps = 30.0
        musical_beats = None

        if profile_id:
            cached_data = ProfileCacheManager.load_cached_pose_sequence(profile_id)
            ref_sequence = cached_data["landmarks_3d"]
            ref_fps = cached_data["fps"]
            musical_beats = cached_data["musical_beats"]
            genre = cached_data["profile"].get("genre", genre)
        elif ref_video_path:
            extractor = PoseExtractor()
            ref_sequence, ref_fps, _ = extractor.process_video(ref_video_path)
            ref_sequence = np.asarray(ref_sequence, dtype=np.float32)
            musical_beats = extract_musical_beats(ref_video_path)

        if ref_sequence is None or len(ref_sequence) == 0:
            raise ValueError("No reference pose sequence available.")

        # Stage 2: Extract Dancer Attempt Pose
        ANALYSIS_JOBS[job_id] = {"status": "processing", "stage": "extracting_attempt_pose", "pct": 35}
        extractor = PoseExtractor()
        att_landmarks, att_fps, att_duration = extractor.process_video(att_video_path)
        att_sequence = np.asarray(att_landmarks, dtype=np.float32)

        # Stage 3: Confidence & Quality Gate
        ANALYSIS_JOBS[job_id] = {"status": "processing", "stage": "evaluating_quality_gate", "pct": 55}
        quality_report = pose_quality_report([lm for lm in att_sequence])

        # Stage 4: Scoring Engine (Kinematics, Soft-DTW & Beat Alignment Score)
        ANALYSIS_JOBS[job_id] = {"status": "processing", "stage": "scoring_kinematics_and_rhythm", "pct": 75}
        scoring_result = DanceScoringEngine.evaluate_performance(
            ref_sequence=ref_sequence,
            att_sequence=att_sequence,
            genre_key=genre,
            fps=att_fps,
            musical_beats=musical_beats
        )
        res_dict = scoring_result.to_dict()

        # Stage 5: Optional Qwen Visual Correction on Worst Frame
        qwen_correction_output = None
        if run_qwen_correction and scoring_result.worst_keyframe:
            ANALYSIS_JOBS[job_id] = {"status": "processing", "stage": "generating_qwen_pose_correction", "pct": 90}
            worst_frame_idx = scoring_result.worst_keyframe.get("attempt_frame_index", 0)

            # Extract worst frame from video
            cap = cv2.VideoCapture(att_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, worst_frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                corrector = get_qwen_corrector(hf_token=hf_token)
                if corrector:
                    correction_res = corrector.analyze_and_correct_frame(
                        bgr_frame=frame,
                        target_pose_name=genre
                    )
                    buf = io.BytesIO()
                    correction_res["comparison_image_pil"].save(buf, format="PNG")
                    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                    qwen_correction_output = {
                        "comparison_image_url": f"data:image/png;base64,{b64_str}",
                        "qwen_prompt": correction_res["qwen_prompt"],
                        "generation_status": correction_res["generation_status"]
                    }

        elapsed_sec = round(time.perf_counter() - t0, 2)
        res_dict["execution_time_seconds"] = elapsed_sec
        res_dict["qwen_visual_correction"] = qwen_correction_output

        ANALYSIS_JOBS[job_id] = {
            "status": "completed",
            "stage": "finalized",
            "pct": 100,
            "result": res_dict
        }

    except Exception as e:
        ANALYSIS_JOBS[job_id] = {
            "status": "failed",
            "stage": "error",
            "pct": 0,
            "error": str(e)
        }


@app.post("/api/v1/attempts")
async def create_attempt_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    profile_id: Optional[str] = Form(None),
    reference_video: Optional[UploadFile] = File(None),
    genre: str = Form("kathak"),
    run_qwen_correction: bool = Form(False),
    hf_token: Optional[str] = Form(None)
):
    """
    Submits an attempt video for analysis against a cached reference profile or reference video.
    Returns immediately with a job_id for async polling.
    """
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    att_path = os.path.join(temp_dir, f"att_{job_id}.mp4")

    with open(att_path, "wb") as f_att:
        shutil.copyfileobj(video.file, f_att)

    ref_path = None
    if reference_video is not None:
        ref_path = os.path.join(temp_dir, f"ref_{job_id}.mp4")
        with open(ref_path, "wb") as f_ref:
            shutil.copyfileobj(reference_video.file, f_ref)

    ANALYSIS_JOBS[job_id] = {
        "status": "queued",
        "stage": "queued",
        "pct": 0
    }

    background_tasks.add_task(
        _run_async_attempt_analysis,
        job_id=job_id,
        att_video_path=att_path,
        profile_id=profile_id,
        ref_video_path=ref_path,
        genre=genre,
        run_qwen_correction=run_qwen_correction,
        hf_token=hf_token
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Attempt analysis queued. Poll /api/v1/attempts/{job_id}/status for progress."
    }


@app.get("/api/v1/attempts/{job_id}/status")
def get_attempt_status(job_id: str):
    """Polls status and staged progress of an attempt analysis job."""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Attempt Job ID not found")
    return ANALYSIS_JOBS[job_id]


# ==============================================================================
# 3. DIRECT SYNCHRONOUS VIDEO COMPARISON
# ==============================================================================

@app.post("/api/v1/analyze/videos")
async def analyze_dance_videos(
    reference_video: UploadFile = File(...),
    attempt_video: UploadFile = File(...),
    genre: str = Form("western_freestyle")
):
    """
    Synchronously analyzes reference video & dancer attempt video, running 3D pose extraction,
    Procrustes normalization, Soft-DTW alignment, Beat Alignment Scoring (BAS), and keyframe diagnostics.
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

            ref_arr = np.asarray(ref_landmarks, dtype=np.float32)
            att_arr = np.asarray(att_landmarks, dtype=np.float32)

            # 2. Extract Musical Beats
            musical_beats = extract_musical_beats(att_path)

            # 3. Evaluate Performance
            result = DanceScoringEngine.evaluate_performance(
                ref_sequence=ref_arr,
                att_sequence=att_arr,
                genre_key=genre,
                fps=att_fps,
                musical_beats=musical_beats
            )

            res_dict = result.to_dict()
            response_payload = {
                "job_id": job_id,
                "status": "completed",
                "result": res_dict
            }
            ANALYSIS_JOBS[job_id] = response_payload
            return response_payload

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ==============================================================================
# 4. MULTIMODAL QWEN POSE CORRECTION LAYER
# ==============================================================================

@app.post("/api/v1/pose/visual-correction")
async def generate_visual_pose_correction(
    frame_image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None),
    pose_preset: str = Form("kathak_hastak"),
    genre: str = Form("indian_classical"),
    hf_token: Optional[str] = Form(None),
    use_local_gpu: bool = Form(False)
):
    """
    Generates AI Image-to-Image target pose correction using Qwen/Qwen-Image-Edit-2511,
    YOLO Pose keypoints, and MediaPipe 3D joint analytics on the actual dancer image.
    """
    if not HAS_QWEN_ENGINE or QwenPoseCorrectorEngine is None:
        raise HTTPException(status_code=503, detail="Qwen Pose Corrector Engine unavailable.")

    try:
        # 1. Decode Actual Dancer Frame
        contents = await frame_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr_frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

        # 2. Decode Optional Reference Instructor Frame
        target_ref_bgr = None
        if reference_image is not None:
            ref_bytes = await reference_image.read()
            if len(ref_bytes) > 0:
                ref_arr = np.frombuffer(ref_bytes, np.uint8)
                target_ref_bgr = cv2.imdecode(ref_arr, cv2.IMREAD_COLOR)

        # 3. Instantiate engine with requested token/GPU mode
        token_to_use = hf_token or os.environ.get("HF_TOKEN", "")
        corrector = QwenPoseCorrectorEngine(
            hf_token=token_to_use,
            use_local_diffusers=use_local_gpu
        )

        # 4. Run Multimodal Pipeline (YOLO + MediaPipe + Qwen)
        results = corrector.analyze_and_correct_frame(
            bgr_frame=bgr_frame,
            target_pose_name=pose_preset,
            target_reference_bgr=target_ref_bgr
        )

        # Helper to convert PIL Image to Base64 data URL
        def pil_to_data_url(pil_img: Image.Image) -> str:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"

        corrected_data_url = pil_to_data_url(results["corrected_image_pil"])
        annotated_orig_url = pil_to_data_url(results["annotated_original_pil"])
        comparison_url = pil_to_data_url(results["comparison_image_pil"])

        return {
            "status": "success",
            "genre": genre,
            "pose_standard_name": results["pose_standard_name"],
            "accuracy_score": results["accuracy_score"],
            "joint_angles": results["joint_angles"],
            "target_angles": results["target_angles"],
            "angle_deviations": results["angle_deviations"],
            "coaching_cues": results["coaching_cues"],
            "qwen_prompt": results["qwen_prompt"],
            "generation_status": results["generation_status"],
            "corrected_image_url": corrected_data_url,
            "annotated_original_url": annotated_orig_url,
            "comparison_image_url": comparison_url,
            "yolo_bbox": results["yolo_bbox"],
            "yolo_confidence": results.get("yolo_confidence", 0.0)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual pose correction failed: {str(e)}")


@app.get("/api/v1/analyze/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return ANALYSIS_JOBS[job_id]
