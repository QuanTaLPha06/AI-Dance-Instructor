"""
Celery Task Worker for Async Dance Analysis Jobs
Integrates Redis message broker & Celery async task execution for heavy ML pipelines.
"""

from celery import Celery
import os
import numpy as np
from app.pipeline.pose_extractor import PoseExtractor
from app.pipeline.scoring_engine import DanceScoringEngine

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "dance_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task(name="tasks.process_video_analysis")
def process_video_analysis_task(ref_video_path: str, att_video_path: str, genre: str):
    """
    Async Celery task for extracting 3D pose landmarks and running Procrustes/DTW scoring.
    """
    extractor = PoseExtractor()
    ref_landmarks, ref_fps, _ = extractor.process_video(ref_video_path)
    att_landmarks, att_fps, _ = extractor.process_video(att_video_path)

    result = DanceScoringEngine.evaluate_performance(
        ref_sequence=ref_landmarks,
        att_sequence=att_landmarks,
        genre_key=genre,
        fps=att_fps
    )

    return result.to_dict()
