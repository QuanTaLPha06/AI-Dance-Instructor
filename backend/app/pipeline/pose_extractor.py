"""
MediaPipe & YOLO Pose Landmark Extractor
Processes input video files frame-by-frame and extracts 3D normalized landmark matrices (T, 33, 3).
Handles local .task files and video frame decoding via OpenCV.
"""

import cv2
import os
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional

class PoseExtractor:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or r"d:\aidance\pose_landmarker_lite.task"
        self._init_mediapipe_landmarker()

    def _init_mediapipe_landmarker(self):
        """Initializes Python MediaPipe Vision Task Landmarker if model file exists."""
        if os.path.exists(self.model_path):
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_poses=1
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            self.use_tasks_api = True
        else:
            # Fallback to legacy MP Pose solutions if task file missing
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_tasks_api = False

    def process_video(self, video_path: str) -> Tuple[np.ndarray, float, int]:
        """
        Reads video file and extracts 3D landmarks for all frames.
        
        Returns:
            (landmarks_array, fps, total_frames)
            landmarks_array: Shape (T, 33, 3) representing 3D (x, y, z) coordinates.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_landmarks_list = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.use_tasks_api:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((frame_idx / fps) * 1000)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]])
                else:
                    pts = np.zeros((33, 3))
            else:
                results = self.pose.process(rgb_frame)
                if results.pose_landmarks:
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                else:
                    pts = np.zeros((33, 3))

            frame_landmarks_list.append(pts)
            frame_idx += 1

        cap.release()

        if not frame_landmarks_list:
            raise ValueError(f"No frames could be extracted from video: {video_path}")

        landmarks_matrix = np.array(frame_landmarks_list)
        return landmarks_matrix, float(fps), frame_idx
