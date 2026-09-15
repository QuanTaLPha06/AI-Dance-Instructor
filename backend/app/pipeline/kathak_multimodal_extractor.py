"""
Kathak Multimodal Extractor: Body Pose, Hands (Mudras), and Facial Expressions (Bhavas)

Integrates:
1. MediaPipe Pose (33 skeletal body landmarks).
2. MediaPipe Hands (21 hand landmarks per hand for Kathak Mudra / Hastas evaluation).
3. MediaPipe Face Mesh (468 facial landmarks for Kathak Facial Expressions / Bhavas evaluation).
"""

import cv2
import os
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, List, Optional

class KathakMultimodalExtractor:
    def __init__(
        self,
        pose_model_path: str = r"d:\aidance\pose_landmarker_lite.task",
        hand_model_path: str = r"d:\aidance\hand_landmarker.task"
    ):
        self.pose_model_path = pose_model_path
        self.hand_model_path = hand_model_path
        self._init_mediapipe_solutions()

    def _init_mediapipe_solutions(self):
        """Initializes Holistic/Solutions MediaPipe models for pose, hand mudras, and face mesh."""
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_multimodal_features_from_frame(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """
        Extracts body pose, hand mudras, and face mesh landmarks from a single RGB frame.
        
        Returns:
            Dict containing:
            - 'pose': (33, 3) body keypoints
            - 'left_hand': (21, 3) left hand mudra keypoints or None
            - 'right_hand': (21, 3) right hand mudra keypoints or None
            - 'face_bhava': (468, 3) face mesh keypoints or None
        """
        results = self.holistic.process(rgb_frame)
        
        # 1. Pose landmarks (33)
        if results.pose_landmarks:
            pose_pts = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        else:
            pose_pts = np.zeros((33, 3))
            
        # 2. Left Hand Mudra (21)
        if results.left_hand_landmarks:
            left_hand_pts = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        else:
            left_hand_pts = np.zeros((21, 3))
            
        # 3. Right Hand Mudra (21)
        if results.right_hand_landmarks:
            right_hand_pts = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        else:
            right_hand_pts = np.zeros((21, 3))
            
        # 4. Facial Expression / Bhava (468)
        if results.face_landmarks:
            face_pts = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        else:
            face_pts = np.zeros((468, 3))
            
        return {
            "pose": pose_pts,
            "left_hand": left_hand_pts,
            "right_hand": right_hand_pts,
            "face_bhava": face_pts
        }

    def process_video_multimodal(self, video_path: str) -> Dict[str, Any]:
        """
        Processes entire video and returns frame sequences for pose, left/right hand mudras, and face expressions.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        pose_list = []
        left_hand_list = []
        right_hand_list = []
        face_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted = self.extract_multimodal_features_from_frame(rgb_frame)
            
            pose_list.append(extracted["pose"])
            left_hand_list.append(extracted["left_hand"])
            right_hand_list.append(extracted["right_hand"])
            face_list.append(extracted["face_bhava"])
            
        cap.release()
        
        return {
            "pose": np.array(pose_list),           # (T, 33, 3)
            "left_hand": np.array(left_hand_list),   # (T, 21, 3)
            "right_hand": np.array(right_hand_list), # (T, 21, 3)
            "face_bhava": np.array(face_list),       # (T, 468, 3)
            "fps": float(fps)
        }
