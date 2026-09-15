"""
3D Kinematic Joint Angles Analytics Engine
Computes 3D joint angles (knees, elbows, hips, shoulders) from 3D pose landmark coordinates.
Translates coordinate deltas into interpretable posture features (e.g. Araimandi knee bend angle).
"""

import numpy as np
from typing import Dict, List, Any

# Standard MediaPipe 33 landmark index mapping
MP_LANDMARKS = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,     "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,     "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

class JointAnalyticsEngine:
    @staticmethod
    def calculate_vector_angle_3d(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculates angle in degrees between two 3D vectors v1 and v2.
        angle = arccos( (v1 . v2) / (|v1| * |v2|) )
        """
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return float(np.degrees(angle_rad))

    @classmethod
    def calculate_triplet_angle(cls, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculates joint angle at vertex point B given points A, B, C in 3D.
        Vectors: BA = A - B, BC = C - B
        """
        ba = a - b
        bc = c - b
        return cls.calculate_vector_angle_3d(ba, bc)

    @classmethod
    def extract_frame_joint_angles(cls, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extracts key 3D joint angles for a single frame of pose landmarks.
        landmarks: numpy array of shape (33, 3) or (N, 3)
        """
        angles = {}
        
        # Helper point getter
        def get_pt(name: str) -> np.ndarray:
            idx = MP_LANDMARKS[name]
            return landmarks[idx]

        try:
            # Left Knee: Left Hip -> Left Knee -> Left Ankle
            angles["left_knee"] = cls.calculate_triplet_angle(
                get_pt("LEFT_HIP"), get_pt("LEFT_KNEE"), get_pt("LEFT_ANKLE")
            )
            # Right Knee: Right Hip -> Right Knee -> Right Ankle
            angles["right_knee"] = cls.calculate_triplet_angle(
                get_pt("RIGHT_HIP"), get_pt("RIGHT_KNEE"), get_pt("RIGHT_ANKLE")
            )
            # Left Elbow: Left Shoulder -> Left Elbow -> Left Wrist
            angles["left_elbow"] = cls.calculate_triplet_angle(
                get_pt("LEFT_SHOULDER"), get_pt("LEFT_ELBOW"), get_pt("LEFT_WRIST")
            )
            # Right Elbow: Right Shoulder -> Right Elbow -> Right Wrist
            angles["right_elbow"] = cls.calculate_triplet_angle(
                get_pt("RIGHT_SHOULDER"), get_pt("RIGHT_ELBOW"), get_pt("RIGHT_WRIST")
            )
            # Left Hip / Torso: Left Shoulder -> Left Hip -> Left Knee
            angles["left_hip"] = cls.calculate_triplet_angle(
                get_pt("LEFT_SHOULDER"), get_pt("LEFT_HIP"), get_pt("LEFT_KNEE")
            )
            # Right Hip / Torso: Right Shoulder -> Right Hip -> Right Knee
            angles["right_hip"] = cls.calculate_triplet_angle(
                get_pt("RIGHT_SHOULDER"), get_pt("RIGHT_HIP"), get_pt("RIGHT_KNEE")
            )
            # Spine / Torso Inclination (Mid-Shoulder -> Mid-Hip -> Mid-Ankle)
            mid_shoulder = (get_pt("LEFT_SHOULDER") + get_pt("RIGHT_SHOULDER")) / 2.0
            mid_hip = (get_pt("LEFT_HIP") + get_pt("RIGHT_HIP")) / 2.0
            mid_ankle = (get_pt("LEFT_ANKLE") + get_pt("RIGHT_ANKLE")) / 2.0
            angles["spine_torso"] = cls.calculate_triplet_angle(mid_shoulder, mid_hip, mid_ankle)

        except Exception as e:
            # Fallback if indices out of bounds or missing
            pass

        return angles

    @classmethod
    def compute_sequence_joint_angles(cls, pose_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes joint angle trajectories over a sequence of frames T.
        pose_sequence: Shape (T, 33, 3)
        Returns dictionary mapping joint_name -> array of shape (T,)
        """
        T = len(pose_sequence)
        joint_names = ["left_knee", "right_knee", "left_elbow", "right_elbow", "left_hip", "right_hip", "spine_torso"]
        trajectories = {j: np.zeros(T) for j in joint_names}

        for t in range(T):
            frame_angles = cls.extract_frame_joint_angles(pose_sequence[t])
            for j in joint_names:
                trajectories[j][t] = frame_angles.get(j, 0.0)

        return trajectories
