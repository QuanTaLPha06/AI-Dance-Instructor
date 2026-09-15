"""
Genre Configuration Engine
Defines dynamic scoring profiles, joint tolerances, keyframe sensitivity,
and sub-model requirements for Western (Pop/Freestyle) vs Indian Classical (Bharatanatyam/Kathak).
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field

class GenreProfile(BaseModel):
    name: str
    description: str
    weight_form_spatial: float = Field(..., description="Weight for Procrustes 3D posture match (0.0 to 1.0)")
    weight_joint_angles: float = Field(..., description="Weight for 3D kinematic joint angles match (0.0 to 1.0)")
    weight_temporal_rhythm: float = Field(..., description="Weight for Soft-DTW temporal/rhythm alignment (0.0 to 1.0)")
    joint_tolerance_degrees: float = Field(..., description="Acceptable angular deviation threshold in degrees")
    keyframe_strictness: float = Field(..., description="Strictness multiplier for keyframe matches")
    active_mudra_detection: bool = Field(False, description="Enable hand landmarker mudra processing layer")
    key_joints: List[str] = Field(default_factory=list, description="Key joint angle triplets prioritized for scoring")


GENRE_PROFILES: Dict[str, GenreProfile] = {
    "western_freestyle": GenreProfile(
        name="Western Freestyle / Pop / Hip-Hop",
        description="Emphasizes rhythm, groove, flow, and tempo over precise angular posture compliance.",
        weight_form_spatial=0.25,
        weight_joint_angles=0.35,
        weight_temporal_rhythm=0.40,
        joint_tolerance_degrees=22.5,
        keyframe_strictness=1.0,
        active_mudra_detection=False,
        key_joints=["left_elbow", "right_elbow", "left_knee", "right_knee", "left_hip", "right_hip"]
    ),
    "indian_classical": GenreProfile(
        name="Indian Classical (Bharatanatyam / Kathak)",
        description="Rigorous geometry, exact Araimandi knee bends, torso posture alignment, and mudra accuracy.",
        weight_form_spatial=0.35,
        weight_joint_angles=0.45,
        weight_temporal_rhythm=0.20,
        joint_tolerance_degrees=10.0,
        keyframe_strictness=2.0,
        active_mudra_detection=True,
        key_joints=["left_knee", "right_knee", "left_hip", "right_hip", "left_elbow", "right_elbow", "spine_torso"]
    ),
    "kathak": GenreProfile(
        name="Kathak Classical Dance & Teaching Engine",
        description="Evaluates Tatkars (footwork), Hastaks (hand movements), Mudras, and Facial Expressions (Bhavas).",
        weight_form_spatial=0.30,
        weight_joint_angles=0.40,
        weight_temporal_rhythm=0.30,
        joint_tolerance_degrees=8.0,
        keyframe_strictness=2.5,
        active_mudra_detection=True,
        key_joints=["left_elbow", "right_elbow", "left_knee", "right_knee", "left_shoulder", "right_shoulder"]
    )
}

def get_genre_profile(genre_key: str) -> GenreProfile:
    """Returns the genre configuration profile or defaults to western_freestyle."""
    return GENRE_PROFILES.get(genre_key.lower(), GENRE_PROFILES["western_freestyle"])
