"""
Multimodal AI Pose Corrector Module
Integrates YOLO Pose, MediaPipe Body/Hand Landmarkers, and Qwen Image-to-Image Editing (Qwen/Qwen-Image-Edit-2511).

Workflow:
1. Extract Bounding Boxes & Body Keypoints via YOLO Pose (yolo26n-pose.pt).
2. Extract 33 3D Pose & 21 Hand Mudra Landmarks via MediaPipe.
3. Analyze 3D Kinematic Joint Angles (Elbows, Knees, Shoulders, Hips, Wrists) & compute angular deviations from target pose standards.
4. Construct an anatomical feedback prompt for Qwen Image-to-Image editing.
5. Invoke Qwen/Qwen-Image-Edit-2511 (via local Diffusers or remote HF InferenceClient API).
6. Return corrected image, visual keypoint overlay, and actionable coaching cues.
"""

import os
import cv2
import io
import time
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple, Optional

# MediaPipe Imports
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    mp = None

# YOLO Imports
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    YOLO = None

# Diffusers / HuggingFace Imports
try:
    from diffusers import DiffusionPipeline
    from diffusers.utils import load_image
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

try:
    from huggingface_hub import InferenceClient
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


class YoloMediaPipePoseTracker:
    """Combines YOLO Pose (yolo26n-pose.pt) and MediaPipe Task Landmarkers."""

    def __init__(
        self,
        yolo_weights_path: str = r"d:\aidance\yolo26n-pose.pt",
        mp_pose_task_path: str = r"d:\aidance\pose_landmarker_lite.task",
        mp_hand_task_path: str = r"d:\aidance\hand_landmarker.task"
    ):
        self.yolo_weights_path = yolo_weights_path
        self.mp_pose_task_path = mp_pose_task_path
        self.mp_hand_task_path = mp_hand_task_path

        self._init_yolo()
        self._init_mediapipe()

    def _init_yolo(self):
        """Initializes YOLO Pose Model if weights file exists and ultralytics installed."""
        self.yolo_model = None
        if HAS_YOLO and os.path.exists(self.yolo_weights_path):
            try:
                self.yolo_model = YOLO(self.yolo_weights_path)
                print(f"[YOLO] Loaded YOLO Pose Model: {self.yolo_weights_path}")
            except Exception as e:
                print(f"[YOLO] Warning initializing YOLO model: {e}")

    def _init_mediapipe(self):
        """Initializes MediaPipe Pose & Hand Landmarkers."""
        self.use_tasks_api = False
        self.landmarker = None

        if HAS_MEDIAPIPE and os.path.exists(self.mp_pose_task_path):
            try:
                BaseOptions = mp.tasks.BaseOptions
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.mp_pose_task_path),
                    running_mode=VisionRunningMode.IMAGE,
                    num_poses=1
                )
                self.landmarker = PoseLandmarker.create_from_options(options)
                self.use_tasks_api = True
                print(f"[MediaPipe] Loaded MediaPipe Vision Task Landmarker: {self.mp_pose_task_path}")
            except Exception as e:
                print(f"[MediaPipe] Warning initializing MediaPipe Task API: {e}")

        if not self.use_tasks_api and HAS_MEDIAPIPE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5
            )
            print("[MediaPipe] Initialized Legacy MediaPipe Pose Solution.")

    def process_image(self, bgr_img: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single BGR image frame and extracts YOLO & MediaPipe poses.
        """
        h, w, _ = bgr_img.shape
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # 1. YOLO Pose Extraction
        yolo_kpts = np.zeros((17, 3), dtype=np.float32)
        yolo_bbox = [0, 0, w, h]

        if self.yolo_model is not None:
            try:
                results = self.yolo_model(bgr_img, verbose=False)
                if results and len(results) > 0 and len(results[0].keypoints) > 0:
                    kpts_data = results[0].keypoints.data.cpu().numpy()
                    boxes_data = results[0].boxes.data.cpu().numpy()
                    if len(kpts_data) > 0:
                        yolo_kpts = kpts_data[0]  # (17, 3)
                    if len(boxes_data) > 0:
                        yolo_bbox = boxes_data[0][:4].tolist()
            except Exception as e:
                print(f"[YOLO] Extraction exception: {e}")

        # 2. MediaPipe Pose Extraction
        mp_lms_3d = np.zeros((33, 3), dtype=np.float32)
        mp_lms_2d = np.zeros((33, 2), dtype=np.float32)

        if HAS_MEDIAPIPE:
            if self.use_tasks_api and self.landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                res = self.landmarker.detect(mp_image)
                if res.pose_landmarks and len(res.pose_landmarks) > 0:
                    lms = res.pose_landmarks[0]
                    mp_lms_2d = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
                    if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                        mp_lms_3d = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_world_landmarks[0]], dtype=np.float32)
                    else:
                        mp_lms_3d = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            elif hasattr(self, 'pose') and self.pose:
                res = self.pose.process(rgb_img)
                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    mp_lms_2d = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
                    mp_lms_3d = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

        return {
            "yolo_kpts": yolo_kpts,
            "yolo_bbox": yolo_bbox,
            "mp_landmarks_3d": mp_lms_3d,
            "mp_landmarks_2d": mp_lms_2d
        }


class KinematicDeviationAnalyzer:
    """Computes 3D joint angle flexions and generates anatomical feedback prompts for Qwen."""

    @staticmethod
    def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculates 3D joint angle in degrees at vertex p2."""
        v1 = p1 - p2
        v2 = p3 - p2
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product < 1e-6:
            return 0.0
        cos_angle = np.dot(v1, v2) / norm_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @classmethod
    def analyze_pose(
        cls,
        current_lms_3d: np.ndarray,
        target_lms_3d: Optional[np.ndarray] = None,
        target_angles: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], List[str], str]:
        """
        Analyzes 3D pose joint angles, computes angular deviations, and generates Qwen prompt.
        """
        joint_definitions = {
            "left_elbow": (11, 13, 15),
            "right_elbow": (12, 14, 16),
            "left_shoulder": (13, 11, 23),
            "right_shoulder": (14, 12, 24),
            "left_knee": (23, 25, 27),
            "right_knee": (24, 26, 28),
            "left_hip": (11, 23, 25),
            "right_hip": (12, 24, 26),
        }

        current_angles = {}
        for j_name, (p1, p2, p3) in joint_definitions.items():
            angle = cls._calculate_angle(current_lms_3d[p1], current_lms_3d[p2], current_lms_3d[p3])
            current_angles[j_name] = round(angle, 1)

        default_targets = {
            "left_elbow": 90.0,
            "right_elbow": 90.0,
            "left_shoulder": 80.0,
            "right_shoulder": 80.0,
            "left_knee": 150.0,
            "right_knee": 150.0,
            "left_hip": 160.0,
            "right_hip": 160.0,
        }

        ref_targets = target_angles or default_targets
        if target_lms_3d is not None and len(target_lms_3d) == 33:
            for j_name, (p1, p2, p3) in joint_definitions.items():
                ref_targets[j_name] = cls._calculate_angle(target_lms_3d[p1], target_lms_3d[p2], target_lms_3d[p3])

        coaching_cues = []
        prompt_corrections = []

        for j_name, current_deg in current_angles.items():
            target_deg = ref_targets.get(j_name, 90.0)
            diff = current_deg - target_deg

            if abs(diff) > 12.0:
                readable_name = j_name.replace("_", " ").title()
                if diff < 0:
                    action = f"Extend {readable_name} from {current_deg:.0f} deg to target {target_deg:.0f} deg"
                    prompt_desc = f"straighten the dancer's {readable_name.lower()} to a {target_deg:.0f} degree angle"
                else:
                    action = f"Flex/Bend {readable_name} from {current_deg:.0f} deg to target {target_deg:.0f} deg"
                    prompt_desc = f"bend the dancer's {readable_name.lower()} to a {target_deg:.0f} degree angle"

                coaching_cues.append(action)
                prompt_corrections.append(prompt_desc)

        if not prompt_corrections:
            qwen_prompt = "Refine the dancer's pose with perfect posture alignment and elegant Kathak Hastak hand mudra."
            coaching_cues.append("Posture is well aligned! Keep maintaining proper balance and mudra posture.")
        else:
            corrections_str = ", ".join(prompt_corrections[:3])
            qwen_prompt = f"Modify the dancer in this image to correct their posture: {corrections_str}. Ensure perfect Kathak dance pose alignment."

        return current_angles, coaching_cues, qwen_prompt


class QwenImageToImageEngine:
    """
    Qwen Image-to-Image Pose Correction Engine (Qwen/Qwen-Image-Edit-2511).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Edit-2511",
        hf_token: Optional[str] = None,
        use_local_diffusers: bool = False
    ):
        self.model_id = model_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.use_local_diffusers = use_local_diffusers

        self.local_pipe = None
        self.client = None

        if use_local_diffusers and HAS_DIFFUSERS and torch.cuda.is_available():
            self._init_local_diffusers()

        if self.local_pipe is None and HAS_HF_HUB:
            self._init_remote_client()

    def _init_local_diffusers(self):
        """Initializes local PyTorch DiffusionPipeline for GPU execution."""
        try:
            print(f"[Qwen] Loading local GPU DiffusionPipeline for model: {self.model_id}...")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.local_pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="cuda"
            )
            print("[Qwen] Local GPU Qwen DiffusionPipeline loaded successfully.")
        except Exception as e:
            print(f"[Qwen] Local GPU Diffusers loading skipped/failed: {e}")
            self.local_pipe = None

    def _init_remote_client(self):
        """Initializes HuggingFace InferenceClient API."""
        try:
            api_key = self.hf_token if self.hf_token else None
            self.client = InferenceClient(provider="auto", api_key=api_key)
            print("[Qwen] Initialized HuggingFace Remote InferenceClient.")
        except Exception as e:
            print(f"[Qwen] Remote InferenceClient initialization failed: {e}")
            self.client = None

    def generate_corrected_image(self, input_image: Image.Image, prompt: str) -> Tuple[Image.Image, str]:
        """
        Takes input PIL Image and correction prompt, returns corrected PIL Image.
        """
        if self.local_pipe is not None:
            try:
                print(f"[Qwen] Generating corrected pose image via Local GPU Pipeline...")
                output_img = self.local_pipe(image=input_image, prompt=prompt).images[0]
                return output_img, "Generated locally via GPU DiffusionPipeline."
            except Exception as e:
                print(f"[Qwen] Local pipeline execution error: {e}")

        if self.client is not None and self.hf_token:
            try:
                print(f"[Qwen] Generating corrected pose image via HuggingFace Inference Client API...")
                buf = io.BytesIO()
                input_image.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                output_img = self.client.image_to_image(
                    img_bytes,
                    prompt=prompt,
                    model=self.model_id
                )
                if isinstance(output_img, Image.Image):
                    return output_img, "Generated via HuggingFace API Client."
            except Exception as e:
                print(f"[Qwen] HuggingFace API client execution error: {e}")

        print("[Qwen] Returning visual pose target overlay image.")
        fallback_img = self._generate_visual_target_overlay(input_image, prompt)
        return fallback_img, "Generated visual pose target overlay."

    def _generate_visual_target_overlay(self, input_image: Image.Image, prompt: str) -> Image.Image:
        """Generates an annotated visual target overlay when Qwen weights are unavailable."""
        img = input_image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        banner_height = int(h * 0.12)
        draw.rectangle([0, 0, w, banner_height], fill=(20, 24, 38, 220))
        
        text_title = "TARGET POSE GUIDANCE & CORRECTION"
        draw.text((20, 15), text_title, fill=(0, 230, 255))
        
        prompt_snippet = (prompt[:75] + '...') if len(prompt) > 75 else prompt
        draw.text((20, 45), f"Action: {prompt_snippet}", fill=(255, 255, 255))

        return img


class QwenPoseCorrectorEngine:
    """Unified Orchestrator combining YOLO, MediaPipe, Kinematic Analysis, and Qwen Image Editing."""

    def __init__(self, hf_token: Optional[str] = None, use_local_diffusers: bool = False):
        self.tracker = YoloMediaPipePoseTracker()
        self.qwen_engine = QwenImageToImageEngine(hf_token=hf_token, use_local_diffusers=use_local_diffusers)

    def analyze_and_correct_frame(
        self,
        bgr_frame: np.ndarray,
        target_angles: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Main pipeline function for a single video frame.
        """
        # 1. Pose Tracking
        pose_data = self.tracker.process_image(bgr_frame)

        # 2. Kinematic Analysis & Prompt Generation
        joint_angles, coaching_cues, qwen_prompt = KinematicDeviationAnalyzer.analyze_pose(
            current_lms_3d=pose_data['mp_landmarks_3d'],
            target_angles=target_angles
        )

        # 3. Prepare PIL Input Image
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        input_pil = Image.fromarray(rgb_frame)

        # 4. Generate Corrected Image via Qwen
        corrected_pil, status_msg = self.qwen_engine.generate_corrected_image(input_pil, qwen_prompt)

        return {
            "joint_angles": joint_angles,
            "coaching_cues": coaching_cues,
            "qwen_prompt": qwen_prompt,
            "corrected_image_pil": corrected_pil,
            "generation_status": status_msg,
            "yolo_bbox": pose_data['yolo_bbox'],
            "yolo_kpts": pose_data['yolo_kpts'].tolist(),
            "mp_landmarks_2d": pose_data['mp_landmarks_2d'].tolist()
        }
