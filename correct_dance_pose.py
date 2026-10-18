"""
CLI Tool: Multimodal AI Dance Pose Corrector (YOLO + MediaPipe + Qwen Image-to-Image)
Runs pose extraction, kinematic joint angle deviation analysis, and generates the corrected perfect dance position using Qwen/Qwen-Image-Edit-2511.
"""

import os
import sys
import argparse
import cv2
from PIL import Image

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.pipeline.qwen_pose_corrector import QwenPoseCorrectorEngine, DancePoseStandards


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal AI Dance Pose Corrector (YOLO + MediaPipe + Qwen)")
    parser.add_argument("--image", type=str, default="sample_dancer_actual.jpg", help="Path to actual dancer image")
    parser.add_argument("--video", type=str, default=None, help="Optional: Path to video file to extract frame from")
    parser.add_argument("--frame-idx", type=int, default=30, help="Frame index if extracting from video")
    parser.add_argument("--reference-image", type=str, default=None, help="Optional: Reference master dancer image")
    parser.add_argument(
        "--pose",
        type=str,
        default="kathak_hastak",
        choices=["kathak_hastak", "kathak_aramandi", "kathak_tatkar", "kathak_nataraja", "western_freestyle"],
        help="Target dance stance standard"
    )
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API Token for cloud Qwen inference")
    parser.add_argument("--local-gpu", action="store_true", help="Use local GPU PyTorch Diffusers pipeline")
    parser.add_argument("--output-prefix", type=str, default="dance_correction", help="Prefix for output images")
    return parser.parse_args()


def load_target_frame(args) -> cv2.Mat:
    """Loads frame from image path or video file."""
    if args.video and os.path.exists(args.video):
        print(f"[Input] Extracting frame #{args.frame_idx} from video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = min(args.frame_idx, max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return frame
        print(f"[Warning] Failed extracting frame #{frame_idx}, falling back to --image")

    if os.path.exists(args.image):
        print(f"[Input] Loading dancer image: {args.image}")
        img = cv2.imread(args.image)
        if img is not None:
            return img

    print(f"[Error] Could not find input image at {args.image}")
    sys.exit(1)


def main():
    args = parse_args()
    print("=" * 70)
    print("  Multimodal AI Dance Pose Corrector")
    print("  Layer: YOLO Keypoints + MediaPipe 3D Kinematics + Qwen Image-to-Image")
    print("=" * 70)

    # 1. Load Input Frame
    bgr_frame = load_target_frame(args)
    h, w = bgr_frame.shape[:2]
    print(f"[Input] Frame loaded successfully: {w}x{h} px")

    # 2. Load Optional Reference Image
    target_ref_bgr = None
    if args.reference_image and os.path.exists(args.reference_image):
        print(f"[Reference] Loading master dancer reference image: {args.reference_image}")
        target_ref_bgr = cv2.imread(args.reference_image)

    # 3. Initialize Engine
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    corrector = QwenPoseCorrectorEngine(
        hf_token=hf_token,
        use_local_diffusers=args.local_gpu
    )

    # 4. Run Analysis & Pose Correction
    print(f"\n[Processing] Analyzing posture against standard: {args.pose}...")
    results = corrector.analyze_and_correct_frame(
        bgr_frame=bgr_frame,
        target_pose_name=args.pose,
        target_reference_bgr=target_ref_bgr
    )

    # 5. Display Kinematic Results
    print("\n" + "=" * 70)
    print(f"  KINEMATIC ACCURACY ANALYSIS: {results['pose_standard_name']}")
    print(f"  Overall Alignment Score: {results['accuracy_score']:.1f}%")
    print("=" * 70)

    print("\nJoint Angles & Angular Deviations:")
    for j_name, angle in results["joint_angles"].items():
        tgt = results["target_angles"].get(j_name, 0.0)
        diff = results["angle_deviations"].get(j_name, 0.0)
        status = "OK" if abs(diff) <= 12.0 else f"ERR: {diff:+.1f} deg"
        print(f"   - {j_name:16s} : Measured {angle:5.1f}° | Target {tgt:5.1f}° | [{status}]")

    print("\nActionable Dance Coaching Cues:")
    for cue in results["coaching_cues"]:
        print(f"   * {cue}")

    print("\nSynthesized Qwen Image-to-Image Prompt:")
    print(f"   \"{results['qwen_prompt']}\"")

    print(f"\nGeneration Status: {results['generation_status']}")
    print("=" * 70)

    # 6. Save Outputs
    output_comparison_path = f"{args.output_prefix}_comparison.png"
    output_corrected_path = f"{args.output_prefix}_corrected.png"
    output_annotated_path = f"{args.output_prefix}_annotated_original.png"

    results["comparison_image_pil"].save(output_comparison_path)
    results["corrected_image_pil"].save(output_corrected_path)
    results["annotated_original_pil"].save(output_annotated_path)

    print(f"\n[Saved Outputs]:")
    print(f"  1. Side-by-Side Comparison : {os.path.abspath(output_comparison_path)}")
    print(f"  2. Corrected Perfect Pose  : {os.path.abspath(output_corrected_path)}")
    print(f"  3. Annotated Original Frame: {os.path.abspath(output_annotated_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
