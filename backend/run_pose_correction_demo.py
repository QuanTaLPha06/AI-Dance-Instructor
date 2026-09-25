"""
Standalone Demo Script: Multimodal AI Pose Corrector (YOLO + MediaPipe + Qwen Image-to-Image)
Tests joint angle analysis, prompt construction, and image-to-image pose correction.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.pipeline.qwen_pose_corrector import QwenPoseCorrectorEngine

def create_sample_dancer_image(width=640, height=640) -> np.ndarray:
    """Generates a synthetic test frame depicting a Kathak dancer pose."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        r = int(20 + 30 * (y / height))
        g = int(30 + 40 * (y / height))
        b = int(60 + 50 * (y / height))
        img[y, :] = (b, g, r)

    # Draw dancer torso silhouette
    cv2.circle(img, (320, 160), 40, (220, 200, 180), -1)  # Head
    cv2.line(img, (320, 200), (320, 380), (220, 200, 180), 12)  # Torso
    
    # Left Arm (Bent at incorrect angle ~45 deg)
    cv2.line(img, (320, 230), (230, 270), (220, 200, 180), 8)
    cv2.line(img, (230, 270), (180, 220), (220, 200, 180), 8)

    # Right Arm (Straight out ~90 deg)
    cv2.line(img, (320, 230), (410, 230), (220, 200, 180), 8)
    cv2.line(img, (410, 230), (480, 230), (220, 200, 180), 8)

    # Legs
    cv2.line(img, (320, 380), (270, 520), (220, 200, 180), 10)
    cv2.line(img, (320, 380), (370, 520), (220, 200, 180), 10)

    # Title Banner
    cv2.putText(img, "Dancer Input Frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return img

def main():
    print("==================================================================")
    print("Multimodal AI Pose Corrector Demo (YOLO + MediaPipe + Qwen)")
    print("==================================================================")

    # Initialize Engine (HF_TOKEN from environment if set)
    hf_token = os.environ.get("HF_TOKEN", "")
    corrector = QwenPoseCorrectorEngine(hf_token=hf_token, use_local_diffusers=False)

    # Generate or load test frame
    test_bgr = create_sample_dancer_image()

    print("\nProcessing video frame through YOLO + MediaPipe + Qwen pipeline...")
    results = corrector.analyze_and_correct_frame(test_bgr)

    print("\n" + "=" * 66)
    print("KINEMATIC POSE ANALYSIS RESULTS")
    print("=" * 66)
    print("Detected 3D Joint Angles:")
    for j, angle in results["joint_angles"].items():
        print(f"   - {j:16s}: {angle:.1f} deg")

    print("\nActionable Coaching Cues:")
    for cue in results["coaching_cues"]:
        print(f"   * {cue}")

    print("\nConstructed Qwen Prompt:")
    print(f"   \"{results['qwen_prompt']}\"")

    print(f"\nStatus: {results['generation_status']}")
    print("=" * 66)

    # Save Side-by-Side Comparison Output Image
    orig_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
    orig_pil = Image.fromarray(orig_rgb)
    corrected_pil = results["corrected_image_pil"]

    w, h = orig_pil.size
    combined = Image.new("RGB", (w * 2, h + 60), (15, 18, 28))
    
    draw = ImageDraw.Draw(combined)
    draw.text((20, 15), "ORIGINAL DANCER FRAME (YOLO + MEDIAPIPE)", fill=(0, 230, 255))
    draw.text((w + 20, 15), "QWEN GENERATED CORRECTED TARGET POSE", fill=(0, 255, 128))

    combined.paste(orig_pil, (0, 50))
    combined.paste(corrected_pil, (w, 50))

    output_path = os.path.abspath("pose_correction_comparison.png")
    combined.save(output_path)

    print(f"\nComparison output saved to: {output_path}")
    print("==================================================================")

if __name__ == "__main__":
    main()
