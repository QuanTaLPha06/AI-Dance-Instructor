import { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker, HandLandmarker } from "@mediapipe/tasks-vision";

export interface DetectionResult {
  fullBodyDetected: boolean;
  poseLandmarks: any[] | null;
  handLandmarks: any[] | null;
  framingStatus: "READY" | "STEP_BACK" | "CENTER_DANCER" | "OUT_OF_FRAME" | "NO_PERSON";
  framingMessage: string;
  isFlashWarning: boolean;
  jointDeviations?: Record<string, { status: "perfect" | "minor" | "critical"; diff: number }>;
}

export function useMediaPipe(active: boolean = true) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [detectionResult, setDetectionResult] = useState<DetectionResult>({
    fullBodyDetected: false,
    poseLandmarks: null,
    handLandmarks: null,
    framingStatus: "NO_PERSON",
    framingMessage: "Initializing MediaPipe camera preview...",
    isFlashWarning: false
  });

  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);

  useEffect(() => {
    let mounted = true;

    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        if (!mounted) return;

        // Initialize Pose Landmarker using local .task model
        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1
        });

        // Initialize Hand Landmarker using local .task model
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });

        poseLandmarkerRef.current = poseLandmarker;
        handLandmarkerRef.current = handLandmarker;
        setIsLoaded(true);
      } catch (err) {
        console.error("Failed to initialize MediaPipe tasks:", err);
      }
    }

    if (active) {
      initMediaPipe();
    }

    return () => {
      mounted = false;
      poseLandmarkerRef.current?.close();
      handLandmarkerRef.current?.close();
    };
  }, [active]);

  const processVideoFrame = (videoElement: HTMLVideoElement, timestampMs: number): DetectionResult => {
    if (!isLoaded || !poseLandmarkerRef.current) {
      return detectionResult;
    }

    try {
      const poseRes = poseLandmarkerRef.current.detectForVideo(videoElement, timestampMs);
      const handRes = handLandmarkerRef.current?.detectForVideo(videoElement, timestampMs);

      const hasPose = poseRes.landmarks && poseRes.landmarks.length > 0;
      const posePoints = hasPose ? poseRes.landmarks[0] : null;

      let framingStatus: DetectionResult["framingStatus"] = "NO_PERSON";
      let framingMessage = "No dancer detected in camera frame. Step into view.";
      let fullBodyDetected = false;
      let isFlashWarning = false;

      if (hasPose && posePoints) {
        // Robust key joint landmark indices:
        // 0: nose, 11: L shoulder, 12: R shoulder, 23: L hip, 24: R hip, 27: L ankle, 28: R ankle
        const shouldersVisible = (posePoints[11]?.visibility ?? 0) > 0.35 && (posePoints[12]?.visibility ?? 0) > 0.35;
        const hipsVisible = (posePoints[23]?.visibility ?? 0) > 0.35 && (posePoints[24]?.visibility ?? 0) > 0.35;
        const anklesVisible = (posePoints[27]?.visibility ?? 0) > 0.35 && (posePoints[28]?.visibility ?? 0) > 0.35;

        // Position coordinates normalized [0, 1]
        const noseY = posePoints[0]?.y ?? 0.5;
        const leftAnkleY = posePoints[27]?.y ?? 0.8;
        const rightAnkleY = posePoints[28]?.y ?? 0.8;
        const leftSideX = Math.min(posePoints[11]?.x ?? 0.5, posePoints[15]?.x ?? 0.5);
        const rightSideX = Math.max(posePoints[12]?.x ?? 0.5, posePoints[16]?.x ?? 0.5);

        // Boundary margins check:
        // Left margin: 0.04 | Right margin: 0.96 | Top: 0.03 | Bottom: 0.97
        if (leftSideX < 0.04) {
          framingStatus = "OUT_OF_FRAME";
          isFlashWarning = true;
          framingMessage = "⚠️ FLASH WARNING: Move Right! Breaching left edge.";
        } else if (rightSideX > 0.96) {
          framingStatus = "OUT_OF_FRAME";
          isFlashWarning = true;
          framingMessage = "⚠️ FLASH WARNING: Move Left! Breaching right edge.";
        } else if (noseY < 0.03) {
          framingStatus = "OUT_OF_FRAME";
          isFlashWarning = true;
          framingMessage = "⚠️ FLASH WARNING: Tilt camera up or step back! Head clipping.";
        } else if ((leftAnkleY > 0.97 || rightAnkleY > 0.97) || !anklesVisible) {
          framingStatus = "STEP_BACK";
          isFlashWarning = true;
          framingMessage = "⚠️ STEP BACK: Move back until feet and ghungroos are visible.";
        } else if (shouldersVisible && hipsVisible && anklesVisible) {
          fullBodyDetected = true;
          framingStatus = "READY";
          framingMessage = "Perfect framing! Dancer is centered and ready.";
        } else {
          framingStatus = "CENTER_DANCER";
          framingMessage = "Center yourself in the camera view.";
        }
      }

      const res: DetectionResult = {
        fullBodyDetected,
        poseLandmarks: posePoints,
        handLandmarks: handRes?.landmarks || null,
        framingStatus,
        framingMessage,
        isFlashWarning
      };

      setDetectionResult(res);
      return res;
    } catch (e) {
      return detectionResult;
    }
  };

  return {
    isLoaded,
    detectionResult,
    processVideoFrame
  };
}
