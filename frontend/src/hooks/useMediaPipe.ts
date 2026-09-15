import { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker, HandLandmarker } from "@mediapipe/tasks-vision";

export interface DetectionResult {
  fullBodyDetected: boolean;
  poseLandmarks: any[] | null;
  handLandmarks: any[] | null;
  framingStatus: "READY" | "STEP_BACK" | "CENTER_DANCER" | "NO_PERSON";
  framingMessage: string;
}

export function useMediaPipe(active: boolean = true) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [detectionResult, setDetectionResult] = useState<DetectionResult>({
    fullBodyDetected: false,
    poseLandmarks: null,
    handLandmarks: null,
    framingStatus: "NO_PERSON",
    framingMessage: "Initializing MediaPipe camera preview..."
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
      let framingMessage = "No dancer detected in camera frame.";
      let fullBodyDetected = false;

      if (hasPose && posePoints) {
        // Check visibility of key joints: shoulders (11,12), hips (23,24), ankles (27,28)
        const shouldersVisible = (posePoints[11]?.visibility ?? 0) > 0.4 && (posePoints[12]?.visibility ?? 0) > 0.4;
        const hipsVisible = (posePoints[23]?.visibility ?? 0) > 0.4 && (posePoints[24]?.visibility ?? 0) > 0.4;
        const anklesVisible = (posePoints[27]?.visibility ?? 0) > 0.4 && (posePoints[28]?.visibility ?? 0) > 0.4;

        if (shouldersVisible && hipsVisible && anklesVisible) {
          fullBodyDetected = true;
          framingStatus = "READY";
          framingMessage = "Full body detected. Ready to record!";
        } else if (shouldersVisible && hipsVisible && !anklesVisible) {
          framingStatus = "STEP_BACK";
          framingMessage = "Step back so your feet and ankles are visible.";
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
        framingMessage
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
