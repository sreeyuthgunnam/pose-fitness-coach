// MediaPipe Pose Detection for Browser
// Uses the newer MediaPipe Tasks Vision API

import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

let poseLandmarker: PoseLandmarker | null = null;
let isInitializing = false;

export interface Keypoint {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
  name?: string;
}

export interface Pose {
  keypoints: Keypoint[];
  score?: number;
}

// Keypoint names from MediaPipe pose model
export const KEYPOINT_NAMES = [
  'nose',           // 0
  'left_eye_inner', // 1
  'left_eye',       // 2
  'left_eye_outer', // 3
  'right_eye_inner',// 4
  'right_eye',      // 5
  'right_eye_outer',// 6
  'left_ear',       // 7
  'right_ear',      // 8
  'mouth_left',     // 9
  'mouth_right',    // 10
  'left_shoulder',  // 11
  'right_shoulder', // 12
  'left_elbow',     // 13
  'right_elbow',    // 14
  'left_wrist',     // 15
  'right_wrist',    // 16
  'left_pinky',     // 17
  'right_pinky',    // 18
  'left_index',     // 19
  'right_index',    // 20
  'left_thumb',     // 21
  'right_thumb',    // 22
  'left_hip',       // 23
  'right_hip',      // 24
  'left_knee',      // 25
  'right_knee',     // 26
  'left_ankle',     // 27
  'right_ankle',    // 28
  'left_heel',      // 29
  'right_heel',     // 30
  'left_foot_index',// 31
  'right_foot_index'// 32
];

// Map to indices for exercise tracking
export const KEYPOINT_INDICES: { [key: string]: number } = {
  'nose': 0,
  'left_eye': 2,
  'right_eye': 5,
  'left_ear': 7,
  'right_ear': 8,
  'left_shoulder': 11,
  'right_shoulder': 12,
  'left_elbow': 13,
  'right_elbow': 14,
  'left_wrist': 15,
  'right_wrist': 16,
  'left_hip': 23,
  'right_hip': 24,
  'left_knee': 25,
  'right_knee': 26,
  'left_ankle': 27,
  'right_ankle': 28,
};

export async function initPoseDetector(): Promise<PoseLandmarker> {
  if (poseLandmarker) return poseLandmarker;
  
  if (isInitializing) {
    while (isInitializing) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (poseLandmarker) return poseLandmarker;
  }

  isInitializing = true;

  try {
    console.log('Initializing pose detector...');
    
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    console.log('Pose detector initialized successfully');
    return poseLandmarker;
  } catch (error) {
    console.error('Failed to initialize pose detector:', error);
    throw error;
  } finally {
    isInitializing = false;
  }
}

export async function detectPose(
  video: HTMLVideoElement,
  timestamp: number
): Promise<Pose | null> {
  if (!poseLandmarker) {
    try {
      await initPoseDetector();
    } catch (error) {
      console.error('Could not initialize pose detector:', error);
      return null;
    }
  }

  if (!poseLandmarker || !video.videoWidth || !video.videoHeight) {
    return null;
  }

  try {
    const results = poseLandmarker.detectForVideo(video, timestamp);
    
    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      const keypoints: Keypoint[] = landmarks.map((lm, idx) => ({
        x: lm.x * video.videoWidth,
        y: lm.y * video.videoHeight,
        z: lm.z,
        visibility: lm.visibility,
        name: KEYPOINT_NAMES[idx]
      }));

      return { keypoints, score: 1.0 };
    }
  } catch (error) {
    console.error('Pose detection error:', error);
  }

  return null;
}

// Helper to get keypoint by name
export function getKeypoint(pose: Pose, name: string): Keypoint | null {
  const idx = KEYPOINT_INDICES[name];
  if (idx !== undefined && pose.keypoints[idx]) {
    return pose.keypoints[idx];
  }
  return pose.keypoints.find(kp => kp.name === name) || null;
}

// Skeleton connections for drawing
const POSE_CONNECTIONS: [number, number][] = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  // Upper body
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [15, 17], [15, 19], [15, 21], // left hand
  [16, 18], [16, 20], [16, 22], // right hand
  // Torso
  [11, 23], [12, 24], [23, 24],
  // Lower body
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31], // left leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32], // right leg
];

export function drawPose(
  ctx: CanvasRenderingContext2D,
  pose: Pose,
  videoWidth: number,
  videoHeight: number
): void {
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  
  const keypoints = pose.keypoints;
  if (!keypoints || keypoints.length === 0) return;

  // Draw connections
  ctx.strokeStyle = '#00FF00';
  ctx.lineWidth = 3;

  for (const [i, j] of POSE_CONNECTIONS) {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];

    if (kp1 && kp2 && 
        (kp1.visibility || 0) > 0.5 && 
        (kp2.visibility || 0) > 0.5) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
    }
  }

  // Draw keypoints
  for (const keypoint of keypoints) {
    if ((keypoint.visibility || 0) > 0.5) {
      // Outer circle
      ctx.fillStyle = '#FF0000';
      ctx.beginPath();
      ctx.arc(keypoint.x, keypoint.y, 6, 0, 2 * Math.PI);
      ctx.fill();

      // Inner circle
      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath();
      ctx.arc(keypoint.x, keypoint.y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}
