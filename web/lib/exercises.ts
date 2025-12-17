// Exercise types and configurations

export interface ExerciseConfig {
  id: string;
  name: string;
  emoji: string;
  description: string;
  halfBody: boolean;
  tips: string[];
}

export interface ExerciseResult {
  repCount: number;
  stage: string;
  feedback: string;
  formScore: number;
}

export interface Keypoint {
  x: number;
  y: number;
  z?: number;
  score?: number;
  visibility?: number;
  name?: string;
}

export interface Pose {
  keypoints: Keypoint[];
  score?: number;
}

// Exercise configurations
export const EXERCISES: ExerciseConfig[] = [
  {
    id: 'bicep_curl',
    name: 'Bicep Curl',
    emoji: '💪',
    description: 'Arm curls for bicep strength',
    halfBody: true,
    tips: ['Keep elbow close to body', 'Full range of motion', 'Control the movement'],
  },
  {
    id: 'shoulder_press',
    name: 'Shoulder Press',
    emoji: '🏋️',
    description: 'Overhead press for shoulders',
    halfBody: true,
    tips: ['Start at shoulder level', 'Press straight up', 'Full arm extension'],
  },
  {
    id: 'lateral_raise',
    name: 'Lateral Raise',
    emoji: '↔️',
    description: 'Side raises for deltoids',
    halfBody: true,
    tips: ['Arms at sides', 'Raise to shoulder level', 'Slight elbow bend'],
  },
  {
    id: 'front_raise',
    name: 'Front Raise',
    emoji: '⬆️',
    description: 'Front raises for anterior deltoids',
    halfBody: true,
    tips: ['Arms in front', 'Raise to shoulder height', 'Keep arms straight'],
  },
  {
    id: 'shoulder_shrug',
    name: 'Shoulder Shrug',
    emoji: '🤷',
    description: 'Shrugs for trapezius',
    halfBody: true,
    tips: ['Relax shoulders first', 'Shrug up to ears', 'Hold briefly at top'],
  },
  {
    id: 'squat',
    name: 'Squat',
    emoji: '🦵',
    description: 'Squats for leg strength',
    halfBody: false,
    tips: ['Feet shoulder-width apart', 'Keep back straight', 'Knees over toes'],
  },
];

// MediaPipe pose keypoint indices (33 landmarks)
export const KEYPOINT_INDICES = {
  nose: 0,
  left_eye_inner: 1,
  left_eye: 2,
  left_eye_outer: 3,
  right_eye_inner: 4,
  right_eye: 5,
  right_eye_outer: 6,
  left_ear: 7,
  right_ear: 8,
  mouth_left: 9,
  mouth_right: 10,
  left_shoulder: 11,
  right_shoulder: 12,
  left_elbow: 13,
  right_elbow: 14,
  left_wrist: 15,
  right_wrist: 16,
  left_pinky: 17,
  right_pinky: 18,
  left_index: 19,
  right_index: 20,
  left_thumb: 21,
  right_thumb: 22,
  left_hip: 23,
  right_hip: 24,
  left_knee: 25,
  right_knee: 26,
  left_ankle: 27,
  right_ankle: 28,
};

// Calculate angle between three points
export function calculateAngle(
  a: Keypoint,
  b: Keypoint,
  c: Keypoint
): number {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180) / Math.PI);
  if (angle > 180) {
    angle = 360 - angle;
  }
  return angle;
}

// Get keypoint by name
export function getKeypoint(keypoints: Keypoint[], name: string): Keypoint | null {
  const index = KEYPOINT_INDICES[name as keyof typeof KEYPOINT_INDICES];
  if (index !== undefined && keypoints[index]) {
    const kp = keypoints[index];
    // Check visibility (MediaPipe) or score (MoveNet)
    const confidence = kp.visibility ?? kp.score ?? 0;
    if (confidence > 0.3) {
      return kp;
    }
  }
  return null;
}

// Exercise tracker class
export class ExerciseTracker {
  private exerciseId: string;
  public repCount: number = 0;
  public stage: string = 'neutral';
  private lastAngle: number = 0;

  constructor(exerciseId: string) {
    this.exerciseId = exerciseId;
  }

  reset(): void {
    this.repCount = 0;
    this.stage = 'neutral';
    this.lastAngle = 0;
  }

  process(keypoints: Keypoint[]): ExerciseResult {
    switch (this.exerciseId) {
      case 'bicep_curl':
        return this.processBicepCurl(keypoints);
      case 'shoulder_press':
        return this.processShoulderPress(keypoints);
      case 'lateral_raise':
        return this.processLateralRaise(keypoints);
      case 'front_raise':
        return this.processFrontRaise(keypoints);
      case 'shoulder_shrug':
        return this.processShoulderShrug(keypoints);
      case 'squat':
        return this.processSquat(keypoints);
      default:
        return this.processBicepCurl(keypoints);
    }
  }

  private processBicepCurl(keypoints: Keypoint[]): ExerciseResult {
    const shoulder = getKeypoint(keypoints, 'right_shoulder');
    const elbow = getKeypoint(keypoints, 'right_elbow');
    const wrist = getKeypoint(keypoints, 'right_wrist');

    if (!shoulder || !elbow || !wrist) {
      return {
        repCount: this.repCount,
        stage: this.stage,
        feedback: 'Position your right arm in view',
        formScore: 0,
      };
    }

    const angle = calculateAngle(shoulder, elbow, wrist);
    let feedback = '';
    let formScore = 100;

    // Check form - elbow should stay relatively stationary
    const elbowShoulderDist = Math.abs(elbow.x - shoulder.x);
    if (elbowShoulderDist > 0.15) {
      feedback = 'Keep elbow close to body';
      formScore -= 20;
    }

    // Rep counting logic
    if (angle > 150) {
      if (this.stage === 'up') {
        this.repCount++;
        feedback = 'Good rep! 💪';
      }
      this.stage = 'down';
    } else if (angle < 50) {
      this.stage = 'up';
      if (!feedback) feedback = 'Good curl! Now extend';
    } else {
      if (!feedback) {
        feedback = this.stage === 'down' ? 'Curl up!' : 'Extend arm fully';
      }
    }

    return {
      repCount: this.repCount,
      stage: this.stage,
      feedback: feedback || 'Keep going!',
      formScore: Math.max(0, formScore),
    };
  }

  private processShoulderPress(keypoints: Keypoint[]): ExerciseResult {
    const shoulder = getKeypoint(keypoints, 'right_shoulder');
    const elbow = getKeypoint(keypoints, 'right_elbow');
    const wrist = getKeypoint(keypoints, 'right_wrist');

    if (!shoulder || !elbow || !wrist) {
      return {
        repCount: this.repCount,
        stage: this.stage,
        feedback: 'Position your right arm in view',
        formScore: 0,
      };
    }

    const angle = calculateAngle(shoulder, elbow, wrist);
    let feedback = '';
    let formScore = 100;

    // Rep counting - arm extended up vs at shoulder
    if (angle > 160 && wrist.y < shoulder.y) {
      if (this.stage === 'down') {
        this.repCount++;
        feedback = 'Great press! 🏋️';
      }
      this.stage = 'up';
    } else if (angle < 100 && wrist.y > elbow.y - 0.05) {
      this.stage = 'down';
      if (!feedback) feedback = 'Press up!';
    }

    if (!feedback) {
      feedback = this.stage === 'up' ? 'Lower to shoulders' : 'Press overhead';
    }

    return {
      repCount: this.repCount,
      stage: this.stage,
      feedback,
      formScore: Math.max(0, formScore),
    };
  }

  private processLateralRaise(keypoints: Keypoint[]): ExerciseResult {
    const shoulder = getKeypoint(keypoints, 'right_shoulder');
    const elbow = getKeypoint(keypoints, 'right_elbow');
    const wrist = getKeypoint(keypoints, 'right_wrist');
    const hip = getKeypoint(keypoints, 'right_hip');

    if (!shoulder || !wrist) {
      return {
        repCount: this.repCount,
        stage: this.stage,
        feedback: 'Position your arms in view',
        formScore: 0,
      };
    }

    // Check arm height relative to shoulder
    const armRaised = wrist.y < shoulder.y + 0.05;
    const armDown = wrist.y > shoulder.y + 0.2;

    let feedback = '';
    let formScore = 100;

    if (armRaised) {
      if (this.stage === 'down') {
        this.repCount++;
        feedback = 'Good raise! ↔️';
      }
      this.stage = 'up';
    } else if (armDown) {
      this.stage = 'down';
      feedback = 'Raise arms to sides';
    }

    if (!feedback) {
      feedback = this.stage === 'up' ? 'Lower slowly' : 'Raise to shoulder level';
    }

    return {
      repCount: this.repCount,
      stage: this.stage,
      feedback,
      formScore: Math.max(0, formScore),
    };
  }

  private processFrontRaise(keypoints: Keypoint[]): ExerciseResult {
    const shoulder = getKeypoint(keypoints, 'right_shoulder');
    const wrist = getKeypoint(keypoints, 'right_wrist');

    if (!shoulder || !wrist) {
      return {
        repCount: this.repCount,
        stage: this.stage,
        feedback: 'Position your arms in view',
        formScore: 0,
      };
    }

    const armRaised = wrist.y < shoulder.y;
    const armDown = wrist.y > shoulder.y + 0.25;

    let feedback = '';
    let formScore = 100;

    if (armRaised) {
      if (this.stage === 'down') {
        this.repCount++;
        feedback = 'Good raise! ⬆️';
      }
      this.stage = 'up';
    } else if (armDown) {
      this.stage = 'down';
      feedback = 'Raise arms forward';
    }

    if (!feedback) {
      feedback = this.stage === 'up' ? 'Lower slowly' : 'Raise to shoulder level';
    }

    return {
      repCount: this.repCount,
      stage: this.stage,
      feedback,
      formScore: Math.max(0, formScore),
    };
  }

  private processShoulderShrug(keypoints: Keypoint[]): ExerciseResult {
    const leftShoulder = getKeypoint(keypoints, 'left_shoulder');
    const rightShoulder = getKeypoint(keypoints, 'right_shoulder');
    const nose = getKeypoint(keypoints, 'nose');

    if (!leftShoulder || !rightShoulder || !nose) {
      return {
        repCount: this.repCount,
        stage: this.stage,
        feedback: 'Position your shoulders in view',
        formScore: 0,
      };
    }

    // Average shoulder height
    const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    const shoulderToNose = nose.y - shoulderY;

    // Shrugged position - shoulders closer to nose
    const isShrugged = shoulderToNose > 0.08;
    const isRelaxed = shoulderToNose < 0.15;

    let feedback = '';
    let formScore = 100;

    if (isShrugged) {
      if (this.stage === 'down') {
        this.repCount++;
        feedback = 'Good shrug! 🤷';
      }
      this.stage = 'up';
    } else if (isRelaxed) {
      this.stage = 'down';
      feedback = 'Shrug shoulders up';
    }

    if (!feedback) {
      feedback = this.stage === 'up' ? 'Hold and lower' : 'Shrug up to ears';
    }

    return {
      repCount: this.repCount,
      stage: this.stage,
      feedback,
      formScore: Math.max(0, formScore),
    };
  }

  private processSquat(keypoints: Keypoint[]): ExerciseResult {
    const hip = getKeypoint(keypoints, 'right_hip');
    const knee = getKeypoint(keypoints, 'right_knee');
    const ankle = getKeypoint(keypoints, 'right_ankle');

    if (!hip || !knee || !ankle) {
      return {
        repCount: this.repCount,
        stage: this.stage,
        feedback: 'Position your full body in view',
        formScore: 0,
      };
    }

    const angle = calculateAngle(hip, knee, ankle);
    let feedback = '';
    let formScore = 100;

    // Check knee position
    if (knee.x > ankle.x + 0.05) {
      feedback = 'Knees going too far forward!';
      formScore -= 20;
    }

    if (angle > 160) {
      if (this.stage === 'down') {
        this.repCount++;
        feedback = 'Good squat! 🦵';
      }
      this.stage = 'up';
    } else if (angle < 100) {
      this.stage = 'down';
      if (!feedback) feedback = 'Good depth! Stand up';
    }

    if (!feedback) {
      feedback = this.stage === 'up' ? 'Squat down!' : 'Go deeper or stand up';
    }

    return {
      repCount: this.repCount,
      stage: this.stage,
      feedback,
      formScore: Math.max(0, formScore),
    };
  }
}
