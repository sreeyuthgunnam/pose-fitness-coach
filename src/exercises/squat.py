"""
Squat Exercise Tracker
======================
Tracks squat repetitions and provides form feedback.

Key landmarks used:
- Hip (left/right)
- Knee (left/right)
- Ankle (left/right)
- Shoulder (for back angle check)

Rep counting logic:
- STANDING: knee angle > 160°
- SQUAT: knee angle < 90°
- Count increments on complete standing→squat→standing cycle
"""

from typing import Dict, Tuple, List
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class SquatTracker(BaseExerciseTracker):
    """
    Tracker for squat exercise.
    
    Monitors knee and hip angles to count reps and checks form including:
    - Knee tracking over toes
    - Back angle (torso upright)
    - Squat depth
    """
    
    # Angle thresholds
    STANDING_KNEE_THRESHOLD = 160  # Standing position
    SQUAT_KNEE_THRESHOLD = 90      # Full squat depth
    
    # Form check thresholds
    KNEE_OVER_TOE_THRESHOLD = 0.05  # How far knee can go past ankle (normalized)
    MIN_BACK_ANGLE = 45             # Minimum acceptable back angle from vertical
    
    def __init__(self, track_side: str = "left"):
        """
        Initialize the squat tracker.
        
        Args:
            track_side: Which side to track ("left" or "right")
        """
        super().__init__()
        self.track_side = track_side.lower()
        
        # State tracking
        self.stage = "standing"
        self.knee_angle = 0
        self.hip_angle = 0
        
        # Track deepest squat in current rep
        self.deepest_angle = 180
        self.reached_depth = False
        
        # Smoothing
        self.angle_history: List[float] = []
        self.history_size = 5
        
    def get_required_landmarks(self) -> List[str]:
        """Get required landmarks for squat tracking."""
        side = self.track_side
        return [
            f"{side}_shoulder",
            f"{side}_hip",
            f"{side}_knee",
            f"{side}_ankle"
        ]
    
    def process(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> ExerciseResult:
        """
        Process landmarks and track squat.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            ExerciseResult with rep count, stage, feedback, and form score
        """
        # Check if required landmarks are visible
        is_valid, missing = self.check_landmarks_visibility(landmarks)
        
        if not is_valid:
            return ExerciseResult(
                rep_count=self.rep_count,
                stage=self.stage,
                feedback=f"Can't see: {', '.join(missing)}",
                form_score=0,
                is_valid_pose=False
            )
        
        # Get landmark positions
        side = self.track_side
        shoulder = landmarks[f"{side}_shoulder"]
        hip = landmarks[f"{side}_hip"]
        knee = landmarks[f"{side}_knee"]
        ankle = landmarks[f"{side}_ankle"]
        
        # Calculate angles
        self.knee_angle = self._calculate_angle(hip, knee, ankle)
        self.hip_angle = self._calculate_angle(shoulder, hip, knee)
        
        # Smooth the knee angle
        self._smooth_angle()
        smoothed_knee_angle = np.mean(self.angle_history) if self.angle_history else self.knee_angle
        
        # Track deepest point
        if smoothed_knee_angle < self.deepest_angle:
            self.deepest_angle = smoothed_knee_angle
            
        # Form checks
        penalties = []
        feedback_messages = []
        
        # Check 1: Knees over toes
        knee_penalty, knee_feedback = self._check_knee_position(knee, ankle)
        if knee_penalty > 0:
            penalties.append(knee_penalty)
            feedback_messages.append(knee_feedback)
            
        # Check 2: Back angle
        back_penalty, back_feedback = self._check_back_angle(shoulder, hip, knee)
        if back_penalty > 0:
            penalties.append(back_penalty)
            feedback_messages.append(back_feedback)
            
        # Check 3: Depth feedback
        depth_feedback = self._check_depth(smoothed_knee_angle)
        
        # Update rep counting
        prev_stage = self.stage
        
        if smoothed_knee_angle > self.STANDING_KNEE_THRESHOLD:
            # Standing position
            if self.stage == "squat":
                # Completed a rep (was squat, now standing)
                self.rep_count += 1
                if self.reached_depth:
                    if not penalties:
                        feedback_messages = ["Great squat! 💪"]
                else:
                    feedback_messages.insert(0, "Go deeper next time! ⬇️")
                # Reset for next rep
                self.deepest_angle = 180
                self.reached_depth = False
            self.stage = "standing"
            
        elif smoothed_knee_angle < self.SQUAT_KNEE_THRESHOLD:
            # Squat position
            self.stage = "squat"
            self.reached_depth = True
            if not feedback_messages:
                feedback_messages = ["Great depth! ✓"]
        else:
            # In between - transitioning
            if self.stage == "standing":
                # Going down
                if not feedback_messages and not depth_feedback:
                    feedback_messages = ["Keep going down... ⬇️"]
        
        # Calculate form score
        self.form_score = self._calculate_form_score(penalties)
        
        # Combine feedback
        if feedback_messages:
            self.feedback = feedback_messages[0]
        elif depth_feedback:
            self.feedback = depth_feedback
        else:
            self.feedback = f"Knee: {int(smoothed_knee_angle)}°"
        
        return ExerciseResult(
            rep_count=self.rep_count,
            stage=self.stage,
            feedback=self.feedback,
            form_score=self.form_score,
            is_valid_pose=True,
            additional_data={
                "knee_angle": smoothed_knee_angle,
                "hip_angle": self.hip_angle,
                "deepest_angle": self.deepest_angle,
                "reached_depth": self.reached_depth
            }
        )
    
    def _calculate_angle(
        self,
        point1: Tuple[float, float, float, float],
        point2: Tuple[float, float, float, float],
        point3: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculate angle at point2 between point1 and point3.
        
        Args:
            point1: First point
            point2: Vertex point
            point3: Third point
            
        Returns:
            Angle in degrees
        """
        a = np.array([point1[0], point1[1]])
        b = np.array([point2[0], point2[1]])
        c = np.array([point3[0], point3[1]])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    
    def _smooth_angle(self):
        """Add current knee angle to history for smoothing."""
        self.angle_history.append(self.knee_angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
    
    def _check_knee_position(
        self,
        knee: Tuple[float, float, float, float],
        ankle: Tuple[float, float, float, float]
    ) -> Tuple[float, str]:
        """
        Check if knees are going too far over toes.
        
        Args:
            knee: Knee landmark
            ankle: Ankle landmark
            
        Returns:
            Tuple of (penalty, feedback_message)
        """
        # Compare x positions (horizontal)
        # In a side view, if knee x is significantly past ankle x, it's over toes
        knee_x = knee[0]
        ankle_x = ankle[0]
        
        # Determine which direction is "forward" based on which side we're tracking
        if self.track_side == "left":
            # Left side view: knee going left of ankle is forward
            overshoot = ankle_x - knee_x
        else:
            # Right side view: knee going right of ankle is forward
            overshoot = knee_x - ankle_x
        
        if overshoot > self.KNEE_OVER_TOE_THRESHOLD:
            penalty = min(35, overshoot * 300)
            return penalty, "Knees going over toes! 🦵"
        
        return 0, ""
    
    def _check_back_angle(
        self,
        shoulder: Tuple[float, float, float, float],
        hip: Tuple[float, float, float, float],
        knee: Tuple[float, float, float, float]
    ) -> Tuple[float, str]:
        """
        Check if back is staying relatively upright.
        
        Args:
            shoulder: Shoulder landmark
            hip: Hip landmark
            knee: Knee landmark (for reference)
            
        Returns:
            Tuple of (penalty, feedback_message)
        """
        # Calculate torso angle from vertical
        # Vertical line is straight up (0, -1)
        shoulder_pos = np.array([shoulder[0], shoulder[1]])
        hip_pos = np.array([hip[0], hip[1]])
        
        torso_vector = shoulder_pos - hip_pos
        vertical = np.array([0, -1])  # Up direction
        
        # Calculate angle from vertical
        cos_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        back_angle = np.degrees(np.arccos(cos_angle))
        
        # Only check when in squat position
        if self.stage == "squat" or self.knee_angle < 120:
            if back_angle > self.MIN_BACK_ANGLE:
                penalty = min(30, (back_angle - self.MIN_BACK_ANGLE) * 1.5)
                return penalty, "Keep your back straight! 🔙"
        
        return 0, ""
    
    def _check_depth(self, knee_angle: float) -> str:
        """
        Provide feedback on squat depth.
        
        Args:
            knee_angle: Current knee angle
            
        Returns:
            Feedback message
        """
        if self.stage == "squat":
            if knee_angle > self.SQUAT_KNEE_THRESHOLD:
                return "Go deeper! ⬇️"
            else:
                return ""
        elif self.stage == "standing" and self.knee_angle < 150:
            return "Going down... 👇"
        return ""
    
    def reset(self):
        """Reset tracker state."""
        super().reset()
        self.stage = "standing"
        self.knee_angle = 0
        self.hip_angle = 0
        self.deepest_angle = 180
        self.reached_depth = False
        self.angle_history.clear()
    
    @property
    def exercise_name(self) -> str:
        """Get exercise name."""
        return "Squat"


# Testing
if __name__ == "__main__":
    tracker = SquatTracker(track_side="left")
    
    # Simulate a squat motion
    test_cases = [
        # Standing position
        {
            "left_shoulder": (0.5, 0.2, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9),
            "left_knee": (0.5, 0.7, 0, 0.9),
            "left_ankle": (0.5, 0.9, 0, 0.9)
        },
        # Going down
        {
            "left_shoulder": (0.5, 0.25, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9),
            "left_knee": (0.45, 0.65, 0, 0.9),
            "left_ankle": (0.5, 0.9, 0, 0.9)
        },
        # Full squat
        {
            "left_shoulder": (0.5, 0.35, 0, 0.9),
            "left_hip": (0.5, 0.55, 0, 0.9),
            "left_knee": (0.45, 0.6, 0, 0.9),
            "left_ankle": (0.5, 0.9, 0, 0.9)
        },
        # Back to standing
        {
            "left_shoulder": (0.5, 0.2, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9),
            "left_knee": (0.5, 0.7, 0, 0.9),
            "left_ankle": (0.5, 0.9, 0, 0.9)
        }
    ]
    
    print("Testing SquatTracker:")
    print("-" * 50)
    
    for i, landmarks in enumerate(test_cases):
        result = tracker.process(landmarks, 640, 480)
        print(f"Frame {i + 1}:")
        print(f"  Reps: {result.rep_count}")
        print(f"  Stage: {result.stage}")
        print(f"  Feedback: {result.feedback}")
        print(f"  Form Score: {result.form_score:.1f}%")
        if result.additional_data:
            print(f"  Knee Angle: {result.additional_data.get('knee_angle', 0):.1f}°")
        print()
