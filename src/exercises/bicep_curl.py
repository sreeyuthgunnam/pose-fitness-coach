"""
Bicep Curl Exercise Tracker
===========================
Tracks bicep curl repetitions and provides form feedback.

Key landmarks used:
- Shoulder (left/right)
- Elbow (left/right)
- Wrist (left/right)
- Hip (for torso stability check)

Rep counting logic:
- DOWN: arm extended (elbow angle > 160°)
- UP: arm curled (elbow angle < 40°)
- Count increments on complete down→up→down cycle
"""

from typing import Dict, Tuple, List, Optional
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class BicepCurlTracker(BaseExerciseTracker):
    """
    Tracker for bicep curl exercise.
    
    Monitors elbow angle to count reps and checks form including:
    - Elbow stability (shouldn't move too far from torso)
    - Full range of motion
    - Body stability (no swinging)
    """
    
    # Angle thresholds
    DOWN_ANGLE_THRESHOLD = 160  # Arm extended
    UP_ANGLE_THRESHOLD = 40     # Arm curled
    
    # Form check thresholds
    ELBOW_DRIFT_THRESHOLD = 0.15  # Maximum horizontal drift of elbow (relative to frame width)
    SHOULDER_STABILITY_THRESHOLD = 0.05  # Maximum shoulder movement (relative to frame)
    
    def __init__(self, track_side: str = "left"):
        """
        Initialize the bicep curl tracker.
        
        Args:
            track_side: Which arm to track ("left" or "right")
        """
        super().__init__()
        self.track_side = track_side.lower()
        
        # State tracking
        self.stage = "down"
        self.angle = 0
        
        # Form tracking
        self.initial_elbow_x = None
        self.initial_shoulder_y = None
        self.elbow_stable = True
        self.body_stable = True
        
        # Smoothing
        self.angle_history: List[float] = []
        self.history_size = 5
        
    def get_required_landmarks(self) -> List[str]:
        """Get required landmarks for bicep curl tracking."""
        side = self.track_side
        return [
            f"{side}_shoulder",
            f"{side}_elbow",
            f"{side}_wrist",
            f"{side}_hip"
        ]
    
    def process(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> ExerciseResult:
        """
        Process landmarks and track bicep curl.
        
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
        elbow = landmarks[f"{side}_elbow"]
        wrist = landmarks[f"{side}_wrist"]
        hip = landmarks[f"{side}_hip"]
        
        # Calculate elbow angle
        self.angle = self._calculate_angle(shoulder, elbow, wrist)
        
        # Smooth the angle
        self._smooth_angle()
        smoothed_angle = np.mean(self.angle_history) if self.angle_history else self.angle
        
        # Form checks
        penalties = []
        feedback_messages = []
        
        # Check 1: Elbow stability
        elbow_penalty, elbow_feedback = self._check_elbow_stability(
            elbow, hip, frame_width
        )
        if elbow_penalty > 0:
            penalties.append(elbow_penalty)
            feedback_messages.append(elbow_feedback)
            
        # Check 2: Shoulder/body stability
        shoulder_penalty, shoulder_feedback = self._check_shoulder_stability(
            shoulder, frame_height
        )
        if shoulder_penalty > 0:
            penalties.append(shoulder_penalty)
            feedback_messages.append(shoulder_feedback)
            
        # Check 3: Range of motion feedback
        rom_feedback = self._check_range_of_motion(smoothed_angle)
        
        # Update rep counting
        prev_stage = self.stage
        
        if smoothed_angle > self.DOWN_ANGLE_THRESHOLD:
            # Arm is extended (down position)
            if self.stage == "up":
                # Completed a rep (was up, now down)
                self.rep_count += 1
                if not penalties:
                    feedback_messages = ["Great rep! 💪"]
            self.stage = "down"
            
        elif smoothed_angle < self.UP_ANGLE_THRESHOLD:
            # Arm is curled (up position)
            self.stage = "up"
            if not feedback_messages:
                feedback_messages = ["Full range of motion! ✓"]
        
        # Calculate form score
        self.form_score = self._calculate_form_score(penalties)
        
        # Combine feedback
        if feedback_messages:
            self.feedback = feedback_messages[0]  # Show most important feedback
        elif rom_feedback:
            self.feedback = rom_feedback
        else:
            self.feedback = f"Angle: {int(smoothed_angle)}°"
        
        return ExerciseResult(
            rep_count=self.rep_count,
            stage=self.stage,
            feedback=self.feedback,
            form_score=self.form_score,
            is_valid_pose=True,
            additional_data={
                "angle": smoothed_angle,
                "elbow_stable": len([p for p in penalties if "elbow" in str(p)]) == 0,
                "body_stable": len([p for p in penalties if "body" in str(p)]) == 0
            }
        )
    
    def _calculate_angle(
        self,
        point1: Tuple[float, float, float, float],
        point2: Tuple[float, float, float, float],
        point3: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculate angle at point2 (elbow) between point1 (shoulder) and point3 (wrist).
        
        Args:
            point1: Shoulder coordinates
            point2: Elbow coordinates  
            point3: Wrist coordinates
            
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
        """Add current angle to history for smoothing."""
        self.angle_history.append(self.angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
    
    def _check_elbow_stability(
        self,
        elbow: Tuple[float, float, float, float],
        hip: Tuple[float, float, float, float],
        frame_width: int
    ) -> Tuple[float, str]:
        """
        Check if elbow stays close to the torso.
        
        Args:
            elbow: Elbow landmark
            hip: Hip landmark
            frame_width: Frame width for normalization
            
        Returns:
            Tuple of (penalty, feedback_message)
        """
        # Initialize reference position
        if self.initial_elbow_x is None:
            self.initial_elbow_x = elbow[0]
            return 0, ""
        
        # Calculate horizontal drift from hip
        elbow_x = elbow[0]
        hip_x = hip[0]
        
        # Distance from hip (normalized)
        drift = abs(elbow_x - hip_x)
        
        if drift > self.ELBOW_DRIFT_THRESHOLD:
            self.elbow_stable = False
            penalty = min(30, (drift - self.ELBOW_DRIFT_THRESHOLD) * 200)
            return penalty, "Keep your elbow close to your body! 📍"
        
        self.elbow_stable = True
        return 0, ""
    
    def _check_shoulder_stability(
        self,
        shoulder: Tuple[float, float, float, float],
        frame_height: int
    ) -> Tuple[float, str]:
        """
        Check if shoulder/body is stable (no swinging).
        
        Args:
            shoulder: Shoulder landmark
            frame_height: Frame height for normalization
            
        Returns:
            Tuple of (penalty, feedback_message)
        """
        if self.initial_shoulder_y is None:
            self.initial_shoulder_y = shoulder[1]
            return 0, ""
        
        # Check vertical movement of shoulder
        shoulder_drift = abs(shoulder[1] - self.initial_shoulder_y)
        
        if shoulder_drift > self.SHOULDER_STABILITY_THRESHOLD:
            self.body_stable = False
            penalty = min(25, shoulder_drift * 300)
            # Update reference to prevent continuous penalty
            self.initial_shoulder_y = shoulder[1] * 0.1 + self.initial_shoulder_y * 0.9
            return penalty, "Don't swing your body! 🚫"
        
        self.body_stable = True
        # Slowly update reference
        self.initial_shoulder_y = shoulder[1] * 0.05 + self.initial_shoulder_y * 0.95
        return 0, ""
    
    def _check_range_of_motion(self, angle: float) -> str:
        """
        Provide feedback on range of motion.
        
        Args:
            angle: Current elbow angle
            
        Returns:
            Feedback message
        """
        if self.stage == "up" and angle > 50:
            return "Curl higher for full ROM! ⬆️"
        elif self.stage == "down" and angle < 150:
            return "Extend arm fully! ⬇️"
        return ""
    
    def reset(self):
        """Reset tracker state."""
        super().reset()
        self.stage = "down"
        self.angle = 0
        self.initial_elbow_x = None
        self.initial_shoulder_y = None
        self.elbow_stable = True
        self.body_stable = True
        self.angle_history.clear()
    
    @property
    def exercise_name(self) -> str:
        """Get exercise name."""
        return "Bicep Curl"


# Testing
if __name__ == "__main__":
    # Create mock landmark data for testing
    tracker = BicepCurlTracker(track_side="left")
    
    # Simulate a curl motion
    test_cases = [
        # Extended arm (down position)
        {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.3, 0.7, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        },
        # Partially curled
        {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.35, 0.4, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        },
        # Fully curled (up position)
        {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.32, 0.32, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        },
        # Back to extended
        {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.3, 0.7, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        }
    ]
    
    print("Testing BicepCurlTracker:")
    print("-" * 50)
    
    for i, landmarks in enumerate(test_cases):
        result = tracker.process(landmarks, 640, 480)
        print(f"Frame {i + 1}:")
        print(f"  Reps: {result.rep_count}")
        print(f"  Stage: {result.stage}")
        print(f"  Feedback: {result.feedback}")
        print(f"  Form Score: {result.form_score:.1f}%")
        print()
