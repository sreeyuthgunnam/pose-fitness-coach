"""
Lateral Raise Exercise Tracker
==============================
Tracks lateral raise repetitions and provides form feedback.

Key landmarks used (half-body friendly):
- Shoulder (left/right)
- Elbow (left/right)  
- Wrist (left/right)
- Hip (for reference, optional)

Rep counting logic:
- DOWN: arms at sides (shoulder angle < 30°)
- UP: arms raised to shoulder level (shoulder angle > 70°)
"""

from typing import Dict, Tuple, List
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class LateralRaiseTracker(BaseExerciseTracker):
    """
    Tracker for lateral raise exercise.
    
    Monitors arm abduction angle to count reps.
    Works well with front-facing half-body camera view.
    """
    
    # Angle thresholds (angle from vertical)
    DOWN_ANGLE_THRESHOLD = 30    # Arms at sides
    UP_ANGLE_THRESHOLD = 70      # Arms raised to ~shoulder level
    
    def __init__(self, track_side: str = "both"):
        """
        Initialize the lateral raise tracker.
        
        Args:
            track_side: "left", "right", or "both" for tracking both arms
        """
        super().__init__()
        self.track_side = track_side.lower()
        self.stage = "down"
        self.angle_left = 0
        self.angle_right = 0
        self.angle_history: List[float] = []
        self.history_size = 5
        
    def get_required_landmarks(self) -> List[str]:
        """Get required landmarks."""
        if self.track_side == "both":
            return [
                "left_shoulder", "left_elbow", "left_wrist",
                "right_shoulder", "right_elbow", "right_wrist"
            ]
        side = self.track_side
        return [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
    
    def process(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> ExerciseResult:
        """Process landmarks and track lateral raise."""
        
        is_valid, missing = self.check_landmarks_visibility(landmarks)
        if not is_valid:
            return ExerciseResult(
                rep_count=self.rep_count,
                stage=self.stage,
                feedback=f"Can't see: {', '.join(missing)}",
                form_score=0,
                is_valid_pose=False
            )
        
        # Calculate angles for both arms or single arm
        if self.track_side == "both":
            self.angle_left = self._calculate_arm_angle(
                landmarks["left_shoulder"],
                landmarks["left_wrist"]
            )
            self.angle_right = self._calculate_arm_angle(
                landmarks["right_shoulder"],
                landmarks["right_wrist"]
            )
            # Use average of both arms
            current_angle = (self.angle_left + self.angle_right) / 2
        else:
            side = self.track_side
            current_angle = self._calculate_arm_angle(
                landmarks[f"{side}_shoulder"],
                landmarks[f"{side}_wrist"]
            )
        
        # Smooth the angle
        self.angle_history.append(current_angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
        smoothed_angle = np.mean(self.angle_history)
        
        # Form checks
        penalties = []
        feedback_messages = []
        
        # Check arm symmetry if tracking both
        if self.track_side == "both":
            angle_diff = abs(self.angle_left - self.angle_right)
            if angle_diff > 20:
                penalties.append(15)
                feedback_messages.append("Keep arms even! ⚖️")
        
        # Check if elbows are slightly bent (not locked)
        if self.track_side == "both":
            left_elbow_angle = self._calculate_elbow_angle(
                landmarks["left_shoulder"],
                landmarks["left_elbow"],
                landmarks["left_wrist"]
            )
            right_elbow_angle = self._calculate_elbow_angle(
                landmarks["right_shoulder"],
                landmarks["right_elbow"],
                landmarks["right_wrist"]
            )
            avg_elbow = (left_elbow_angle + right_elbow_angle) / 2
        else:
            side = self.track_side
            avg_elbow = self._calculate_elbow_angle(
                landmarks[f"{side}_shoulder"],
                landmarks[f"{side}_elbow"],
                landmarks[f"{side}_wrist"]
            )
        
        if avg_elbow > 175:
            penalties.append(10)
            feedback_messages.append("Slight bend in elbows 💪")
        
        # Rep counting
        if smoothed_angle > self.UP_ANGLE_THRESHOLD:
            if self.stage == "down":
                # Don't count yet - wait for return to down
                pass
            self.stage = "up"
            if not feedback_messages:
                feedback_messages = ["Good height! ✓"]
                
        elif smoothed_angle < self.DOWN_ANGLE_THRESHOLD:
            if self.stage == "up":
                self.rep_count += 1
                if not penalties:
                    feedback_messages = ["Great rep! 💪"]
            self.stage = "down"
        
        self.form_score = self._calculate_form_score(penalties)
        
        return ExerciseResult(
            rep_count=self.rep_count,
            stage=self.stage,
            feedback=feedback_messages[0] if feedback_messages else f"Angle: {int(smoothed_angle)}°",
            form_score=self.form_score,
            is_valid_pose=True,
            additional_data={
                "angle": smoothed_angle,
                "angle_left": self.angle_left,
                "angle_right": self.angle_right
            }
        )
    
    def _calculate_arm_angle(
        self,
        shoulder: Tuple[float, float, float, float],
        wrist: Tuple[float, float, float, float]
    ) -> float:
        """Calculate arm angle from vertical (0 = arm at side)."""
        # Vector from shoulder to wrist
        arm_vector = np.array([wrist[0] - shoulder[0], wrist[1] - shoulder[1]])
        # Vertical vector (pointing down)
        vertical = np.array([0, 1])
        
        cos_angle = np.dot(arm_vector, vertical) / (np.linalg.norm(arm_vector) * np.linalg.norm(vertical) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _calculate_elbow_angle(self, p1, p2, p3) -> float:
        """Calculate angle at elbow."""
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        ba = a - b
        bc = c - b
        
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def reset(self):
        """Reset tracker."""
        super().reset()
        self.stage = "down"
        self.angle_left = 0
        self.angle_right = 0
        self.angle_history.clear()
    
    @property
    def exercise_name(self) -> str:
        return "Lateral Raise"
