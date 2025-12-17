"""
Shoulder Press Exercise Tracker
===============================
Tracks shoulder press repetitions and provides form feedback.

Key landmarks used (half-body friendly):
- Shoulder (left/right)
- Elbow (left/right)
- Wrist (left/right)

Rep counting logic:
- DOWN: arms at shoulder level (elbow angle ~90°)
- UP: arms extended overhead (elbow angle >160°)
"""

from typing import Dict, Tuple, List
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class ShoulderPressTracker(BaseExerciseTracker):
    """
    Tracker for shoulder press exercise.
    
    Monitors elbow angle and wrist position to count reps.
    Works well with half-body camera view.
    """
    
    # Angle thresholds
    DOWN_ANGLE_THRESHOLD = 100   # Arms at shoulder level
    UP_ANGLE_THRESHOLD = 160     # Arms extended overhead
    
    def __init__(self, track_side: str = "left"):
        """Initialize the shoulder press tracker."""
        super().__init__()
        self.track_side = track_side.lower()
        self.stage = "down"
        self.angle = 0
        self.angle_history: List[float] = []
        self.history_size = 5
        
    def get_required_landmarks(self) -> List[str]:
        """Get required landmarks - only upper body needed."""
        side = self.track_side
        return [
            f"{side}_shoulder",
            f"{side}_elbow",
            f"{side}_wrist"
        ]
    
    def process(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> ExerciseResult:
        """Process landmarks and track shoulder press."""
        
        is_valid, missing = self.check_landmarks_visibility(landmarks)
        if not is_valid:
            return ExerciseResult(
                rep_count=self.rep_count,
                stage=self.stage,
                feedback=f"Can't see: {', '.join(missing)}",
                form_score=0,
                is_valid_pose=False
            )
        
        side = self.track_side
        shoulder = landmarks[f"{side}_shoulder"]
        elbow = landmarks[f"{side}_elbow"]
        wrist = landmarks[f"{side}_wrist"]
        
        # Calculate elbow angle
        self.angle = self._calculate_angle(shoulder, elbow, wrist)
        
        # Smooth the angle
        self._smooth_angle()
        smoothed_angle = np.mean(self.angle_history) if self.angle_history else self.angle
        
        # Form checks
        penalties = []
        feedback_messages = []
        
        # Check if wrist is above shoulder (proper overhead position)
        if self.stage == "up" or smoothed_angle > 140:
            if wrist[1] > shoulder[1]:  # wrist below shoulder (y increases downward)
                penalties.append(20)
                feedback_messages.append("Push higher! Arms overhead ⬆️")
        
        # Check elbow position - shouldn't flare too wide
        elbow_shoulder_dist = abs(elbow[0] - shoulder[0])
        if elbow_shoulder_dist > 0.2:
            penalties.append(15)
            feedback_messages.append("Keep elbows in front 💪")
        
        # Rep counting
        if smoothed_angle > self.UP_ANGLE_THRESHOLD:
            if self.stage == "down":
                self.rep_count += 1
                if not penalties:
                    feedback_messages = ["Great press! 💪"]
            self.stage = "up"
            if not feedback_messages:
                feedback_messages = ["Good lockout! ✓"]
                
        elif smoothed_angle < self.DOWN_ANGLE_THRESHOLD:
            self.stage = "down"
            if not feedback_messages:
                feedback_messages = ["Ready to press ⬆️"]
        
        self.form_score = self._calculate_form_score(penalties)
        
        return ExerciseResult(
            rep_count=self.rep_count,
            stage=self.stage,
            feedback=feedback_messages[0] if feedback_messages else f"Angle: {int(smoothed_angle)}°",
            form_score=self.form_score,
            is_valid_pose=True,
            additional_data={"angle": smoothed_angle}
        )
    
    def _calculate_angle(self, p1, p2, p3) -> float:
        """Calculate angle at p2."""
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        ba = a - b
        bc = c - b
        
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _smooth_angle(self):
        """Smooth angle values."""
        self.angle_history.append(self.angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
    
    def reset(self):
        """Reset tracker."""
        super().reset()
        self.stage = "down"
        self.angle = 0
        self.angle_history.clear()
    
    @property
    def exercise_name(self) -> str:
        return "Shoulder Press"
