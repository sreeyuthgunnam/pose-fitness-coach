"""
Tricep Extension Exercise Tracker
=================================
Tracks overhead tricep extension repetitions and provides form feedback.

Key landmarks used (half-body friendly):
- Shoulder (left/right)
- Elbow (left/right)
- Wrist (left/right)

Rep counting logic:
- DOWN: elbow bent, hand behind head (angle < 60°)
- UP: arm extended overhead (angle > 150°)
"""

from typing import Dict, Tuple, List
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class TricepExtensionTracker(BaseExerciseTracker):
    """
    Tracker for overhead tricep extension exercise.
    
    Monitors elbow angle while keeping upper arm stationary.
    Works well with side-view half-body camera.
    """
    
    # Angle thresholds
    DOWN_ANGLE_THRESHOLD = 60    # Elbow fully bent
    UP_ANGLE_THRESHOLD = 150     # Arm extended
    
    def __init__(self, track_side: str = "left"):
        """Initialize the tricep extension tracker."""
        super().__init__()
        self.track_side = track_side.lower()
        self.stage = "down"
        self.angle = 0
        self.angle_history: List[float] = []
        self.history_size = 5
        self.initial_elbow_pos = None
        
    def get_required_landmarks(self) -> List[str]:
        """Get required landmarks."""
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
        """Process landmarks and track tricep extension."""
        
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
        
        # Check if elbow is above shoulder (overhead position)
        if elbow[1] > shoulder[1] + 0.1:  # Elbow below shoulder
            return ExerciseResult(
                rep_count=self.rep_count,
                stage=self.stage,
                feedback="Raise elbow overhead! ⬆️",
                form_score=50,
                is_valid_pose=True
            )
        
        # Calculate elbow angle
        self.angle = self._calculate_angle(shoulder, elbow, wrist)
        
        # Smooth
        self.angle_history.append(self.angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
        smoothed_angle = np.mean(self.angle_history)
        
        # Set initial elbow position
        if self.initial_elbow_pos is None:
            self.initial_elbow_pos = (elbow[0], elbow[1])
        
        # Form checks
        penalties = []
        feedback_messages = []
        
        # Check elbow stays relatively stationary
        elbow_drift = abs(elbow[0] - self.initial_elbow_pos[0])
        if elbow_drift > 0.08:
            penalties.append(20)
            feedback_messages.append("Keep elbow still! 📍")
            # Update reference slowly
            self.initial_elbow_pos = (
                self.initial_elbow_pos[0] * 0.9 + elbow[0] * 0.1,
                self.initial_elbow_pos[1] * 0.9 + elbow[1] * 0.1
            )
        
        # Rep counting
        if smoothed_angle > self.UP_ANGLE_THRESHOLD:
            if self.stage == "down":
                self.rep_count += 1
                if not penalties:
                    feedback_messages = ["Great extension! 💪"]
            self.stage = "up"
            if not feedback_messages:
                feedback_messages = ["Full extension! ✓"]
                
        elif smoothed_angle < self.DOWN_ANGLE_THRESHOLD:
            self.stage = "down"
            if not feedback_messages:
                feedback_messages = ["Good stretch 👍"]
        
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
    
    def reset(self):
        """Reset tracker."""
        super().reset()
        self.stage = "down"
        self.angle = 0
        self.angle_history.clear()
        self.initial_elbow_pos = None
    
    @property
    def exercise_name(self) -> str:
        return "Tricep Extension"
