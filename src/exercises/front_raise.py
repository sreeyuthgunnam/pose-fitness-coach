"""
Front Raise Exercise Tracker
============================
Tracks front raise repetitions and provides form feedback.

Key landmarks used (half-body friendly):
- Shoulder (left/right)
- Elbow (left/right)
- Wrist (left/right)

Rep counting logic:
- DOWN: arms at sides
- UP: arms raised in front to shoulder level
"""

from typing import Dict, Tuple, List
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class FrontRaiseTracker(BaseExerciseTracker):
    """
    Tracker for front raise exercise.
    
    Monitors arm position relative to torso to count reps.
    Works well with side-view half-body camera.
    """
    
    # Position thresholds (wrist relative to shoulder)
    DOWN_THRESHOLD = 0.1    # Wrist near hip level
    UP_THRESHOLD = 0.05     # Wrist at shoulder level
    
    def __init__(self, track_side: str = "left"):
        """Initialize the front raise tracker."""
        super().__init__()
        self.track_side = track_side.lower()
        self.stage = "down"
        self.wrist_height_history: List[float] = []
        self.history_size = 5
        
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
        """Process landmarks and track front raise."""
        
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
        
        # Calculate wrist height relative to shoulder
        # Negative = wrist above shoulder, Positive = wrist below
        wrist_relative_height = wrist[1] - shoulder[1]
        
        # Smooth
        self.wrist_height_history.append(wrist_relative_height)
        if len(self.wrist_height_history) > self.history_size:
            self.wrist_height_history.pop(0)
        smoothed_height = np.mean(self.wrist_height_history)
        
        # Calculate elbow angle
        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
        
        # Form checks
        penalties = []
        feedback_messages = []
        
        # Check elbow is relatively straight
        if elbow_angle < 150:
            penalties.append(15)
            feedback_messages.append("Keep arms straighter 💪")
        
        # Check wrist doesn't go above shoulder too much
        if smoothed_height < -0.1:
            penalties.append(10)
            feedback_messages.append("Stop at shoulder level ⏸️")
        
        # Rep counting based on wrist height
        if smoothed_height < self.UP_THRESHOLD:  # Wrist at or above shoulder
            if self.stage == "down":
                pass  # Don't count until return
            self.stage = "up"
            if not feedback_messages:
                feedback_messages = ["Good height! ✓"]
                
        elif smoothed_height > self.DOWN_THRESHOLD + 0.15:  # Wrist well below shoulder
            if self.stage == "up":
                self.rep_count += 1
                if not penalties:
                    feedback_messages = ["Great rep! 💪"]
            self.stage = "down"
        
        self.form_score = self._calculate_form_score(penalties)
        
        return ExerciseResult(
            rep_count=self.rep_count,
            stage=self.stage,
            feedback=feedback_messages[0] if feedback_messages else f"Height: {-smoothed_height:.2f}",
            form_score=self.form_score,
            is_valid_pose=True,
            additional_data={
                "wrist_height": smoothed_height,
                "elbow_angle": elbow_angle
            }
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
        self.wrist_height_history.clear()
    
    @property
    def exercise_name(self) -> str:
        return "Front Raise"
