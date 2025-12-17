"""
Shoulder Shrug Exercise Tracker
===============================
Tracks shoulder shrug repetitions and provides form feedback.

Key landmarks used (half-body friendly):
- Shoulder (left/right)
- Ear (for reference height)

Rep counting logic:
- DOWN: shoulders relaxed
- UP: shoulders raised toward ears
"""

from typing import Dict, Tuple, List
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class ShoulderShrugTracker(BaseExerciseTracker):
    """
    Tracker for shoulder shrug exercise.
    
    Monitors shoulder height relative to ears to count reps.
    Very simple upper-body only exercise.
    """
    
    # Thresholds for shoulder-ear distance
    RELAXED_THRESHOLD = 0.12    # Shoulders relaxed
    SHRUGGED_THRESHOLD = 0.06   # Shoulders raised
    
    def __init__(self):
        """Initialize the shoulder shrug tracker."""
        super().__init__()
        self.stage = "down"
        self.distance_history: List[float] = []
        self.history_size = 5
        self.baseline_distance = None
        
    def get_required_landmarks(self) -> List[str]:
        """Get required landmarks - minimal upper body."""
        return [
            "left_shoulder",
            "right_shoulder",
            "left_ear",
            "right_ear"
        ]
    
    def process(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> ExerciseResult:
        """Process landmarks and track shoulder shrugs."""
        
        is_valid, missing = self.check_landmarks_visibility(landmarks)
        if not is_valid:
            return ExerciseResult(
                rep_count=self.rep_count,
                stage=self.stage,
                feedback=f"Can't see: {', '.join(missing)}",
                form_score=0,
                is_valid_pose=False
            )
        
        left_shoulder = landmarks["left_shoulder"]
        right_shoulder = landmarks["right_shoulder"]
        left_ear = landmarks["left_ear"]
        right_ear = landmarks["right_ear"]
        
        # Calculate average shoulder height
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Calculate average ear height
        avg_ear_y = (left_ear[1] + right_ear[1]) / 2
        
        # Distance from shoulder to ear (smaller = shrugged)
        shoulder_ear_distance = avg_shoulder_y - avg_ear_y
        
        # Set baseline on first valid frame
        if self.baseline_distance is None:
            self.baseline_distance = shoulder_ear_distance
        
        # Smooth
        self.distance_history.append(shoulder_ear_distance)
        if len(self.distance_history) > self.history_size:
            self.distance_history.pop(0)
        smoothed_distance = np.mean(self.distance_history)
        
        # Form checks
        penalties = []
        feedback_messages = []
        
        # Check shoulder symmetry
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_diff > 0.03:
            penalties.append(15)
            feedback_messages.append("Keep shoulders even! ⚖️")
        
        # Rep counting
        if smoothed_distance < self.SHRUGGED_THRESHOLD:
            if self.stage == "down":
                pass  # Wait for return
            self.stage = "up"
            if not feedback_messages:
                feedback_messages = ["Good squeeze! ✓"]
                
        elif smoothed_distance > self.RELAXED_THRESHOLD:
            if self.stage == "up":
                self.rep_count += 1
                if not penalties:
                    feedback_messages = ["Great shrug! 💪"]
            self.stage = "down"
        
        self.form_score = self._calculate_form_score(penalties)
        
        return ExerciseResult(
            rep_count=self.rep_count,
            stage=self.stage,
            feedback=feedback_messages[0] if feedback_messages else "Shrug shoulders up ⬆️",
            form_score=self.form_score,
            is_valid_pose=True,
            additional_data={"shoulder_ear_distance": smoothed_distance}
        )
    
    def reset(self):
        """Reset tracker."""
        super().reset()
        self.stage = "down"
        self.distance_history.clear()
        self.baseline_distance = None
    
    @property
    def exercise_name(self) -> str:
        return "Shoulder Shrug"
