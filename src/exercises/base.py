"""
Base Exercise Tracker
=====================
Abstract base class for all exercise trackers.

All exercise trackers should inherit from this class and implement
the required methods for consistent interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from enum import Enum


class ExerciseStage(Enum):
    """Enum for exercise stages."""
    UP = "up"
    DOWN = "down"
    STANDING = "standing"
    SQUAT = "squat"
    NEUTRAL = "neutral"


@dataclass
class ExerciseResult:
    """
    Data class for exercise tracking results.
    
    Attributes:
        rep_count: Number of completed repetitions
        stage: Current stage of the exercise (up/down/etc.)
        feedback: Current feedback message
        form_score: Form quality score (0-100)
        is_valid_pose: Whether the required landmarks are visible
        additional_data: Any additional exercise-specific data
    """
    rep_count: int
    stage: str
    feedback: str
    form_score: float
    is_valid_pose: bool = True
    additional_data: Dict = None
    
    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}


class BaseExerciseTracker(ABC):
    """
    Abstract base class for exercise trackers.
    
    All exercise trackers must implement:
    - process(): Process landmarks and return ExerciseResult
    - get_required_landmarks(): Return list of required landmark names
    - reset(): Reset the tracker state
    """
    
    def __init__(self):
        """Initialize base tracker."""
        self.rep_count = 0
        self.stage = "neutral"
        self.feedback = ""
        self.form_score = 100.0
        self.feedback_history: List[str] = []
        
    @abstractmethod
    def process(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> ExerciseResult:
        """
        Process landmarks and update exercise state.
        
        Args:
            landmarks: Dictionary of landmark name to (x, y, z, visibility)
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            ExerciseResult with current state
        """
        pass
    
    @abstractmethod
    def get_required_landmarks(self) -> List[str]:
        """
        Get list of landmark names required for this exercise.
        
        Returns:
            List of landmark names (e.g., ["left_shoulder", "left_elbow", "left_wrist"])
        """
        pass
    
    def reset(self):
        """Reset the tracker to initial state."""
        self.rep_count = 0
        self.stage = "neutral"
        self.feedback = ""
        self.form_score = 100.0
        self.feedback_history.clear()
        
    def check_landmarks_visibility(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        threshold: float = 0.5
    ) -> Tuple[bool, List[str]]:
        """
        Check if all required landmarks are visible.
        
        Args:
            landmarks: Dictionary of landmarks
            threshold: Minimum visibility threshold
            
        Returns:
            Tuple of (is_valid, list_of_missing_landmarks)
        """
        required = self.get_required_landmarks()
        missing = []
        
        for name in required:
            if name not in landmarks:
                missing.append(name)
            elif landmarks[name][3] < threshold:  # visibility is 4th element
                missing.append(name)
                
        return len(missing) == 0, missing
    
    def _add_feedback(self, feedback: str, is_positive: bool = False):
        """
        Add feedback message and track history.
        
        Args:
            feedback: Feedback message
            is_positive: Whether this is positive feedback
        """
        self.feedback = feedback
        if feedback and feedback not in self.feedback_history[-5:] if self.feedback_history else True:
            self.feedback_history.append(feedback)
            
    def _calculate_form_score(self, penalties: List[float]) -> float:
        """
        Calculate form score based on penalties.
        
        Args:
            penalties: List of penalty values (each 0-100)
            
        Returns:
            Form score (0-100)
        """
        if not penalties:
            return 100.0
        total_penalty = sum(penalties)
        return max(0.0, 100.0 - total_penalty)
    
    @property
    def exercise_name(self) -> str:
        """Get the exercise name. Override in subclasses."""
        return self.__class__.__name__.replace("Tracker", "")
