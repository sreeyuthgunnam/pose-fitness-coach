"""
Pose Validation Utilities
=========================
Utilities for validating pose detection results and handling edge cases.

Includes:
- Landmark visibility validation
- Pose completeness checks
- User positioning feedback
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class VisibilityStatus(Enum):
    """Status of landmark visibility."""
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    NOT_VISIBLE = "not_visible"


@dataclass
class ValidationResult:
    """Result of pose validation."""
    is_valid: bool
    confidence: float
    feedback: str
    missing_landmarks: List[str]
    suggestions: List[str]


class PoseValidator:
    """
    Validates pose detection results and provides user feedback.
    
    Handles edge cases like:
    - Person not fully visible
    - Low confidence landmarks
    - Multiple people in frame
    - Poor lighting conditions
    """
    
    # Body region landmark groups
    BODY_REGIONS = {
        'upper_body': [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist'
        ],
        'lower_body': [
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ],
        'face': [
            'nose', 'left_eye', 'right_eye',
            'left_ear', 'right_ear'
        ],
        'torso': [
            'left_shoulder', 'right_shoulder',
            'left_hip', 'right_hip'
        ]
    }
    
    # Exercise-specific required landmarks
    EXERCISE_LANDMARKS = {
        'bicep_curl': ['shoulder', 'elbow', 'wrist', 'hip'],
        'squat': ['shoulder', 'hip', 'knee', 'ankle'],
        'pushup': ['shoulder', 'elbow', 'wrist', 'hip', 'ankle']
    }
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the pose validator.
        
        Args:
            confidence_threshold: Minimum confidence for landmark validity
        """
        self.confidence_threshold = confidence_threshold
    
    def is_pose_valid(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        confidence_threshold: Optional[float] = None
    ) -> bool:
        """
        Check if the pose detection is valid (enough landmarks visible).
        
        Args:
            landmarks: Dictionary of landmark name to (x, y, z, visibility)
            confidence_threshold: Override default confidence threshold
            
        Returns:
            True if pose is valid
        """
        if not landmarks:
            return False
        
        threshold = confidence_threshold or self.confidence_threshold
        
        # Count visible landmarks
        visible_count = sum(
            1 for lm in landmarks.values()
            if lm[3] >= threshold  # visibility is 4th element
        )
        
        # Require at least 50% of landmarks to be visible
        return visible_count >= len(landmarks) * 0.5
    
    def validate_for_exercise(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        exercise: str,
        side: str = 'left'
    ) -> ValidationResult:
        """
        Validate pose for a specific exercise.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            exercise: Exercise name
            side: Which side to check ('left' or 'right')
            
        Returns:
            ValidationResult with details
        """
        if exercise not in self.EXERCISE_LANDMARKS:
            return ValidationResult(
                is_valid=False,
                confidence=0,
                feedback="Unknown exercise",
                missing_landmarks=[],
                suggestions=["Select a valid exercise"]
            )
        
        required_parts = self.EXERCISE_LANDMARKS[exercise]
        required_landmarks = [f"{side}_{part}" for part in required_parts]
        
        missing = []
        low_confidence = []
        total_visibility = 0
        
        for lm_name in required_landmarks:
            if lm_name not in landmarks:
                missing.append(lm_name)
            else:
                visibility = landmarks[lm_name][3]
                total_visibility += visibility
                if visibility < self.confidence_threshold:
                    low_confidence.append(lm_name)
        
        # Calculate overall confidence
        if required_landmarks:
            confidence = total_visibility / len(required_landmarks)
        else:
            confidence = 0
        
        # Generate feedback and suggestions
        feedback, suggestions = self._generate_feedback(
            missing, low_confidence, exercise
        )
        
        is_valid = len(missing) == 0 and len(low_confidence) <= 1
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            feedback=feedback,
            missing_landmarks=missing + low_confidence,
            suggestions=suggestions
        )
    
    def get_visibility_feedback(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]]
    ) -> str:
        """
        Get feedback about user positioning based on landmark visibility.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Positioning feedback message
        """
        if not landmarks:
            return "No person detected - step into frame"
        
        # Check different body regions
        upper_visible = self._check_region_visibility(landmarks, 'upper_body')
        lower_visible = self._check_region_visibility(landmarks, 'lower_body')
        face_visible = self._check_region_visibility(landmarks, 'face')
        
        # Determine positioning issue
        if not face_visible:
            return "Move back - can't see your head"
        
        if not upper_visible:
            return "Move back - can't see upper body"
        
        if not lower_visible:
            return "Move back - can't see your legs"
        
        # Check if person is centered
        center_feedback = self._check_centering(landmarks)
        if center_feedback:
            return center_feedback
        
        # Check if person is at appropriate distance
        distance_feedback = self._check_distance(landmarks)
        if distance_feedback:
            return distance_feedback
        
        return "Position: Good ✓"
    
    def _check_region_visibility(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        region: str
    ) -> bool:
        """Check if a body region is visible."""
        if region not in self.BODY_REGIONS:
            return True
        
        region_landmarks = self.BODY_REGIONS[region]
        visible_count = 0
        
        for lm_name in region_landmarks:
            if lm_name in landmarks:
                if landmarks[lm_name][3] >= self.confidence_threshold:
                    visible_count += 1
        
        # Require at least half of region landmarks to be visible
        return visible_count >= len(region_landmarks) * 0.5
    
    def _check_centering(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]]
    ) -> Optional[str]:
        """Check if person is centered in frame."""
        # Get torso center
        torso_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        x_coords = []
        
        for lm_name in torso_landmarks:
            if lm_name in landmarks and landmarks[lm_name][3] >= 0.3:
                x_coords.append(landmarks[lm_name][0])
        
        if not x_coords:
            return None
        
        center_x = sum(x_coords) / len(x_coords)
        
        # Check if center is within middle 60% of frame
        if center_x < 0.2:
            return "Move right - you're too far left"
        elif center_x > 0.8:
            return "Move left - you're too far right"
        
        return None
    
    def _check_distance(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]]
    ) -> Optional[str]:
        """Check if person is at appropriate distance from camera."""
        # Estimate body size using shoulder width
        if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
            left = landmarks['left_shoulder']
            right = landmarks['right_shoulder']
            
            if left[3] >= 0.5 and right[3] >= 0.5:
                shoulder_width = abs(right[0] - left[0])
                
                if shoulder_width > 0.6:
                    return "Move back - you're too close"
                elif shoulder_width < 0.1:
                    return "Move closer - you're too far"
        
        return None
    
    def _generate_feedback(
        self,
        missing: List[str],
        low_confidence: List[str],
        exercise: str
    ) -> Tuple[str, List[str]]:
        """Generate feedback message and suggestions."""
        suggestions = []
        
        if missing:
            # Determine which body part is missing
            missing_parts = set()
            for lm in missing:
                for part in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']:
                    if part in lm:
                        missing_parts.add(part)
            
            feedback = f"Can't see: {', '.join(missing_parts)}"
            
            if 'ankle' in missing_parts or 'knee' in missing_parts:
                suggestions.append("Move back to show full body")
            if 'shoulder' in missing_parts or 'elbow' in missing_parts:
                suggestions.append("Make sure arms are visible")
            
        elif low_confidence:
            feedback = "Detection uncertain - improve lighting or position"
            suggestions.append("Move to a well-lit area")
            suggestions.append("Wear contrasting colors")
        else:
            feedback = "Ready to track!"
            
        return feedback, suggestions
    
    def get_best_visible_side(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]]
    ) -> str:
        """
        Determine which side (left/right) has better visibility.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            'left' or 'right'
        """
        left_visibility = 0
        right_visibility = 0
        
        for name, coords in landmarks.items():
            if 'left_' in name:
                left_visibility += coords[3]
            elif 'right_' in name:
                right_visibility += coords[3]
        
        return 'left' if left_visibility >= right_visibility else 'right'


def is_pose_valid(
    landmarks: Dict[str, Tuple[float, float, float, float]],
    confidence_threshold: float = 0.5
) -> bool:
    """
    Convenience function to check pose validity.
    
    Args:
        landmarks: Dictionary of landmark coordinates
        confidence_threshold: Minimum visibility threshold
        
    Returns:
        True if pose is valid
    """
    validator = PoseValidator(confidence_threshold)
    return validator.is_pose_valid(landmarks)


def get_visibility_feedback(
    landmarks: Dict[str, Tuple[float, float, float, float]]
) -> str:
    """
    Convenience function to get visibility feedback.
    
    Args:
        landmarks: Dictionary of landmark coordinates
        
    Returns:
        Feedback message
    """
    validator = PoseValidator()
    return validator.get_visibility_feedback(landmarks)


# Testing
if __name__ == "__main__":
    validator = PoseValidator(confidence_threshold=0.5)
    
    # Test cases
    print("Testing PoseValidator:")
    print("-" * 50)
    
    # Test 1: Good visibility
    good_landmarks = {
        'left_shoulder': (0.3, 0.3, 0, 0.9),
        'left_elbow': (0.3, 0.5, 0, 0.9),
        'left_wrist': (0.3, 0.7, 0, 0.9),
        'left_hip': (0.35, 0.6, 0, 0.9),
        'right_shoulder': (0.7, 0.3, 0, 0.9),
        'right_elbow': (0.7, 0.5, 0, 0.9),
        'right_wrist': (0.7, 0.7, 0, 0.9),
        'right_hip': (0.65, 0.6, 0, 0.9),
    }
    
    result = validator.validate_for_exercise(good_landmarks, 'bicep_curl')
    print(f"Good pose - Valid: {result.is_valid}, Confidence: {result.confidence:.2f}")
    print(f"  Feedback: {result.feedback}")
    
    # Test 2: Missing landmarks
    partial_landmarks = {
        'left_shoulder': (0.3, 0.3, 0, 0.9),
        'left_elbow': (0.3, 0.5, 0, 0.9),
    }
    
    result = validator.validate_for_exercise(partial_landmarks, 'bicep_curl')
    print(f"\nPartial pose - Valid: {result.is_valid}")
    print(f"  Missing: {result.missing_landmarks}")
    print(f"  Suggestions: {result.suggestions}")
    
    # Test 3: Position feedback
    print(f"\nPosition feedback: {validator.get_visibility_feedback(good_landmarks)}")
    
    print("\nDone!")
