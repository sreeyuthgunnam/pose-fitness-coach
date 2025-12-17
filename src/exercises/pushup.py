"""
Push-Up Exercise Tracker
========================
Tracks push-up repetitions and provides form feedback.

Key landmarks used:
- Shoulder (left/right)
- Elbow (left/right)
- Wrist (left/right)
- Hip (for body alignment check)
- Ankle (for full body alignment)

Rep counting logic:
- UP: arms extended (elbow angle > 160°)
- DOWN: chest near ground (elbow angle < 90°)
- Count increments on complete up→down→up cycle

Best tracked from SIDE VIEW for accurate angle detection.
"""

from typing import Dict, Tuple, List
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class PushUpTracker(BaseExerciseTracker):
    """
    Tracker for push-up exercise.
    
    Monitors elbow angle to count reps and checks form including:
    - Body alignment (straight line from shoulders to ankles)
    - Push-up depth
    - Elbow flare (elbows shouldn't go too wide)
    """
    
    # Angle thresholds
    UP_ANGLE_THRESHOLD = 160     # Arms extended
    DOWN_ANGLE_THRESHOLD = 90    # Chest near ground
    
    # Form check thresholds
    BODY_ALIGNMENT_THRESHOLD = 20  # Maximum deviation from straight line (degrees)
    ELBOW_FLARE_THRESHOLD = 0.15   # Maximum elbow distance from shoulder (normalized)
    
    def __init__(self, track_side: str = "left"):
        """
        Initialize the push-up tracker.
        
        Args:
            track_side: Which side to track ("left" or "right")
        """
        super().__init__()
        self.track_side = track_side.lower()
        
        # State tracking
        self.stage = "up"
        self.elbow_angle = 0
        
        # Track lowest point in current rep
        self.lowest_angle = 180
        self.reached_depth = False
        
        # Smoothing
        self.angle_history: List[float] = []
        self.history_size = 5
        
        # Form tracking
        self.body_alignment_score = 100
        
    def get_required_landmarks(self) -> List[str]:
        """Get required landmarks for push-up tracking."""
        side = self.track_side
        return [
            f"{side}_shoulder",
            f"{side}_elbow",
            f"{side}_wrist",
            f"{side}_hip",
            f"{side}_ankle"
        ]
    
    def process(
        self,
        landmarks: Dict[str, Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int
    ) -> ExerciseResult:
        """
        Process landmarks and track push-up.
        
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
                feedback=f"Can't see: {', '.join(missing)}. Use side view!",
                form_score=0,
                is_valid_pose=False
            )
        
        # Get landmark positions
        side = self.track_side
        shoulder = landmarks[f"{side}_shoulder"]
        elbow = landmarks[f"{side}_elbow"]
        wrist = landmarks[f"{side}_wrist"]
        hip = landmarks[f"{side}_hip"]
        ankle = landmarks[f"{side}_ankle"]
        
        # Calculate elbow angle
        self.elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
        
        # Smooth the angle
        self._smooth_angle()
        smoothed_angle = np.mean(self.angle_history) if self.angle_history else self.elbow_angle
        
        # Track lowest point
        if smoothed_angle < self.lowest_angle:
            self.lowest_angle = smoothed_angle
        
        # Form checks
        penalties = []
        feedback_messages = []
        
        # Check 1: Body alignment (straight line)
        alignment_penalty, alignment_feedback = self._check_body_alignment(
            shoulder, hip, ankle
        )
        if alignment_penalty > 0:
            penalties.append(alignment_penalty)
            feedback_messages.append(alignment_feedback)
            
        # Check 2: Elbow flare
        flare_penalty, flare_feedback = self._check_elbow_flare(
            shoulder, elbow, frame_width
        )
        if flare_penalty > 0:
            penalties.append(flare_penalty)
            feedback_messages.append(flare_feedback)
            
        # Check 3: Depth feedback
        depth_feedback = self._check_depth(smoothed_angle)
        
        # Update rep counting
        prev_stage = self.stage
        
        if smoothed_angle > self.UP_ANGLE_THRESHOLD:
            # Arms extended (up position)
            if self.stage == "down":
                # Completed a rep (was down, now up)
                self.rep_count += 1
                if self.reached_depth:
                    if not penalties:
                        feedback_messages = ["Great push-up! 💪"]
                else:
                    feedback_messages.insert(0, "Go lower next time! ⬇️")
                # Reset for next rep
                self.lowest_angle = 180
                self.reached_depth = False
            self.stage = "up"
            
        elif smoothed_angle < self.DOWN_ANGLE_THRESHOLD:
            # Chest near ground (down position)
            self.stage = "down"
            self.reached_depth = True
            if not feedback_messages:
                feedback_messages = ["Good depth! ✓"]
        else:
            # In between - transitioning
            if self.stage == "up" and smoothed_angle < 140:
                if not feedback_messages:
                    feedback_messages = ["Keep going down... ⬇️"]
        
        # Calculate form score
        self.form_score = self._calculate_form_score(penalties)
        
        # Combine feedback
        if feedback_messages:
            self.feedback = feedback_messages[0]
        elif depth_feedback:
            self.feedback = depth_feedback
        else:
            self.feedback = f"Angle: {int(smoothed_angle)}°"
        
        return ExerciseResult(
            rep_count=self.rep_count,
            stage=self.stage,
            feedback=self.feedback,
            form_score=self.form_score,
            is_valid_pose=True,
            additional_data={
                "elbow_angle": smoothed_angle,
                "lowest_angle": self.lowest_angle,
                "reached_depth": self.reached_depth,
                "body_alignment_score": self.body_alignment_score
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
        self.angle_history.append(self.elbow_angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
    
    def _check_body_alignment(
        self,
        shoulder: Tuple[float, float, float, float],
        hip: Tuple[float, float, float, float],
        ankle: Tuple[float, float, float, float]
    ) -> Tuple[float, str]:
        """
        Check if body forms a straight line from shoulders to ankles.
        
        Args:
            shoulder: Shoulder landmark
            hip: Hip landmark
            ankle: Ankle landmark
            
        Returns:
            Tuple of (penalty, feedback_message)
        """
        # Calculate the angle at hip between shoulder and ankle
        # Should be close to 180° for a straight line
        body_angle = self._calculate_angle(shoulder, hip, ankle)
        
        # Ideal is 180° (straight line)
        deviation = abs(180 - body_angle)
        
        # Update alignment score
        self.body_alignment_score = max(0, 100 - deviation * 2)
        
        if deviation > self.BODY_ALIGNMENT_THRESHOLD:
            penalty = min(35, deviation * 1.5)
            
            # Determine if hips are too high or sagging
            shoulder_y = shoulder[1]
            hip_y = hip[1]
            ankle_y = ankle[1]
            
            # Expected hip position for straight line
            expected_hip_y = (shoulder_y + ankle_y) / 2
            
            if hip_y < expected_hip_y - 0.05:
                return penalty, "Hips too high! Lower them 📉"
            elif hip_y > expected_hip_y + 0.05:
                return penalty, "Don't let hips sag! 📈"
            else:
                return penalty, "Keep body straight! 📏"
        
        return 0, ""
    
    def _check_elbow_flare(
        self,
        shoulder: Tuple[float, float, float, float],
        elbow: Tuple[float, float, float, float],
        frame_width: int
    ) -> Tuple[float, str]:
        """
        Check if elbows are flaring out too much.
        
        In side view, this checks if elbow is too far from shoulder horizontally.
        
        Args:
            shoulder: Shoulder landmark
            elbow: Elbow landmark
            frame_width: Frame width for normalization
            
        Returns:
            Tuple of (penalty, feedback_message)
        """
        # Calculate horizontal distance between shoulder and elbow
        # In side view, significant z-difference indicates flare
        elbow_x = elbow[0]
        shoulder_x = shoulder[0]
        
        horizontal_diff = abs(elbow_x - shoulder_x)
        
        # Only check during the down phase
        if self.elbow_angle < 140:  # Going down or at bottom
            if horizontal_diff > self.ELBOW_FLARE_THRESHOLD:
                penalty = min(25, horizontal_diff * 150)
                return penalty, "Don't flare elbows! Keep them tucked 💪"
        
        return 0, ""
    
    def _check_depth(self, angle: float) -> str:
        """
        Provide feedback on push-up depth.
        
        Args:
            angle: Current elbow angle
            
        Returns:
            Feedback message
        """
        if self.stage == "down" or (self.stage == "up" and angle < 140):
            if angle > self.DOWN_ANGLE_THRESHOLD and angle < 140:
                return "Go lower! ⬇️"
        return ""
    
    def _is_in_plank_position(
        self,
        shoulder: Tuple[float, float, float, float],
        hip: Tuple[float, float, float, float],
        ankle: Tuple[float, float, float, float]
    ) -> bool:
        """
        Check if user is in plank position (horizontal body).
        
        Args:
            shoulder: Shoulder landmark
            hip: Hip landmark
            ankle: Ankle landmark
            
        Returns:
            True if in plank position
        """
        # Check if body is roughly horizontal
        # All y-coordinates should be similar
        y_values = [shoulder[1], hip[1], ankle[1]]
        y_range = max(y_values) - min(y_values)
        
        # If y-range is small, body is horizontal
        return y_range < 0.3
    
    def reset(self):
        """Reset tracker state."""
        super().reset()
        self.stage = "up"
        self.elbow_angle = 0
        self.lowest_angle = 180
        self.reached_depth = False
        self.body_alignment_score = 100
        self.angle_history.clear()
    
    @property
    def exercise_name(self) -> str:
        """Get exercise name."""
        return "Push-Up"


# Testing
if __name__ == "__main__":
    tracker = PushUpTracker(track_side="left")
    
    # Simulate a push-up motion (side view, body horizontal)
    test_cases = [
        # Up position (arms extended)
        {
            "left_shoulder": (0.3, 0.5, 0, 0.9),
            "left_elbow": (0.4, 0.5, 0, 0.9),
            "left_wrist": (0.5, 0.5, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9),
            "left_ankle": (0.8, 0.5, 0, 0.9)
        },
        # Going down
        {
            "left_shoulder": (0.3, 0.55, 0, 0.9),
            "left_elbow": (0.35, 0.6, 0, 0.9),
            "left_wrist": (0.5, 0.55, 0, 0.9),
            "left_hip": (0.5, 0.55, 0, 0.9),
            "left_ankle": (0.8, 0.52, 0, 0.9)
        },
        # Down position (chest near ground)
        {
            "left_shoulder": (0.3, 0.6, 0, 0.9),
            "left_elbow": (0.32, 0.65, 0, 0.9),
            "left_wrist": (0.5, 0.6, 0, 0.9),
            "left_hip": (0.5, 0.6, 0, 0.9),
            "left_ankle": (0.8, 0.55, 0, 0.9)
        },
        # Back to up
        {
            "left_shoulder": (0.3, 0.5, 0, 0.9),
            "left_elbow": (0.4, 0.5, 0, 0.9),
            "left_wrist": (0.5, 0.5, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9),
            "left_ankle": (0.8, 0.5, 0, 0.9)
        }
    ]
    
    print("Testing PushUpTracker:")
    print("-" * 50)
    
    for i, landmarks in enumerate(test_cases):
        result = tracker.process(landmarks, 640, 480)
        print(f"Frame {i + 1}:")
        print(f"  Reps: {result.rep_count}")
        print(f"  Stage: {result.stage}")
        print(f"  Feedback: {result.feedback}")
        print(f"  Form Score: {result.form_score:.1f}%")
        if result.additional_data:
            print(f"  Elbow Angle: {result.additional_data.get('elbow_angle', 0):.1f}°")
        print()
