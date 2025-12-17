# Adding New Exercises

This guide explains how to add new exercise trackers to the Pose-Based Fitness Coach.

## 📋 Overview

Each exercise tracker is a Python class that:
1. Inherits from `BaseExerciseTracker`
2. Processes landmark data to detect exercise movements
3. Counts repetitions
4. Provides form feedback

## 🏗️ Step-by-Step Guide

### Step 1: Understand the Landmarks

MediaPipe Pose provides 33 body landmarks. The most commonly used ones are:

| Landmark | Index | Use Case |
|----------|-------|----------|
| `left_shoulder` | 11 | Upper body exercises |
| `right_shoulder` | 12 | Upper body exercises |
| `left_elbow` | 13 | Arm exercises |
| `right_elbow` | 14 | Arm exercises |
| `left_wrist` | 15 | Arm exercises |
| `right_wrist` | 16 | Arm exercises |
| `left_hip` | 23 | Core/leg exercises |
| `right_hip` | 24 | Core/leg exercises |
| `left_knee` | 25 | Leg exercises |
| `right_knee` | 26 | Leg exercises |
| `left_ankle` | 27 | Leg exercises |
| `right_ankle` | 28 | Leg exercises |

Each landmark provides:
- `x`: Horizontal position (0-1, normalized)
- `y`: Vertical position (0-1, normalized)
- `z`: Depth (smaller = closer to camera)
- `visibility`: Confidence score (0-1)

### Step 2: Create the Tracker File

Create a new file in `src/exercises/`:

```python
# src/exercises/lateral_raise.py
"""
Lateral Raise Exercise Tracker
==============================
Tracks lateral raise repetitions and provides form feedback.
"""

from typing import Dict, Tuple, List
import numpy as np

from .base import BaseExerciseTracker, ExerciseResult


class LateralRaiseTracker(BaseExerciseTracker):
    """
    Tracker for lateral raise exercise.
    
    Monitors shoulder abduction angle to count reps.
    """
    
    # Define angle thresholds for your exercise
    DOWN_ANGLE_THRESHOLD = 20   # Arms at sides
    UP_ANGLE_THRESHOLD = 80     # Arms raised to shoulder height
    
    def __init__(self, track_side: str = "left"):
        """Initialize the tracker."""
        super().__init__()
        self.track_side = track_side.lower()
        self.stage = "down"
        self.angle = 0
        self.angle_history: List[float] = []
        
    def get_required_landmarks(self) -> List[str]:
        """Define which landmarks this exercise needs."""
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
        """Process landmarks and track the exercise."""
        
        # Step 1: Check if required landmarks are visible
        is_valid, missing = self.check_landmarks_visibility(landmarks)
        if not is_valid:
            return ExerciseResult(
                rep_count=self.rep_count,
                stage=self.stage,
                feedback=f"Can't see: {', '.join(missing)}",
                form_score=0,
                is_valid_pose=False
            )
        
        # Step 2: Get landmark positions
        side = self.track_side
        shoulder = landmarks[f"{side}_shoulder"]
        elbow = landmarks[f"{side}_elbow"]
        wrist = landmarks[f"{side}_wrist"]
        hip = landmarks[f"{side}_hip"]
        
        # Step 3: Calculate relevant angles
        self.angle = self._calculate_shoulder_angle(shoulder, hip, wrist)
        
        # Step 4: Perform form checks
        penalties = []
        feedback_messages = []
        
        # Example form check: elbow should stay relatively straight
        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
        if elbow_angle < 150:
            penalties.append(20)
            feedback_messages.append("Keep arms straight!")
        
        # Step 5: Update rep counting logic
        if self.angle > self.UP_ANGLE_THRESHOLD:
            if self.stage == "down":
                # Transitioned from down to up
                pass
            self.stage = "up"
            if not feedback_messages:
                feedback_messages = ["Good height! ✓"]
                
        elif self.angle < self.DOWN_ANGLE_THRESHOLD:
            if self.stage == "up":
                # Completed a rep
                self.rep_count += 1
                if not penalties:
                    feedback_messages = ["Great rep! 💪"]
            self.stage = "down"
        
        # Step 6: Calculate form score
        self.form_score = self._calculate_form_score(penalties)
        
        # Step 7: Return result
        return ExerciseResult(
            rep_count=self.rep_count,
            stage=self.stage,
            feedback=feedback_messages[0] if feedback_messages else f"Angle: {int(self.angle)}°",
            form_score=self.form_score,
            is_valid_pose=True,
            additional_data={"angle": self.angle}
        )
    
    def _calculate_shoulder_angle(
        self,
        shoulder: Tuple,
        hip: Tuple,
        wrist: Tuple
    ) -> float:
        """Calculate shoulder abduction angle."""
        # Vector from shoulder to hip (torso)
        torso = np.array([hip[0] - shoulder[0], hip[1] - shoulder[1]])
        # Vector from shoulder to wrist (arm)
        arm = np.array([wrist[0] - shoulder[0], wrist[1] - shoulder[1]])
        
        # Calculate angle
        cos_angle = np.dot(torso, arm) / (np.linalg.norm(torso) * np.linalg.norm(arm) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _calculate_angle(self, p1, p2, p3) -> float:
        """Calculate angle at p2 between p1 and p3."""
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        ba = a - b
        bc = c - b
        
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.degrees(np.arccos(cos_angle))
    
    @property
    def exercise_name(self) -> str:
        return "Lateral Raise"
```

### Step 3: Register the Exercise

Add your new exercise to `src/app.py`:

```python
from .exercises.lateral_raise import LateralRaiseTracker

class FitnessCoachApp:
    EXERCISES = {
        'bicep_curl': BicepCurlTracker,
        'squat': SquatTracker,
        'pushup': PushUpTracker,
        'lateral_raise': LateralRaiseTracker,  # Add here
    }
```

Update keyboard controls:

```python
def _handle_key(self, key: int):
    # ...
    elif key == ord('4'):
        self.switch_exercise('lateral_raise')
```

### Step 4: Add Tests

Create tests in `tests/test_exercises.py`:

```python
class TestLateralRaiseTracker:
    @pytest.fixture
    def tracker(self):
        return LateralRaiseTracker(track_side="left")
    
    def test_initialization(self, tracker):
        assert tracker.rep_count == 0
        assert tracker.stage == "down"
    
    def test_required_landmarks(self, tracker):
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "left_wrist" in required
    
    def test_rep_counting(self, tracker):
        # Add test for full rep cycle
        pass
```

### Step 5: Update Documentation

Add your exercise to the README.md table:

```markdown
| **Lateral Raise** | Shoulder, Elbow, Wrist, Hip | Arm straightness, height |
```

## 🎯 Best Practices

### 1. Angle Thresholds

- Start with estimated values based on exercise mechanics
- Test with actual users and adjust
- Add some buffer (don't use exactly 90°, use 85° or 95°)

### 2. Form Checks

Common form checks to implement:

| Check | How to Detect |
|-------|---------------|
| Body stability | Track shoulder/hip Y position changes |
| Straight arm | Calculate elbow angle (should be >160°) |
| Knee position | Compare knee X to ankle X |
| Back angle | Calculate torso angle from vertical |

### 3. Smoothing

Use angle history for smoother detection:

```python
def _smooth_angle(self, angle: float) -> float:
    self.angle_history.append(angle)
    if len(self.angle_history) > 5:
        self.angle_history.pop(0)
    return np.mean(self.angle_history)
```

### 4. Feedback Messages

- Keep feedback short (< 30 characters)
- Use emojis for visual clarity
- Provide both corrective and positive feedback
- Color-code in UI (green=good, red=bad)

## 📊 Exercise Ideas

Here are some exercises you could implement:

| Exercise | Key Landmarks | Key Angle |
|----------|--------------|-----------|
| Shoulder Press | shoulder, elbow, wrist | Elbow angle |
| Lunges | hip, knee, ankle | Knee angles |
| Plank | shoulder, hip, ankle | Hip alignment |
| Jumping Jacks | shoulder, hip, wrist, ankle | Arm/leg spread |
| Tricep Dips | shoulder, elbow, wrist | Elbow angle |
| Calf Raises | knee, ankle, foot_index | Ankle angle |

## ❓ Common Issues

### Issue: False Rep Counts

**Solution:** Add debouncing or require the angle to stay in position for multiple frames:

```python
if self.frames_in_position > 3:
    self.rep_count += 1
```

### Issue: Jittery Form Feedback

**Solution:** Use smoothing and rate limiting:

```python
if time.time() - self.last_feedback_time > 2.0:
    self.feedback = new_feedback
    self.last_feedback_time = time.time()
```

### Issue: Wrong Side Detection

**Solution:** Detect which side is more visible:

```python
def get_best_side(self, landmarks):
    left_vis = landmarks.get('left_shoulder', (0,0,0,0))[3]
    right_vis = landmarks.get('right_shoulder', (0,0,0,0))[3]
    return 'left' if left_vis > right_vis else 'right'
```

---

Need help? Open an issue on GitHub!
