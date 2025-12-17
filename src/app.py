"""
Fitness Coach Main Application
==============================
Main OpenCV-based application that ties all components together.

Features:
- Real-time webcam feed with pose overlay
- Exercise selection (bicep curls, squats, push-ups)
- Rep counter display
- Form feedback with color coding
- Keyboard controls
- Optional voice feedback
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import time

# Import pose detector
from .pose_detector import PoseDetector, WebcamCapture

# Import exercise trackers
from .exercises.bicep_curl import BicepCurlTracker
from .exercises.squat import SquatTracker
from .exercises.pushup import PushUpTracker
from .exercises.shoulder_press import ShoulderPressTracker
from .exercises.lateral_raise import LateralRaiseTracker
from .exercises.front_raise import FrontRaiseTracker
from .exercises.shoulder_shrug import ShoulderShrugTracker
from .exercises.tricep_extension import TricepExtensionTracker
from .exercises.base import ExerciseResult

# Try to import voice feedback
try:
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Warning: pyttsx3 not installed. Voice feedback disabled.")


class VoiceFeedback:
    """
    Voice feedback system using pyttsx3.
    
    Provides audio feedback for rep counts and form corrections.
    """
    
    def __init__(self):
        """Initialize the voice engine."""
        self.engine = None
        self.enabled = False
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.0  # Seconds between voice feedback
        
        if VOICE_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)  # Speed of speech
                self.engine.setProperty('volume', 0.9)
                self.enabled = True
            except Exception as e:
                print(f"Could not initialize voice engine: {e}")
                self.enabled = False
    
    def speak(self, text: str, force: bool = False):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            force: If True, bypass cooldown
        """
        if not self.enabled or not self.engine:
            return
            
        current_time = time.time()
        if not force and (current_time - self.last_feedback_time) < self.feedback_cooldown:
            return
            
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_feedback_time = current_time
        except Exception as e:
            print(f"Voice error: {e}")
    
    def speak_async(self, text: str):
        """
        Queue text for speaking (non-blocking).
        
        Args:
            text: Text to speak
        """
        if not self.enabled or not self.engine:
            return
            
        try:
            self.engine.say(text)
            # Don't wait - let it play in background
        except Exception:
            pass
    
    def toggle(self):
        """Toggle voice feedback on/off."""
        self.enabled = not self.enabled
        return self.enabled


class UIRenderer:
    """
    UI rendering utilities for the fitness coach display.
    
    Handles drawing overlay elements like rep counter, feedback, etc.
    """
    
    # Colors (BGR format)
    COLORS = {
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'blue': (255, 128, 0),
        'orange': (0, 165, 255),
        'purple': (255, 0, 128)
    }
    
    @staticmethod
    def draw_rounded_rect(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = -1,
        radius: int = 10,
        alpha: float = 0.7
    ):
        """
        Draw a rounded rectangle with transparency.
        
        Args:
            img: Image to draw on
            pt1: Top-left corner
            pt2: Bottom-right corner
            color: BGR color
            thickness: -1 for filled
            radius: Corner radius
            alpha: Transparency (0-1)
        """
        overlay = img.copy()
        
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw rounded rectangle on overlay
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corners
        cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        
        # Blend with original
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    @staticmethod
    def draw_info_panel(
        img: np.ndarray,
        exercise_name: str,
        rep_count: int,
        stage: str,
        feedback: str,
        form_score: float,
        fps: int,
        voice_enabled: bool = False
    ):
        """
        Draw the main information panel.
        
        Args:
            img: Image to draw on
            exercise_name: Current exercise name
            rep_count: Number of reps
            stage: Current stage (up/down)
            feedback: Feedback message
            form_score: Form score (0-100)
            fps: Current FPS
            voice_enabled: Whether voice is enabled
        """
        h, w = img.shape[:2]
        
        # Main panel background (left side)
        panel_width = 300
        panel_height = 280
        UIRenderer.draw_rounded_rect(
            img,
            (10, 10),
            (panel_width, panel_height),
            UIRenderer.COLORS['black'],
            alpha=0.6
        )
        
        # Exercise name
        cv2.putText(
            img, exercise_name.upper(),
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, UIRenderer.COLORS['blue'], 2
        )
        
        # Rep counter (large)
        cv2.putText(
            img, "REPS",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, UIRenderer.COLORS['white'], 1
        )
        cv2.putText(
            img, str(rep_count),
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0, UIRenderer.COLORS['green'], 3
        )
        
        # Stage indicator
        stage_color = UIRenderer.COLORS['yellow'] if stage.lower() == 'up' else UIRenderer.COLORS['orange']
        cv2.putText(
            img, f"Stage: {stage.upper()}",
            (120, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, stage_color, 2
        )
        
        # Form score bar
        cv2.putText(
            img, f"Form: {int(form_score)}%",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, UIRenderer.COLORS['white'], 1
        )
        
        # Score bar background
        bar_width = 250
        bar_height = 20
        cv2.rectangle(img, (20, 185), (20 + bar_width, 185 + bar_height), (50, 50, 50), -1)
        
        # Score bar fill
        fill_width = int(bar_width * form_score / 100)
        score_color = (
            UIRenderer.COLORS['green'] if form_score >= 70
            else UIRenderer.COLORS['yellow'] if form_score >= 40
            else UIRenderer.COLORS['red']
        )
        cv2.rectangle(img, (20, 185), (20 + fill_width, 185 + bar_height), score_color, -1)
        
        # Feedback message
        feedback_color = (
            UIRenderer.COLORS['green'] if form_score >= 70 or '✓' in feedback or '💪' in feedback
            else UIRenderer.COLORS['red'] if form_score < 40 or '!' in feedback
            else UIRenderer.COLORS['yellow']
        )
        
        # Wrap long feedback
        feedback_text = feedback[:35] + "..." if len(feedback) > 35 else feedback
        cv2.putText(
            img, feedback_text,
            (20, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, feedback_color, 1
        )
        
        # Voice indicator
        voice_text = "Voice: ON" if voice_enabled else "Voice: OFF"
        voice_color = UIRenderer.COLORS['green'] if voice_enabled else UIRenderer.COLORS['red']
        cv2.putText(
            img, voice_text,
            (20, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, voice_color, 1
        )
        
        # FPS (top right)
        cv2.putText(
            img, f"FPS: {fps}",
            (w - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, UIRenderer.COLORS['green'], 2
        )
    
    @staticmethod
    def draw_controls_help(img: np.ndarray):
        """
        Draw keyboard controls help.
        
        Args:
            img: Image to draw on
        """
        h, w = img.shape[:2]
        
        # Help panel (bottom)
        panel_y = h - 80
        UIRenderer.draw_rounded_rect(
            img,
            (10, panel_y),
            (w - 10, h - 10),
            UIRenderer.COLORS['black'],
            alpha=0.5
        )
        
        controls = [
            "1: Bicep Curl",
            "2: Squat", 
            "3: Push-up",
            "4: Shoulder Press",
            "5: Lateral Raise",
            "6: Front Raise",
            "7: Shrug",
            "8: Tricep Ext",
            "R: Reset",
            "V: Voice",
            "Q: Quit"
        ]
        
        x_start = 30
        x_spacing = (w - 60) // len(controls)
        
        for i, control in enumerate(controls):
            cv2.putText(
                img, control,
                (x_start + i * x_spacing, panel_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, UIRenderer.COLORS['white'], 1
            )


class FitnessCoachApp:
    """
    Main fitness coach application.
    
    Orchestrates pose detection, exercise tracking, and UI rendering.
    """
    
    EXERCISES = {
        'bicep_curl': BicepCurlTracker,
        'squat': SquatTracker,
        'pushup': PushUpTracker,
        'shoulder_press': ShoulderPressTracker,
        'lateral_raise': LateralRaiseTracker,
        'front_raise': FrontRaiseTracker,
        'shoulder_shrug': ShoulderShrugTracker,
        'tricep_extension': TricepExtensionTracker
    }
    
    def __init__(
        self,
        starting_exercise: str = 'bicep_curl',
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        voice_enabled: bool = False
    ):
        """
        Initialize the fitness coach application.
        
        Args:
            starting_exercise: Exercise to start with
            camera_id: Camera device ID
            width: Video width
            height: Video height
            voice_enabled: Enable voice feedback
        """
        # Initialize pose detector
        self.detector = PoseDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.webcam = WebcamCapture(
            camera_id=camera_id,
            width=width,
            height=height
        )
        
        # Initialize exercise trackers
        self.trackers = {
            name: tracker_class()
            for name, tracker_class in self.EXERCISES.items()
        }
        
        # Current exercise
        self.current_exercise = starting_exercise
        self.tracker = self.trackers.get(starting_exercise, self.trackers['bicep_curl'])
        
        # Initialize voice feedback
        self.voice = VoiceFeedback()
        self.voice.enabled = voice_enabled
        
        # UI renderer
        self.ui = UIRenderer()
        
        # State
        self.running = False
        self.last_rep_count = 0
        self.last_feedback = ""
        
    def switch_exercise(self, exercise_name: str):
        """
        Switch to a different exercise.
        
        Args:
            exercise_name: Name of the exercise to switch to
        """
        if exercise_name in self.trackers:
            self.current_exercise = exercise_name
            self.tracker = self.trackers[exercise_name]
            self.tracker.reset()
            self.last_rep_count = 0
            
            if self.voice.enabled:
                self.voice.speak(f"Switched to {self.tracker.exercise_name}", force=True)
                
            print(f"Switched to: {self.tracker.exercise_name}")
    
    def reset_counter(self):
        """Reset the current exercise counter."""
        self.tracker.reset()
        self.last_rep_count = 0
        
        if self.voice.enabled:
            self.voice.speak("Counter reset", force=True)
            
        print("Counter reset")
    
    def run(self):
        """Run the main application loop."""
        # Start webcam
        if not self.webcam.start():
            print("Error: Could not start webcam")
            return
        
        self.running = True
        print("\n" + "="*50)
        print("FITNESS COACH STARTED")
        print("="*50)
        print(f"Exercise: {self.tracker.exercise_name}")
        print("Press 'Q' to quit")
        print("="*50 + "\n")
        
        try:
            while self.running:
                # Read frame
                success, frame = self.webcam.read()
                if not success:
                    print("Failed to read frame")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Get frame dimensions
                h, w = frame.shape[:2]
                
                # Process frame with pose detector
                frame = self.detector.process_frame(frame)
                
                # Draw landmarks
                frame = self.detector.draw_landmarks(frame)
                
                # Process exercise if pose detected
                result = ExerciseResult(
                    rep_count=self.tracker.rep_count,
                    stage=self.tracker.stage,
                    feedback="No pose detected - stand in frame",
                    form_score=0,
                    is_valid_pose=False
                )
                
                if self.detector.is_pose_detected():
                    # Get all landmarks
                    landmarks = self.detector.get_all_landmarks()
                    
                    # Process with exercise tracker
                    result = self.tracker.process(landmarks, w, h)
                    
                    # Voice feedback for reps
                    if result.rep_count > self.last_rep_count:
                        self.last_rep_count = result.rep_count
                        if self.voice.enabled:
                            self.voice.speak(str(result.rep_count))
                    
                    # Voice feedback for form issues
                    if (result.feedback != self.last_feedback and 
                        result.form_score < 70 and 
                        '!' in result.feedback):
                        self.last_feedback = result.feedback
                        if self.voice.enabled:
                            # Clean feedback for speech
                            clean_feedback = result.feedback.replace('!', '').replace('📍', '').replace('🚫', '').replace('⬇️', '').replace('⬆️', '')
                            self.voice.speak(clean_feedback)
                
                # Draw UI
                self.ui.draw_info_panel(
                    frame,
                    exercise_name=self.tracker.exercise_name,
                    rep_count=result.rep_count,
                    stage=result.stage,
                    feedback=result.feedback,
                    form_score=result.form_score,
                    fps=int(self.detector.fps),
                    voice_enabled=self.voice.enabled
                )
                
                # Draw controls help
                self.ui.draw_controls_help(frame)
                
                # Display frame
                cv2.imshow("Fitness Coach", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self._cleanup()
    
    def _handle_key(self, key: int):
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey
        """
        if key == ord('q') or key == ord('Q'):
            self.running = False
        elif key == ord('1'):
            self.switch_exercise('bicep_curl')
        elif key == ord('2'):
            self.switch_exercise('squat')
        elif key == ord('3'):
            self.switch_exercise('pushup')
        elif key == ord('4'):
            self.switch_exercise('shoulder_press')
        elif key == ord('5'):
            self.switch_exercise('lateral_raise')
        elif key == ord('6'):
            self.switch_exercise('front_raise')
        elif key == ord('7'):
            self.switch_exercise('shoulder_shrug')
        elif key == ord('8'):
            self.switch_exercise('tricep_extension')
        elif key == ord('r') or key == ord('R'):
            self.reset_counter()
        elif key == ord('v') or key == ord('V'):
            enabled = self.voice.toggle()
            print(f"Voice feedback: {'ON' if enabled else 'OFF'}")
    
    def _cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        self.webcam.release()
        self.detector.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


# Entry point for direct execution
if __name__ == "__main__":
    app = FitnessCoachApp(
        starting_exercise='bicep_curl',
        voice_enabled=False
    )
    app.run()
