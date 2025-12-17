"""
Pose Detector Module
====================
Core module for real-time pose detection using MediaPipe.

This module provides the PoseDetector class which handles:
- Webcam video capture
- MediaPipe Pose landmark detection
- Skeleton visualization
- Angle calculations for exercise tracking
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Dict, Tuple, Optional, List


class PoseDetector:
    """
    A class for detecting human pose landmarks using MediaPipe.
    
    Attributes:
        min_detection_confidence (float): Minimum confidence for initial detection
        min_tracking_confidence (float): Minimum confidence for landmark tracking
        static_image_mode (bool): Whether to treat each frame independently
    """
    
    # MediaPipe landmark indices mapping
    LANDMARK_NAMES = {
        0: "nose",
        1: "left_eye_inner",
        2: "left_eye",
        3: "left_eye_outer",
        4: "right_eye_inner",
        5: "right_eye",
        6: "right_eye_outer",
        7: "left_ear",
        8: "right_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        17: "left_pinky",
        18: "right_pinky",
        19: "left_index",
        20: "right_index",
        21: "left_thumb",
        22: "right_thumb",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
        29: "left_heel",
        30: "right_heel",
        31: "left_foot_index",
        32: "right_foot_index"
    }
    
    # Reverse mapping: name to index
    LANDMARK_INDICES = {v: k for k, v in LANDMARK_NAMES.items()}
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the PoseDetector.
        
        Args:
            static_image_mode: If True, treats each image independently (slower but more accurate)
            model_complexity: Complexity of the pose model (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            enable_segmentation: Whether to enable segmentation mask
            min_detection_confidence: Minimum confidence for person detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Store configuration
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Landmarks storage
        self.landmarks = None
        self.world_landmarks = None
        self.results = None
        
        # FPS calculation
        self.prev_time = 0
        self.fps = 0
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a video frame and detect pose landmarks.
        
        Args:
            frame: BGR image from OpenCV (numpy array)
            
        Returns:
            The processed frame (can be used for further processing)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make detection
        self.results = self.pose.process(rgb_frame)
        
        # Store landmarks if detected
        if self.results.pose_landmarks:
            self.landmarks = self.results.pose_landmarks.landmark
            self.world_landmarks = self.results.pose_world_landmarks.landmark if self.results.pose_world_landmarks else None
        else:
            self.landmarks = None
            self.world_landmarks = None
            
        # Update FPS
        self._update_fps()
        
        return frame
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        draw_connections: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        circle_radius: int = 3
    ) -> np.ndarray:
        """
        Draw pose landmarks and skeleton on the frame.
        
        Args:
            frame: BGR image to draw on
            draw_connections: Whether to draw connections between landmarks
            landmark_color: BGR color for landmarks
            connection_color: BGR color for connections
            thickness: Line thickness
            circle_radius: Radius of landmark circles
            
        Returns:
            Frame with landmarks drawn
        """
        if self.results and self.results.pose_landmarks:
            # Use custom drawing for more control
            self.mp_drawing.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS if draw_connections else None,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=landmark_color,
                    thickness=thickness,
                    circle_radius=circle_radius
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=connection_color,
                    thickness=thickness
                )
            )
        return frame
    
    def get_landmark(
        self,
        name: str,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Get coordinates of a specific landmark by name.
        
        Args:
            name: Landmark name (e.g., "left_shoulder", "right_elbow")
            frame_width: Optional frame width for pixel coordinate conversion
            frame_height: Optional frame height for pixel coordinate conversion
            
        Returns:
            Tuple of (x, y, z, visibility) or None if landmark not found
            If frame dimensions provided, x and y are in pixels
            Otherwise, x and y are normalized (0-1)
        """
        if self.landmarks is None:
            return None
            
        name_lower = name.lower()
        if name_lower not in self.LANDMARK_INDICES:
            raise ValueError(f"Unknown landmark name: {name}. Valid names: {list(self.LANDMARK_INDICES.keys())}")
            
        idx = self.LANDMARK_INDICES[name_lower]
        landmark = self.landmarks[idx]
        
        x, y, z = landmark.x, landmark.y, landmark.z
        visibility = landmark.visibility
        
        # Convert to pixel coordinates if dimensions provided
        if frame_width and frame_height:
            x = x * frame_width
            y = y * frame_height
            
        return (x, y, z, visibility)
    
    def get_all_landmarks(
        self,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get all landmarks as a dictionary.
        
        Args:
            frame_width: Optional frame width for pixel coordinate conversion
            frame_height: Optional frame height for pixel coordinate conversion
            
        Returns:
            Dictionary mapping landmark names to (x, y, z, visibility) tuples
        """
        if self.landmarks is None:
            return {}
            
        landmarks_dict = {}
        for idx, name in self.LANDMARK_NAMES.items():
            landmark = self.landmarks[idx]
            x, y, z = landmark.x, landmark.y, landmark.z
            visibility = landmark.visibility
            
            if frame_width and frame_height:
                x = x * frame_width
                y = y * frame_height
                
            landmarks_dict[name] = (x, y, z, visibility)
            
        return landmarks_dict
    
    @staticmethod
    def calculate_angle(
        point1: Tuple[float, float, float],
        point2: Tuple[float, float, float],
        point3: Tuple[float, float, float]
    ) -> float:
        """
        Calculate the angle between three points.
        
        The angle is calculated at point2 (the middle point).
        
        Args:
            point1: First point (x, y, z) or (x, y)
            point2: Middle point (vertex of the angle)
            point3: Third point
            
        Returns:
            Angle in degrees (0-180)
        """
        # Extract x, y coordinates (ignore z for 2D angle)
        a = np.array([point1[0], point1[1]])
        b = np.array([point2[0], point2[1]])
        c = np.array([point3[0], point3[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        
        # Clip to avoid numerical errors with arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    @staticmethod
    def calculate_distance(
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance between the points
        """
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            self.fps = 1 / time_diff
        self.prev_time = current_time
        
    def draw_fps(
        self,
        frame: np.ndarray,
        position: Tuple[int, int] = (10, 30),
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw FPS counter on the frame.
        
        Args:
            frame: BGR image to draw on
            position: Top-left position for the text
            font_scale: Font size scale
            color: BGR color for text
            thickness: Text thickness
            
        Returns:
            Frame with FPS drawn
        """
        cv2.putText(
            frame,
            f"FPS: {int(self.fps)}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness
        )
        return frame
    
    def is_pose_detected(self) -> bool:
        """Check if a pose was detected in the last processed frame."""
        return self.landmarks is not None
    
    def get_landmark_visibility(self, name: str) -> float:
        """
        Get the visibility/confidence score for a specific landmark.
        
        Args:
            name: Landmark name
            
        Returns:
            Visibility score (0-1) or 0 if not detected
        """
        landmark = self.get_landmark(name)
        if landmark:
            return landmark[3]  # visibility is the 4th element
        return 0.0
    
    def release(self):
        """Release MediaPipe resources."""
        self.pose.close()


class WebcamCapture:
    """
    A class for capturing video from webcam with helper methods.
    
    Attributes:
        camera_id (int): Camera device ID
        width (int): Frame width
        height (int): Frame height
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720
    ):
        """
        Initialize webcam capture.
        
        Args:
            camera_id: Camera device ID (0 for default camera)
            width: Desired frame width
            height: Desired frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        
    def start(self) -> bool:
        """
        Start video capture.
        
        Returns:
            True if capture started successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Get actual resolution (may differ from requested)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera started at {self.width}x{self.height}")
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the webcam.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get current frame dimensions."""
        return self.width, self.height
    
    def release(self):
        """Release webcam resources."""
        if self.cap:
            self.cap.release()
            
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()


# Example usage and testing
if __name__ == "__main__":
    print("Testing PoseDetector...")
    
    # Initialize detector and webcam
    detector = PoseDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    webcam = WebcamCapture(camera_id=0)
    
    if not webcam.start():
        print("Failed to start webcam")
        exit(1)
    
    print("Press 'q' to quit")
    print("Press 's' to show all landmarks")
    
    while True:
        success, frame = webcam.read()
        if not success:
            print("Failed to read frame")
            break
            
        # Process frame
        frame = detector.process_frame(frame)
        
        # Draw landmarks
        frame = detector.draw_landmarks(frame)
        
        # Draw FPS
        frame = detector.draw_fps(frame)
        
        # Show some landmark info if detected
        if detector.is_pose_detected():
            # Get left shoulder position
            left_shoulder = detector.get_landmark("left_shoulder")
            if left_shoulder:
                h, w = frame.shape[:2]
                x, y = int(left_shoulder[0] * w), int(left_shoulder[1] * h)
                cv2.putText(
                    frame,
                    f"L.Shoulder: ({x}, {y})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
            # Calculate and display elbow angle
            left_shoulder = detector.get_landmark("left_shoulder")
            left_elbow = detector.get_landmark("left_elbow")
            left_wrist = detector.get_landmark("left_wrist")
            
            if all([left_shoulder, left_elbow, left_wrist]):
                angle = detector.calculate_angle(
                    left_shoulder[:3],
                    left_elbow[:3],
                    left_wrist[:3]
                )
                cv2.putText(
                    frame,
                    f"L.Elbow Angle: {int(angle)}°",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        else:
            cv2.putText(
                frame,
                "No pose detected",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        # Display frame
        cv2.imshow("Pose Detection Test", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Print all landmarks
            landmarks = detector.get_all_landmarks()
            for name, coords in landmarks.items():
                print(f"{name}: x={coords[0]:.3f}, y={coords[1]:.3f}, z={coords[2]:.3f}, vis={coords[3]:.3f}")
            print("-" * 50)
    
    # Cleanup
    webcam.release()
    detector.release()
    cv2.destroyAllWindows()
    print("Done!")
