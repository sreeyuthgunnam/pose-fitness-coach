"""
Unit Tests for Pose Detector
============================
Tests for the PoseDetector class and helper functions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose_detector import PoseDetector


class TestAngleCalculation:
    """Tests for angle calculation function."""
    
    def test_straight_angle(self):
        """Test 180 degree angle (straight line)."""
        point1 = (0, 0, 0)
        point2 = (1, 0, 0)
        point3 = (2, 0, 0)
        
        angle = PoseDetector.calculate_angle(point1, point2, point3)
        assert abs(angle - 180) < 1  # Allow 1 degree tolerance
    
    def test_right_angle(self):
        """Test 90 degree angle."""
        point1 = (0, 0, 0)
        point2 = (0, 1, 0)
        point3 = (1, 1, 0)
        
        angle = PoseDetector.calculate_angle(point1, point2, point3)
        assert abs(angle - 90) < 1
    
    def test_acute_angle(self):
        """Test acute angle (< 90 degrees)."""
        point1 = (0, 0, 0)
        point2 = (1, 0, 0)
        point3 = (1.5, 0.5, 0)
        
        angle = PoseDetector.calculate_angle(point1, point2, point3)
        assert angle < 90
    
    def test_obtuse_angle(self):
        """Test obtuse angle (> 90 degrees)."""
        point1 = (0, 0, 0)
        point2 = (1, 0, 0)
        point3 = (0.5, -0.5, 0)
        
        angle = PoseDetector.calculate_angle(point1, point2, point3)
        assert angle > 90
    
    def test_zero_length_vector(self):
        """Test handling of zero-length vector (same points)."""
        point1 = (0, 0, 0)
        point2 = (0, 0, 0)
        point3 = (1, 0, 0)
        
        # Should not raise exception due to epsilon in denominator
        angle = PoseDetector.calculate_angle(point1, point2, point3)
        assert 0 <= angle <= 180


class TestDistanceCalculation:
    """Tests for distance calculation function."""
    
    def test_horizontal_distance(self):
        """Test distance on horizontal line."""
        point1 = (0, 0)
        point2 = (3, 0)
        
        distance = PoseDetector.calculate_distance(point1, point2)
        assert abs(distance - 3) < 0.001
    
    def test_vertical_distance(self):
        """Test distance on vertical line."""
        point1 = (0, 0)
        point2 = (0, 4)
        
        distance = PoseDetector.calculate_distance(point1, point2)
        assert abs(distance - 4) < 0.001
    
    def test_diagonal_distance(self):
        """Test distance on diagonal (3-4-5 triangle)."""
        point1 = (0, 0)
        point2 = (3, 4)
        
        distance = PoseDetector.calculate_distance(point1, point2)
        assert abs(distance - 5) < 0.001
    
    def test_same_point_distance(self):
        """Test distance between same point."""
        point1 = (5, 5)
        point2 = (5, 5)
        
        distance = PoseDetector.calculate_distance(point1, point2)
        assert distance == 0


class TestLandmarkNames:
    """Tests for landmark name mappings."""
    
    def test_landmark_names_count(self):
        """Test that we have all 33 landmarks mapped."""
        assert len(PoseDetector.LANDMARK_NAMES) == 33
    
    def test_landmark_indices_inverse(self):
        """Test that LANDMARK_INDICES is proper inverse of LANDMARK_NAMES."""
        for idx, name in PoseDetector.LANDMARK_NAMES.items():
            assert PoseDetector.LANDMARK_INDICES[name] == idx
    
    def test_key_landmarks_present(self):
        """Test that key landmarks are in the mapping."""
        key_landmarks = [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        for lm in key_landmarks:
            assert lm in PoseDetector.LANDMARK_INDICES


class TestPoseDetectorInit:
    """Tests for PoseDetector initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        detector = PoseDetector()
        assert detector.min_detection_confidence == 0.5
        assert detector.min_tracking_confidence == 0.5
        assert detector.landmarks is None
        detector.release()
    
    def test_custom_confidence(self):
        """Test initialization with custom confidence values."""
        detector = PoseDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8
        )
        assert detector.min_detection_confidence == 0.7
        assert detector.min_tracking_confidence == 0.8
        detector.release()
    
    def test_is_pose_detected_false_initially(self):
        """Test that no pose is detected before processing."""
        detector = PoseDetector()
        assert not detector.is_pose_detected()
        detector.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
