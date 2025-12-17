"""
Unit Tests for Exercise Trackers
================================
Tests for all exercise trackers.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exercises.bicep_curl import BicepCurlTracker
from src.exercises.squat import SquatTracker
from src.exercises.pushup import PushUpTracker
from src.exercises.shoulder_press import ShoulderPressTracker
from src.exercises.lateral_raise import LateralRaiseTracker
from src.exercises.front_raise import FrontRaiseTracker
from src.exercises.shoulder_shrug import ShoulderShrugTracker
from src.exercises.tricep_extension import TricepExtensionTracker


class TestBicepCurlTracker:
    """Tests for BicepCurlTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return BicepCurlTracker(track_side="left")
    
    def test_initialization(self, tracker):
        """Test initial state."""
        assert tracker.rep_count == 0
        assert tracker.stage == "down"
        assert tracker.track_side == "left"
    
    def test_required_landmarks(self, tracker):
        """Test required landmarks list."""
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "left_elbow" in required
        assert "left_wrist" in required
        assert "left_hip" in required
    
    def test_missing_landmarks(self, tracker):
        """Test handling of missing landmarks."""
        # Only provide shoulder
        landmarks = {
            "left_shoulder": (0.3, 0.3, 0, 0.9)
        }
        result = tracker.process(landmarks, 640, 480)
        assert not result.is_valid_pose
        assert "Can't see" in result.feedback
    
    def test_down_position_detection(self, tracker):
        """Test detection of down position (arm extended)."""
        # Extended arm - straight line
        landmarks = {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.3, 0.7, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        }
        result = tracker.process(landmarks, 640, 480)
        assert result.stage == "down"
    
    def test_up_position_detection(self, tracker):
        """Test detection of up position (arm curled)."""
        # Curled arm - wrist near shoulder
        landmarks = {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.32, 0.32, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        }
        result = tracker.process(landmarks, 640, 480)
        assert result.stage == "up"
    
    def test_rep_counting(self, tracker):
        """Test full rep counting cycle."""
        # Start in down position
        down_landmarks = {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.3, 0.7, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        }
        
        # Curled position
        up_landmarks = {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.32, 0.32, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        }
        
        # Initial down
        result = tracker.process(down_landmarks, 640, 480)
        assert result.rep_count == 0
        
        # Move to up (several frames for smoothing)
        for _ in range(5):
            result = tracker.process(up_landmarks, 640, 480)
        
        # Back to down - should count rep
        for _ in range(5):
            result = tracker.process(down_landmarks, 640, 480)
        
        assert result.rep_count == 1
    
    def test_reset(self, tracker):
        """Test reset functionality."""
        tracker.rep_count = 5
        tracker.stage = "up"
        tracker.reset()
        
        assert tracker.rep_count == 0
        assert tracker.stage == "down"
    
    def test_form_score_calculation(self, tracker):
        """Test that form score is calculated."""
        landmarks = {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.3, 0.7, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        }
        result = tracker.process(landmarks, 640, 480)
        assert 0 <= result.form_score <= 100


class TestSquatTracker:
    """Tests for SquatTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return SquatTracker(track_side="left")
    
    def test_initialization(self, tracker):
        """Test initial state."""
        assert tracker.rep_count == 0
        assert tracker.stage == "standing"
    
    def test_required_landmarks(self, tracker):
        """Test required landmarks list."""
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "left_hip" in required
        assert "left_knee" in required
        assert "left_ankle" in required
    
    def test_standing_position(self, tracker):
        """Test detection of standing position."""
        landmarks = {
            "left_shoulder": (0.5, 0.2, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9),
            "left_knee": (0.5, 0.7, 0, 0.9),
            "left_ankle": (0.5, 0.9, 0, 0.9)
        }
        result = tracker.process(landmarks, 640, 480)
        assert result.stage == "standing"
    
    def test_squat_position(self, tracker):
        """Test detection of squat position."""
        # Deep squat - knee bent significantly
        landmarks = {
            "left_shoulder": (0.5, 0.35, 0, 0.9),
            "left_hip": (0.5, 0.55, 0, 0.9),
            "left_knee": (0.45, 0.55, 0, 0.9),  # Knee bent
            "left_ankle": (0.5, 0.9, 0, 0.9)
        }
        
        # Process multiple frames
        for _ in range(5):
            result = tracker.process(landmarks, 640, 480)
        
        assert result.stage == "squat"
    
    def test_rep_counting(self, tracker):
        """Test squat rep counting."""
        standing = {
            "left_shoulder": (0.5, 0.2, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9),
            "left_knee": (0.5, 0.7, 0, 0.9),
            "left_ankle": (0.5, 0.9, 0, 0.9)
        }
        
        squat = {
            "left_shoulder": (0.5, 0.35, 0, 0.9),
            "left_hip": (0.5, 0.55, 0, 0.9),
            "left_knee": (0.45, 0.55, 0, 0.9),
            "left_ankle": (0.5, 0.9, 0, 0.9)
        }
        
        # Start standing
        for _ in range(3):
            tracker.process(standing, 640, 480)
        
        # Go to squat
        for _ in range(5):
            tracker.process(squat, 640, 480)
        
        # Back to standing
        for _ in range(5):
            result = tracker.process(standing, 640, 480)
        
        assert result.rep_count == 1


class TestPushUpTracker:
    """Tests for PushUpTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return PushUpTracker(track_side="left")
    
    def test_initialization(self, tracker):
        """Test initial state."""
        assert tracker.rep_count == 0
        assert tracker.stage == "up"
    
    def test_required_landmarks(self, tracker):
        """Test required landmarks list."""
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "left_elbow" in required
        assert "left_wrist" in required
        assert "left_hip" in required
        assert "left_ankle" in required
    
    def test_up_position(self, tracker):
        """Test detection of up position (arms extended)."""
        # Plank position with arms extended
        landmarks = {
            "left_shoulder": (0.3, 0.5, 0, 0.9),
            "left_elbow": (0.4, 0.5, 0, 0.9),
            "left_wrist": (0.5, 0.5, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9),
            "left_ankle": (0.8, 0.5, 0, 0.9)
        }
        result = tracker.process(landmarks, 640, 480)
        assert result.stage == "up"
    
    def test_reset(self, tracker):
        """Test reset functionality."""
        tracker.rep_count = 10
        tracker.stage = "down"
        tracker.reset()
        
        assert tracker.rep_count == 0
        assert tracker.stage == "up"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_low_visibility_landmarks(self):
        """Test handling of low visibility landmarks."""
        tracker = BicepCurlTracker()
        
        # Landmarks with low visibility
        landmarks = {
            "left_shoulder": (0.3, 0.3, 0, 0.2),  # Low visibility
            "left_elbow": (0.3, 0.5, 0, 0.2),
            "left_wrist": (0.3, 0.7, 0, 0.2),
            "left_hip": (0.35, 0.6, 0, 0.2)
        }
        
        result = tracker.process(landmarks, 640, 480)
        assert not result.is_valid_pose
    
    def test_empty_landmarks(self):
        """Test handling of empty landmarks dictionary."""
        tracker = BicepCurlTracker()
        result = tracker.process({}, 640, 480)
        assert not result.is_valid_pose
    
    def test_extreme_angles(self):
        """Test handling of extreme landmark positions."""
        tracker = BicepCurlTracker()
        
        # All landmarks at same position (degenerate case)
        landmarks = {
            "left_shoulder": (0.5, 0.5, 0, 0.9),
            "left_elbow": (0.5, 0.5, 0, 0.9),
            "left_wrist": (0.5, 0.5, 0, 0.9),
            "left_hip": (0.5, 0.5, 0, 0.9)
        }
        
        # Should not raise exception
        result = tracker.process(landmarks, 640, 480)
        assert result is not None
    
    def test_normalized_coordinates(self):
        """Test that trackers work with normalized coordinates (0-1)."""
        tracker = BicepCurlTracker()
        
        # Normalized coordinates
        landmarks = {
            "left_shoulder": (0.3, 0.3, 0, 0.9),
            "left_elbow": (0.3, 0.5, 0, 0.9),
            "left_wrist": (0.3, 0.7, 0, 0.9),
            "left_hip": (0.35, 0.6, 0, 0.9)
        }
        
        result = tracker.process(landmarks, 1, 1)  # Normalized frame
        assert result.is_valid_pose


class TestAngleCalculations:
    """Tests for angle calculation in exercise trackers."""
    
    def test_bicep_curl_extended_angle(self):
        """Test angle calculation for extended arm."""
        tracker = BicepCurlTracker()
        
        # Points forming a straight line (180 degrees)
        shoulder = (0, 0, 0, 0.9)
        elbow = (0, 0.5, 0, 0.9)
        wrist = (0, 1, 0, 0.9)
        
        angle = tracker._calculate_angle(shoulder, elbow, wrist)
        assert angle > 170  # Should be close to 180
    
    def test_bicep_curl_bent_angle(self):
        """Test angle calculation for bent arm."""
        tracker = BicepCurlTracker()
        
        # Points forming roughly 90 degree angle
        shoulder = (0, 0, 0, 0.9)
        elbow = (0, 0.5, 0, 0.9)
        wrist = (0.5, 0.5, 0, 0.9)
        
        angle = tracker._calculate_angle(shoulder, elbow, wrist)
        assert 80 < angle < 100  # Should be close to 90


class TestShoulderPressTracker:
    """Tests for ShoulderPressTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return ShoulderPressTracker(track_side="left")
    
    def test_initialization(self, tracker):
        """Test initial state."""
        assert tracker.rep_count == 0
        assert tracker.stage == "down"
    
    def test_required_landmarks(self, tracker):
        """Test required landmarks list."""
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "left_elbow" in required
        assert "left_wrist" in required


class TestLateralRaiseTracker:
    """Tests for LateralRaiseTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return LateralRaiseTracker(track_side="both")
    
    def test_initialization(self, tracker):
        """Test initial state."""
        assert tracker.rep_count == 0
        assert tracker.stage == "down"
    
    def test_required_landmarks(self, tracker):
        """Test required landmarks list."""
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "left_wrist" in required
        assert "right_shoulder" in required
        assert "right_wrist" in required


class TestFrontRaiseTracker:
    """Tests for FrontRaiseTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return FrontRaiseTracker(track_side="left")
    
    def test_initialization(self, tracker):
        """Test initial state."""
        assert tracker.rep_count == 0
        assert tracker.stage == "down"
    
    def test_required_landmarks(self, tracker):
        """Test required landmarks list."""
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "left_wrist" in required


class TestShoulderShrugTracker:
    """Tests for ShoulderShrugTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return ShoulderShrugTracker()
    
    def test_initialization(self, tracker):
        """Test initial state."""
        assert tracker.rep_count == 0
        assert tracker.stage == "down"
    
    def test_required_landmarks(self, tracker):
        """Test required landmarks list."""
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "right_shoulder" in required
        assert "left_ear" in required
        assert "right_ear" in required


class TestTricepExtensionTracker:
    """Tests for TricepExtensionTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return TricepExtensionTracker(track_side="left")
    
    def test_initialization(self, tracker):
        """Test initial state."""
        assert tracker.rep_count == 0
        assert tracker.stage == "down"
    
    def test_required_landmarks(self, tracker):
        """Test required landmarks list."""
        required = tracker.get_required_landmarks()
        assert "left_shoulder" in required
        assert "left_elbow" in required
        assert "left_wrist" in required


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
