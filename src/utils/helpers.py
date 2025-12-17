"""
Helper Utilities
================
General helper functions for the Fitness Coach application.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
import time


def calculate_angle(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    point3: Tuple[float, float]
) -> float:
    """
    Calculate the angle at point2 between point1 and point3.
    
    Args:
        point1: First point (x, y)
        point2: Middle point (vertex)
        point3: Third point
        
    Returns:
        Angle in degrees (0-180)
    """
    a = np.array([point1[0], point1[1]])
    b = np.array([point2[0], point2[1]])
    c = np.array([point3[0], point3[1]])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cosine_angle))


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
        Distance
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def normalize_coordinates(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int
) -> Tuple[float, float]:
    """
    Normalize pixel coordinates to 0-1 range.
    
    Args:
        x: X coordinate in pixels
        y: Y coordinate in pixels
        frame_width: Frame width
        frame_height: Frame height
        
    Returns:
        Normalized (x, y) tuple
    """
    return (x / frame_width, y / frame_height)


def denormalize_coordinates(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int
) -> Tuple[int, int]:
    """
    Convert normalized coordinates to pixel coordinates.
    
    Args:
        x: Normalized x (0-1)
        y: Normalized y (0-1)
        frame_width: Frame width
        frame_height: Frame height
        
    Returns:
        Pixel (x, y) tuple
    """
    return (int(x * frame_width), int(y * frame_height))


class SmoothingFilter:
    """
    Simple moving average filter for smoothing values.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the filter.
        
        Args:
            window_size: Number of samples to average
        """
        self.window_size = window_size
        self.values: List[float] = []
    
    def add(self, value: float) -> float:
        """
        Add a value and return the smoothed result.
        
        Args:
            value: New value to add
            
        Returns:
            Smoothed value
        """
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return np.mean(self.values)
    
    def reset(self):
        """Clear the filter history."""
        self.values.clear()
    
    @property
    def current(self) -> float:
        """Get current smoothed value."""
        return np.mean(self.values) if self.values else 0


class FPSCounter:
    """
    FPS counter for performance monitoring.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize the FPS counter.
        
        Args:
            window_size: Number of frames to average
        """
        self.window_size = window_size
        self.times: List[float] = []
        self.last_time = time.time()
    
    def tick(self) -> float:
        """
        Record a frame and return current FPS.
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        
        if delta > 0:
            self.times.append(1 / delta)
            if len(self.times) > self.window_size:
                self.times.pop(0)
        
        return np.mean(self.times) if self.times else 0
    
    @property
    def fps(self) -> float:
        """Get current FPS."""
        return np.mean(self.times) if self.times else 0


class RateLimiter:
    """
    Rate limiter for controlling action frequency.
    """
    
    def __init__(self, min_interval: float = 1.0):
        """
        Initialize the rate limiter.
        
        Args:
            min_interval: Minimum time between actions (seconds)
        """
        self.min_interval = min_interval
        self.last_time = 0
    
    def can_act(self) -> bool:
        """
        Check if enough time has passed for another action.
        
        Returns:
            True if action is allowed
        """
        current_time = time.time()
        if current_time - self.last_time >= self.min_interval:
            self.last_time = current_time
            return True
        return False
    
    def reset(self):
        """Reset the limiter."""
        self.last_time = 0


def format_time(seconds: float) -> str:
    """
    Format seconds into MM:SS string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def interpolate_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    t: float
) -> Tuple[int, int, int]:
    """
    Interpolate between two colors.
    
    Args:
        color1: First color (B, G, R)
        color2: Second color (B, G, R)
        t: Interpolation factor (0-1)
        
    Returns:
        Interpolated color
    """
    t = clamp(t, 0, 1)
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t)
    )


def get_score_color(score: float) -> Tuple[int, int, int]:
    """
    Get a color based on score (green for high, red for low).
    
    Args:
        score: Score (0-100)
        
    Returns:
        BGR color tuple
    """
    # BGR format
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    
    if score >= 70:
        return interpolate_color(yellow, green, (score - 70) / 30)
    elif score >= 40:
        return interpolate_color(red, yellow, (score - 40) / 30)
    else:
        return red
