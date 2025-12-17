"""
Streamlit Fitness Coach Dashboard
=================================
Web-based dashboard for the Pose-Based Fitness Coach.

Features:
- Live webcam feed with pose overlay
- Exercise selection (8 exercises including half-body friendly)
- Real-time rep counting
- Form score gauge
- Session history

Run with:
    streamlit run src/streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional
import time
from dataclasses import dataclass

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Fitness Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import exercise trackers using relative imports that work when run from project root
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import exercise modules
from src.exercises.bicep_curl import BicepCurlTracker
from src.exercises.squat import SquatTracker
from src.exercises.pushup import PushUpTracker
from src.exercises.shoulder_press import ShoulderPressTracker
from src.exercises.lateral_raise import LateralRaiseTracker
from src.exercises.front_raise import FrontRaiseTracker
from src.exercises.shoulder_shrug import ShoulderShrugTracker
from src.exercises.tricep_extension import TricepExtensionTracker
from src.exercises.base import ExerciseResult


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .rep-counter {
        font-size: 5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .stage-indicator {
        font-size: 1.5rem;
        text-align: center;
        padding: 0.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stage-up {
        background-color: #FFF3E0;
        color: #E65100;
    }
    .stage-down {
        background-color: #E3F2FD;
        color: #1565C0;
    }
    .feedback-good {
        color: #4CAF50;
        font-weight: bold;
    }
    .feedback-warning {
        color: #FF9800;
        font-weight: bold;
    }
    .feedback-bad {
        color: #F44336;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Exercise configuration
EXERCISES = {
    'bicep_curl': {
        'name': 'Bicep Curl 💪',
        'tracker': BicepCurlTracker,
        'tips': [
            "• Stand with arm at your side",
            "• Keep elbow close to body",
            "• Curl weight up to shoulder",
            "• Full extension at bottom"
        ],
        'half_body': True
    },
    'shoulder_press': {
        'name': 'Shoulder Press 🏋️',
        'tracker': ShoulderPressTracker,
        'tips': [
            "• Start with hands at shoulder level",
            "• Press straight up overhead",
            "• Full arm extension at top",
            "• Control the descent"
        ],
        'half_body': True
    },
    'lateral_raise': {
        'name': 'Lateral Raise ↔️',
        'tracker': LateralRaiseTracker,
        'tips': [
            "• Arms at your sides",
            "• Raise arms out to shoulder level",
            "• Keep slight bend in elbows",
            "• Lower with control"
        ],
        'half_body': True
    },
    'front_raise': {
        'name': 'Front Raise ⬆️',
        'tracker': FrontRaiseTracker,
        'tips': [
            "• Arms at your sides",
            "• Raise arms in front to shoulder level",
            "• Keep arms straight",
            "• Lower with control"
        ],
        'half_body': True
    },
    'shoulder_shrug': {
        'name': 'Shoulder Shrug 🤷',
        'tracker': ShoulderShrugTracker,
        'tips': [
            "• Stand relaxed",
            "• Shrug shoulders up to ears",
            "• Hold briefly at top",
            "• Lower smoothly"
        ],
        'half_body': True
    },
    'tricep_extension': {
        'name': 'Tricep Extension 💪',
        'tracker': TricepExtensionTracker,
        'tips': [
            "• Hold weight overhead",
            "• Keep upper arm stationary",
            "• Lower weight behind head",
            "• Extend arm fully"
        ],
        'half_body': True
    },
    'squat': {
        'name': 'Squat 🦵',
        'tracker': SquatTracker,
        'tips': [
            "• Stand with feet shoulder-width",
            "• Keep back straight",
            "• Bend knees to 90°",
            "• Don't let knees go over toes"
        ],
        'half_body': False
    },
    'pushup': {
        'name': 'Push-up 🔥',
        'tracker': PushUpTracker,
        'tips': [
            "• Use SIDE VIEW for best tracking",
            "• Keep body in straight line",
            "• Lower chest to ground",
            "• Full arm extension at top"
        ],
        'half_body': False
    }
}

# Landmark indices for MediaPipe
LANDMARK_NAMES = {
    0: "nose", 
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow", 
    15: "left_wrist", 16: "right_wrist", 
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee", 
    27: "left_ankle", 28: "right_ankle"
}


def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_exercise = 'bicep_curl'
        st.session_state.rep_count = 0
        st.session_state.form_score = 100
        st.session_state.feedback = "Ready to start! Click 'Start Camera' to begin."
        st.session_state.stage = "neutral"
        st.session_state.session_history = {ex: 0 for ex in EXERCISES.keys()}
        st.session_state.camera_running = False
        st.session_state.trackers = {name: config['tracker']() for name, config in EXERCISES.items()}


def get_tracker():
    """Get the current exercise tracker."""
    return st.session_state.trackers[st.session_state.current_exercise]


def reset_tracker():
    """Reset the current tracker."""
    tracker = get_tracker()
    tracker.reset()
    st.session_state.rep_count = 0
    st.session_state.form_score = 100
    st.session_state.feedback = "Counter reset!"
    st.session_state.stage = "neutral"


def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.markdown("# 🏋️ Controls")
    
    # Combined selector
    all_exercise_names = {name: config['name'] for name, config in EXERCISES.items()}
    
    selected = st.sidebar.selectbox(
        "Select Exercise",
        options=list(all_exercise_names.keys()),
        format_func=lambda x: f"{'📸 ' if EXERCISES[x]['half_body'] else '🏃 '}{all_exercise_names[x]}",
        index=list(all_exercise_names.keys()).index(st.session_state.current_exercise)
    )
    
    if selected != st.session_state.current_exercise:
        st.session_state.current_exercise = selected
        st.session_state.rep_count = st.session_state.trackers[selected].rep_count
        st.session_state.form_score = 100
        st.session_state.feedback = f"Switched to {EXERCISES[selected]['name']}"
        st.session_state.stage = st.session_state.trackers[selected].stage
    
    st.sidebar.markdown("---")
    
    # Reset button
    if st.sidebar.button("🔄 Reset Counter", use_container_width=True):
        reset_tracker()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Session history
    st.sidebar.markdown("### 📊 Session History")
    
    for ex_name, count in st.session_state.session_history.items():
        if count > 0:
            st.sidebar.markdown(f"- {EXERCISES[ex_name]['name']}: **{count}**")
    
    total_reps = sum(st.session_state.session_history.values())
    st.sidebar.markdown(f"**Total Reps: {total_reps}**")
    
    st.sidebar.markdown("---")
    
    # Tips for current exercise
    st.sidebar.markdown("### 📝 Tips")
    for tip in EXERCISES[st.session_state.current_exercise]['tips']:
        st.sidebar.markdown(tip)
    
    if EXERCISES[st.session_state.current_exercise]['half_body']:
        st.sidebar.success("✅ Half-body friendly - works with webcam!")


def render_metrics():
    """Render the metrics display."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>REPS</h4>
            <div class="rep-counter">{st.session_state.rep_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        stage = st.session_state.stage
        stage_class = "stage-up" if stage.lower() in ['up', 'standing', 'raised', 'extended'] else "stage-down"
        st.markdown(f"""
        <div class="metric-card">
            <h4>STAGE</h4>
            <div class="stage-indicator {stage_class}">{stage.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>FORM SCORE</h4>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(st.session_state.form_score) / 100)
        
        score = st.session_state.form_score
        color = "#4CAF50" if score >= 70 else "#FF9800" if score >= 40 else "#F44336"
        st.markdown(f"<center><b style='color:{color}'>{int(score)}%</b></center>", unsafe_allow_html=True)


def render_feedback():
    """Render the feedback message."""
    feedback = st.session_state.feedback
    score = st.session_state.form_score
    
    if score >= 70 or '✓' in feedback or '💪' in feedback or 'Good' in feedback:
        color = "#4CAF50"
    elif score >= 40:
        color = "#FF9800"
    else:
        color = "#F44336"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
        <span style="font-size: 1.2rem; font-weight: bold; color: {color};">{feedback}</span>
    </div>
    """, unsafe_allow_html=True)


def process_frame(frame, pose, tracker):
    """Process a single frame with pose detection."""
    h, w = frame.shape[:2]
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    result = None
    
    if results.pose_landmarks:
        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
        
        # Extract landmarks
        landmarks = {}
        for idx, name in LANDMARK_NAMES.items():
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                landmarks[name] = (lm.x, lm.y, lm.z, lm.visibility)
        
        # Process with tracker
        result = tracker.process(landmarks, w, h)
    
    return frame, result


def draw_frame_overlay(frame):
    """Draw information overlay on frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Exercise name
    ex_name = EXERCISES[st.session_state.current_exercise]['name']
    cv2.putText(frame, ex_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (66, 133, 244), 2)
    
    # Rep count
    cv2.putText(frame, f"Reps: {st.session_state.rep_count}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (76, 175, 80), 2)
    
    # Stage
    cv2.putText(frame, f"Stage: {st.session_state.stage.upper()}", (20, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 193, 7), 2)
    
    # Form score
    score = st.session_state.form_score
    score_color = (76, 175, 80) if score >= 70 else (255, 193, 7) if score >= 40 else (244, 67, 54)
    cv2.putText(frame, f"Form: {int(score)}%", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)


def run_camera_feed():
    """Run the camera feed with pose detection using Streamlit's native approach."""
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        st.error("❌ Could not open camera. Please check your camera connection.")
        return
    
    # Create placeholders
    video_placeholder = st.empty()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        stop_button_placeholder = st.empty()
    
    metrics_placeholder = st.empty()
    feedback_placeholder = st.empty()
    
    # Get current tracker
    tracker = get_tracker()
    prev_rep_count = tracker.rep_count
    
    st.session_state.camera_running = True
    
    try:
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, result = process_frame(frame, pose, tracker)
            
            if result:
                # Update session state
                if result.rep_count > prev_rep_count:
                    diff = result.rep_count - prev_rep_count
                    st.session_state.session_history[st.session_state.current_exercise] += diff
                    prev_rep_count = result.rep_count
                
                st.session_state.rep_count = result.rep_count
                st.session_state.form_score = result.form_score
                st.session_state.feedback = result.feedback
                st.session_state.stage = result.stage
            
            # Draw info on frame
            draw_frame_overlay(processed_frame)
            
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Stop button
            if stop_button_placeholder.button("🛑 Stop Camera", key=f"stop_{time.time()}"):
                st.session_state.camera_running = False
                break
            
            # Update metrics
            with metrics_placeholder.container():
                render_metrics()
            
            with feedback_placeholder.container():
                render_feedback()
            
            # Small delay
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        cap.release()
        pose.close()
        st.session_state.camera_running = False
        st.success("✅ Camera stopped. Click 'Start Camera' to begin again.")


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🏋️ Pose-Based Fitness Coach</div>', unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.markdown(f"### Currently: {EXERCISES[st.session_state.current_exercise]['name']}")
    
    if EXERCISES[st.session_state.current_exercise]['half_body']:
        st.success("✅ This exercise works great with a webcam showing your upper body!")
    else:
        st.warning("⚠️ This exercise requires full body visibility for best tracking.")
    
    st.markdown("---")
    
    # Camera controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        start_camera = st.button("📹 Start Camera", use_container_width=True, type="primary")
    
    with col2:
        demo_mode = st.button("🎮 Demo Mode (No Camera)", use_container_width=True)
    
    st.markdown("---")
    
    if start_camera:
        run_camera_feed()
    elif demo_mode:
        st.markdown("### 🎮 Demo Mode")
        st.info("Use the buttons below to simulate exercise tracking.")
        
        # Render current metrics
        render_metrics()
        render_feedback()
        
        st.markdown("---")
        
        # Demo controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("➕ Add Rep", use_container_width=True):
                tracker = get_tracker()
                tracker.rep_count += 1
                st.session_state.rep_count = tracker.rep_count
                st.session_state.session_history[st.session_state.current_exercise] += 1
                st.session_state.feedback = "Great rep! 💪"
                st.session_state.form_score = 95
                st.rerun()
        
        with col2:
            if st.button("🔄 Toggle Stage", use_container_width=True):
                if st.session_state.stage in ['up', 'standing', 'raised', 'extended']:
                    st.session_state.stage = "down"
                else:
                    st.session_state.stage = "up"
                st.rerun()
        
        with col3:
            if st.button("📉 Bad Form", use_container_width=True):
                st.session_state.form_score = max(0, st.session_state.form_score - 20)
                st.session_state.feedback = "Watch your form!"
                st.rerun()
    
    else:
        # Show instructions when no mode selected
        st.markdown("### 👆 Select a mode above to get started!")
        
        st.markdown("""
        **📹 Start Camera**: Uses your webcam for real-time pose detection and exercise tracking.
        
        **🎮 Demo Mode**: Test the interface without a camera.
        
        ---
        
        ### 💡 Half-Body Friendly Exercises (📸)
        
        These exercises work great even if your camera only shows your upper body:
        """)
        
        half_body_list = [f"- **{config['name']}**" for name, config in EXERCISES.items() if config['half_body']]
        st.markdown("\n".join(half_body_list))
        
        st.markdown("""
        ---
        
        ### 🏃 Full-Body Exercises
        
        These exercises require your full body to be visible:
        """)
        
        full_body_list = [f"- **{config['name']}**" for name, config in EXERCISES.items() if not config['half_body']]
        st.markdown("\n".join(full_body_list))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Made with ❤️ using MediaPipe & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
