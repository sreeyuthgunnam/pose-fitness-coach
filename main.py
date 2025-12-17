"""
Pose-Based Fitness Coach
========================
Main entry point for the fitness coaching application.

Usage:
    python main.py [--mode MODE]
    
Modes:
    opencv   - Run with OpenCV window (default)
    streamlit - Run with Streamlit dashboard
"""

import argparse
import sys


def main():
    """Main entry point for the Fitness Coach application."""
    parser = argparse.ArgumentParser(
        description="Pose-Based Fitness Coach - Real-time exercise tracking with AI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["opencv", "streamlit"],
        default="opencv",
        help="UI mode: 'opencv' for simple window, 'streamlit' for web dashboard"
    )
    parser.add_argument(
        "--exercise",
        type=str,
        choices=[
            "bicep_curl", "squat", "pushup",
            "shoulder_press", "lateral_raise", "front_raise",
            "shoulder_shrug", "tricep_extension"
        ],
        default="bicep_curl",
        help="Starting exercise type"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice feedback"
    )
    
    args = parser.parse_args()
    
    if args.mode == "opencv":
        # Run OpenCV-based application
        from src.app import FitnessCoachApp
        app = FitnessCoachApp(
            starting_exercise=args.exercise,
            voice_enabled=args.voice
        )
        app.run()
    elif args.mode == "streamlit":
        # Launch Streamlit app
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/streamlit_app.py",
            "--server.headless", "true"
        ])


if __name__ == "__main__":
    main()
