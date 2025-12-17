# 🏋️ Pose-Based Fitness Coach

> Real-time AI-powered fitness coaching using computer vision

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Overview

The **Pose-Based Fitness Coach** is an AI-powered application that uses your webcam to track exercises in real-time. It counts repetitions, analyzes your form, and provides instant feedback to help you exercise safely and effectively.

**Two versions available:**
- 🖥️ **Desktop App** (Python) - Full-featured with voice feedback
- 🌐 **Web App** (Next.js) - Browser-based, deploy to Vercel

## ✨ Features

- 🎥 **Real-time pose detection** using MediaPipe
- 🔢 **Automatic rep counting** for multiple exercises
- 📊 **Form analysis** with instant feedback
- 🎯 **Form score** (0-100%) for each rep
- 🔊 **Voice feedback** (optional)
- 🖥️ **Two UI modes**: Simple OpenCV window or Streamlit dashboard

## 🏃 Supported Exercises

### Half-Body Friendly (📸 Works with upper body only)

| Exercise | Key Points Tracked | Feedback Provided |
|----------|-------------------|-------------------|
| **Bicep Curls** | Shoulder, Elbow, Wrist | Elbow stability, range of motion |
| **Shoulder Press** | Shoulder, Elbow, Wrist | Arm extension, vertical path |
| **Lateral Raise** | Shoulder, Elbow, Wrist | Arm height, both arms tracked |
| **Front Raise** | Shoulder, Wrist | Wrist height, arm straightness |
| **Shoulder Shrug** | Shoulder, Nose | Shoulder elevation, hold time |
| **Tricep Extension** | Shoulder, Elbow, Wrist | Upper arm stability, extension |

### Full-Body Required (🏃 Need full body visible)

| Exercise | Key Points Tracked | Feedback Provided |
|----------|-------------------|-------------------|
| **Squats** | Hip, Knee, Ankle | Knee position, depth, back angle |
| **Push-ups** | Shoulder, Elbow, Wrist, Hip | Body alignment, depth, elbow flare |

## 🚀 Quick Start

### Option 1: Web App (Recommended for quick start)

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) - works in any modern browser!

### Option 2: Desktop App (Python)

#### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/macOS/Linux

#### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pose-fitness-coach.git
   cd pose-fitness-coach
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

#### Usage

**OpenCV Mode (Simple Window)**
```bash
python main.py --mode opencv
```

**Streamlit Mode (Web Dashboard)**
```bash
python main.py --mode streamlit
# or directly:
streamlit run src/streamlit_app.py
```

**Command Line Options**
```bash
python main.py --help

Options:
  --mode {opencv,streamlit}  UI mode (default: opencv)
  --exercise {bicep_curl,squat,pushup,shoulder_press,lateral_raise,front_raise,shoulder_shrug,tricep_extension}
  --voice  Enable voice feedback
```

## ⌨️ Controls (OpenCV Mode)

| Key | Action |
|-----|--------|
| `1` | Switch to Bicep Curls |
| `2` | Switch to Squats |
| `3` | Switch to Push-ups |
| `4` | Switch to Shoulder Press |
| `5` | Switch to Lateral Raise |
| `6` | Switch to Front Raise |
| `7` | Switch to Shoulder Shrug |
| `8` | Switch to Tricep Extension |
| `r` | Reset rep counter |
| `v` | Toggle voice feedback |
| `q` | Quit application |

## 📁 Project Structure

```
pose-fitness-coach/
├── main.py                 # Python app entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # MIT License
├── src/                   # Python source code
│   ├── __init__.py
│   ├── app.py             # Main OpenCV application
│   ├── streamlit_app.py   # Streamlit dashboard
│   ├── pose_detector.py   # MediaPipe pose detection
│   ├── exercises/         # Exercise trackers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── bicep_curl.py
│   │   ├── squat.py
│   │   ├── pushup.py
│   │   ├── shoulder_press.py
│   │   ├── lateral_raise.py
│   │   ├── front_raise.py
│   │   ├── shoulder_shrug.py
│   │   └── tricep_extension.py
│   └── utils/
│       ├── __init__.py
│       ├── validators.py
│       └── helpers.py
├── web/                   # Next.js web app
│   ├── app/               # Next.js app directory
│   ├── lib/               # TypeScript utilities
│   ├── package.json
│   └── README.md
├── tests/
│   ├── __init__.py
│   ├── test_pose_detector.py
│   └── test_exercises.py
├── docs/
│   └── adding_exercises.md
└── models/                # Reserved for custom models
```

## 🌐 Web App Deployment

The web app can be deployed to Vercel for free:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Set **Root Directory** to `web`
   - Click Deploy!

See [web/README.md](web/README.md) for more details.

## 🔧 Adding New Exercises

See [docs/adding_exercises.md](docs/adding_exercises.md) for a detailed guide on how to add new exercise trackers.

Quick overview:
1. Create a new file in `src/exercises/`
2. Inherit from `BaseExerciseTracker`
3. Implement `process_frame()` and `get_feedback()` methods
4. Register the exercise in `src/exercises/__init__.py`

## ⚠️ Known Limitations

- Works best with a **side view** for push-ups
- Requires **good lighting** for accurate detection
- Single person tracking only (picks largest person if multiple detected)
- May have reduced accuracy with **loose/baggy clothing**

## 🔮 Future Improvements

- [ ] Add more exercises (lunges, planks, jumping jacks)
- [ ] Workout session recording and playback
- [ ] Progress tracking over time
- [ ] Custom exercise builder
- [ ] Mobile app version
- [ ] Multi-person support

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the amazing pose detection
- [OpenCV](https://opencv.org/) for computer vision utilities
- [Streamlit](https://streamlit.io/) for the web dashboard framework

---

Made with ❤️ for fitness enthusiasts and developers
