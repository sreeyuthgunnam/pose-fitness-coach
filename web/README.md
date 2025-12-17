# 🏋️ Pose Fitness Coach

AI-powered fitness coach that tracks your exercises in real-time using your webcam.

![Next.js](https://img.shields.io/badge/Next.js-16-black)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🎥 **Real-time pose detection** - Uses MediaPipe for accurate body tracking
- 🔢 **Automatic rep counting** - Tracks your exercise repetitions
- 📊 **Form feedback** - Get instant feedback on your form
- 🌐 **Browser-based** - No installation required, works in any modern browser
- 📱 **Responsive** - Works on desktop and mobile devices
- 🔒 **Privacy-first** - All processing happens locally in your browser

## 🏃 Exercises

| Exercise | Description | Upper Body Only |
|----------|-------------|:---------------:|
| 💪 Bicep Curl | Arm curls for bicep strength | ✅ |
| 🏋️ Shoulder Press | Overhead press for shoulders | ✅ |
| ↔️ Lateral Raise | Side raises for deltoids | ✅ |
| ⬆️ Front Raise | Front raises for anterior deltoids | ✅ |
| 🤷 Shoulder Shrug | Shrugs for trapezius | ✅ |
| 🦵 Squat | Squats for leg strength | ❌ |

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

## 🌐 Deploy to Vercel

The easiest way to deploy is using [Vercel](https://vercel.com):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/pose-fitness-coach)

### Manual Deployment

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Import your GitHub repository
4. Click Deploy!

## 🛠️ Tech Stack

- **[Next.js 16](https://nextjs.org/)** - React framework
- **[MediaPipe Tasks Vision](https://developers.google.com/mediapipe)** - ML pose detection
- **[Tailwind CSS](https://tailwindcss.com/)** - Styling
- **TypeScript** - Type safety

## 📋 Requirements

- Modern web browser with WebGL support
- Webcam access
- Chrome or Edge recommended for best performance

## 📄 License

MIT License - feel free to use this project for any purpose.

---

Made with ❤️ using AI-powered pose detection
