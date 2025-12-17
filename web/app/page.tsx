'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { initPoseDetector, detectPose, drawPose, Pose } from '../lib/poseDetector';
import { EXERCISES, ExerciseTracker, ExerciseResult, ExerciseConfig, Keypoint } from '../lib/exercises';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const trackerRef = useRef<ExerciseTracker | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [selectedExercise, setSelectedExercise] = useState<ExerciseConfig>(EXERCISES[0]);
  const [result, setResult] = useState<ExerciseResult>({
    repCount: 0,
    stage: 'neutral',
    feedback: 'Select an exercise and start!',
    formScore: 100,
  });
  const [sessionHistory, setSessionHistory] = useState<Record<string, number>>({});

  // Initialize pose detector
  useEffect(() => {
    const init = async () => {
      try {
        await initPoseDetector();
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to load pose detector:', error);
        setCameraError('Failed to load AI model. Please refresh the page.');
        setIsLoading(false);
      }
    };
    init();
  }, []);

  // Initialize tracker when exercise changes
  useEffect(() => {
    trackerRef.current = new ExerciseTracker(selectedExercise.id);
    setResult({
      repCount: 0,
      stage: 'neutral',
      feedback: `Ready for ${selectedExercise.name}!`,
      formScore: 100,
    });
  }, [selectedExercise]);

  // Start camera and detection
  const startCamera = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      // Set canvas size
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      setIsRunning(true);
      setCameraError(null);

      // Start detection loop
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      let lastRepCount = 0;
      let lastTime = performance.now();

      const detect = async () => {
        if (!videoRef.current || !canvasRef.current) return;

        const currentTime = performance.now();
        const pose = await detectPose(videoRef.current, currentTime);

        // Draw skeleton
        if (pose) {
          drawPose(ctx, pose, canvas.width, canvas.height);
        } else {
          // Clear canvas if no pose detected
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        // Process exercise
        if (pose && trackerRef.current) {
          const keypoints = pose.keypoints as Keypoint[];
          const exerciseResult = trackerRef.current.process(keypoints);
          
          // Update rep count in session history
          if (exerciseResult.repCount > lastRepCount) {
            const diff = exerciseResult.repCount - lastRepCount;
            lastRepCount = exerciseResult.repCount;
            setSessionHistory((prev: Record<string, number>) => ({
              ...prev,
              [selectedExercise.id]: (prev[selectedExercise.id] || 0) + diff,
            }));
          }

          setResult(exerciseResult);
        }

        animationRef.current = requestAnimationFrame(detect);
      };

      detect();
    } catch (error) {
      console.error('Camera error:', error);
      setCameraError('Could not access camera. Please allow camera permissions.');
    }
  }, [isRunning, selectedExercise.id]);

  // Stop camera
  const stopCamera = useCallback(() => {
    setIsRunning(false);

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  }, []);

  // Reset counter
  const resetCounter = useCallback(() => {
    if (trackerRef.current) {
      trackerRef.current.reset();
      setResult({
        repCount: 0,
        stage: 'neutral',
        feedback: 'Counter reset!',
        formScore: 100,
      });
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Get form score color
  const getScoreColor = (score: number) => {
    if (score >= 70) return 'text-green-500';
    if (score >= 40) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getScoreBg = (score: number) => {
    if (score >= 70) return 'bg-green-500';
    if (score >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  // Calculate total reps
  const totalReps = Object.values(sessionHistory).reduce((a: number, b: number) => a + b, 0);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-lg text-gray-600">Loading AI Model...</p>
          <p className="text-sm text-gray-400 mt-2">This may take a few seconds</p>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen p-4 md:p-8">
      {/* Header */}
      <header className="text-center mb-6">
        <h1 className="text-3xl md:text-4xl font-bold text-blue-600 mb-2">
          🏋️ Pose Fitness Coach
        </h1>
        <p className="text-gray-600">AI-powered exercise tracking in your browser</p>
      </header>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Video Area */}
        <div className="lg:col-span-2">
          {/* Video Container */}
          <div className="video-container bg-gray-900 aspect-video relative mb-4">
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]"
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]"
            />
            
            {!isRunning && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                <div className="text-center text-white">
                  <p className="text-xl mb-4">📹 Camera Off</p>
                  <p className="text-gray-400">Click "Start Camera" to begin</p>
                </div>
              </div>
            )}

            {cameraError && (
              <div className="absolute inset-0 flex items-center justify-center bg-red-900/80">
                <div className="text-center text-white p-4">
                  <p className="text-xl mb-2">⚠️ Camera Error</p>
                  <p className="text-sm">{cameraError}</p>
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex gap-3 mb-4">
            {!isRunning ? (
              <button
                onClick={startCamera}
                className="flex-1 bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                📹 Start Camera
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="flex-1 bg-red-500 hover:bg-red-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                🛑 Stop Camera
              </button>
            )}
            <button
              onClick={resetCounter}
              className="bg-gray-200 hover:bg-gray-300 text-gray-700 font-semibold py-3 px-6 rounded-lg transition-colors"
            >
              🔄 Reset
            </button>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-3 gap-4">
            {/* Rep Counter */}
            <div className={`bg-white rounded-xl p-4 shadow-lg text-center ${isRunning ? 'rep-counter active' : ''}`}>
              <p className="text-sm text-gray-500 mb-1">REPS</p>
              <p className="text-5xl font-bold text-green-500">{result.repCount}</p>
            </div>

            {/* Stage */}
            <div className="bg-white rounded-xl p-4 shadow-lg text-center">
              <p className="text-sm text-gray-500 mb-1">STAGE</p>
              <p className={`text-2xl font-semibold ${
                result.stage === 'up' ? 'text-orange-500' : 'text-blue-500'
              }`}>
                {result.stage.toUpperCase()}
              </p>
            </div>

            {/* Form Score */}
            <div className="bg-white rounded-xl p-4 shadow-lg text-center">
              <p className="text-sm text-gray-500 mb-1">FORM</p>
              <p className={`text-3xl font-bold ${getScoreColor(result.formScore)}`}>
                {result.formScore}%
              </p>
              <div className="w-full h-2 bg-gray-200 rounded-full mt-2">
                <div 
                  className={`h-full rounded-full transition-all ${getScoreBg(result.formScore)}`}
                  style={{ width: `${result.formScore}%` }}
                />
              </div>
            </div>
          </div>

          {/* Feedback */}
          <div className="mt-4 bg-white rounded-xl p-4 shadow-lg">
            <p className={`text-center text-lg font-medium ${getScoreColor(result.formScore)}`}>
              {result.feedback}
            </p>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Exercise Selection */}
          <div className="bg-white rounded-xl p-4 shadow-lg">
            <h2 className="font-semibold text-gray-700 mb-3">Select Exercise</h2>
            <div className="space-y-2">
              {EXERCISES.map((exercise) => (
                <button
                  key={exercise.id}
                  onClick={() => {
                    setSelectedExercise(exercise);
                  }}
                  className={`exercise-card w-full text-left p-3 rounded-lg border-2 transition-all ${
                    selectedExercise.id === exercise.id
                      ? 'border-blue-500 bg-blue-50 selected'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{exercise.emoji}</span>
                    <div>
                      <p className="font-medium text-gray-800">{exercise.name}</p>
                      <p className="text-xs text-gray-500">
                        {exercise.halfBody ? '📸 Half-body' : '🏃 Full-body'}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Tips */}
          <div className="bg-white rounded-xl p-4 shadow-lg">
            <h2 className="font-semibold text-gray-700 mb-3">💡 Tips</h2>
            <ul className="space-y-2 text-sm text-gray-600">
              {selectedExercise.tips.map((tip, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  {tip}
                </li>
              ))}
            </ul>
            {selectedExercise.halfBody && (
              <p className="mt-3 text-xs text-green-600 bg-green-50 p-2 rounded">
                ✅ Works with upper body only!
              </p>
            )}
          </div>

          {/* Session History */}
          <div className="bg-white rounded-xl p-4 shadow-lg">
            <h2 className="font-semibold text-gray-700 mb-3">📊 Session</h2>
            {Object.keys(sessionHistory).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(sessionHistory).map(([id, count]) => {
                  const ex = EXERCISES.find(e => e.id === id);
                  return (
                    <div key={id} className="flex justify-between text-sm">
                      <span className="text-gray-600">{ex?.emoji} {ex?.name}</span>
                      <span className="font-semibold text-gray-800">{count}</span>
                    </div>
                  );
                })}
                <div className="border-t pt-2 mt-2 flex justify-between font-semibold">
                  <span>Total Reps</span>
                  <span className="text-green-500">{totalReps}</span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-400">Complete some reps to see history</p>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="text-center mt-8 text-sm text-gray-400">
        Made with ❤️ using TensorFlow.js & Next.js
      </footer>
    </main>
  );
}
