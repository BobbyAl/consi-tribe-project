import React, { useState, useEffect, useRef } from 'react';

// We will import your generated JSON data directly
import textEmotions from '../../data/text_emotions.json';
import faceEmotions from '../../data/face_emtions.json';
import audioFeatures from '../../data/audio_features.json';

export default function App() {
  const videoRef = useRef(null);
  const [currentSecond, setCurrentSecond] = useState("0");

  // 1. Sync React State to Video Time 
  useEffect(() => {
    let animationFrameId;
    
    const updateTime = () => {
      if (videoRef.current) {
        // Floor the current time to match our JSON keys ("0", "1", "2"...)
        const sec = Math.floor(videoRef.current.currentTime).toString();
        if (sec !== currentSecond) {
          setCurrentSecond(sec);
        }
      }
      animationFrameId = requestAnimationFrame(updateTime);
    };

    animationFrameId = requestAnimationFrame(updateTime);
    return () => cancelAnimationFrame(animationFrameId);
  }, [currentSecond]);

  // 2. Safely grab the current data for the active second
  const currentFace = faceEmotions[currentSecond] || {};
  const currentAudio = audioFeatures[currentSecond] || { audio_emotions: {}, audio_acoustics: {} };
  const currentText = textEmotions[currentSecond] || {};

  // Optional: Extract top text emotion for subtitle display
  const topTextEmotion = Object.entries(currentText)
    .sort((a, b) => b[1] - a[1])[0];

  return (
    <div className="min-h-screen bg-neutral-950 text-white p-6 font-sans flex flex-col">
      {/* Header */}
      <header className="mb-6 border-b border-neutral-800 pb-4">
        <h1 className="text-2xl font-light tracking-widest text-neutral-300">
          IN-SILICO NEUROSCIENCE <span className="font-bold text-white">TRIBEv2</span>
        </h1>
        <p className="text-sm text-neutral-500 uppercase tracking-wider">Cinematic Multisensory Integration Ablation Study</p>
      </header>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-3 gap-6 flex-grow">
        
        {/* LEFT COLUMN (Video & Text) */}
        <div className="col-span-2 flex flex-col gap-6">
          {/* Video Player */}
          <div className="relative bg-black rounded-xl overflow-hidden shadow-2xl border border-neutral-800 aspect-video flex items-center justify-center">
            <video 
              ref={videoRef}
              src="/clip.mp4" /* Put a copy of clip.mp4 in the viz-app/public folder */
              controls
              className="w-full h-full object-contain"
            />
          </div>

          {/* Transcript / Dialogue Area */}
          <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800 h-48 flex flex-col">
            <h2 className="text-xs uppercase text-neutral-500 mb-2 font-semibold tracking-wider">Semantic Processing</h2>
            <div className="flex-grow flex items-center justify-between">
                <p className="text-2xl text-neutral-300 italic">
                    "Subtitles will sync here..."
                </p>
                {topTextEmotion && (
                    <div className="px-4 py-2 bg-blue-900/30 text-blue-400 rounded-full border border-blue-800/50 text-sm">
                        {topTextEmotion[0]} {(topTextEmotion[1] * 100).toFixed(0)}%
                    </div>
                )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN (Data Viz) */}
        <div className="col-span-1 flex flex-col gap-6">
          
          {/* Facial Emotions */}
          <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800 flex-grow">
            <h2 className="text-xs uppercase text-neutral-500 mb-4 font-semibold tracking-wider">Facial Micro-Expressions</h2>
            <div className="h-full flex items-center justify-center border border-dashed border-neutral-700 rounded text-neutral-600">
                [ Recharts LineGraph Here ]
            </div>
          </div>

          {/* Audio Features */}
          <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800 flex-grow">
            <h2 className="text-xs uppercase text-neutral-500 mb-4 font-semibold tracking-wider">Acoustic & Vocal Sentiment</h2>
            <div className="h-full flex items-center justify-center border border-dashed border-neutral-700 rounded text-neutral-600">
                [ Recharts BarGraph Here ]
            </div>
          </div>

          {/* Brain Activations */}
          <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800 flex-grow">
            <h2 className="text-xs uppercase text-neutral-500 mb-4 font-semibold tracking-wider">TRIBEv2 Neural Activations</h2>
            <div className="h-full flex items-center justify-center border border-dashed border-neutral-700 rounded text-neutral-600 animate-pulse">
                Awaiting Parcellation Data...
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
