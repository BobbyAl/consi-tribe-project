import os
import librosa
import math
import json
from transformers import pipeline

def main():
    base_directory = "/Users/bobby/Documents/consi-tribe-project"
    audio_path = os.path.join(base_directory, "data", "clip_audio.wav")
    transcript_path = os.path.join(base_directory, "data", "transcript.json")
    output_path = os.path.join(base_directory, "data", "audio_features.json")

    print("Loading transcript into...")
    with open(transcript_path, "r") as f:
        transcript_data = json.load(f)

    speech_seconds = set()
    for segment in transcript_data.get("segments", []):
        start = math.floor(segment["start"])
        end = math.ceil(segment["end"])
        for second in range(start, end + 1):
            speech_seconds.add(second)

    print("Loading audio into librosa")
    y, sr = librosa.load(audio_path, sr=16000)

    total_seconds = math.ceil(len(y) / sr)

    print("Loading Huggingface audio emotion model (IEMOCAP)...")
    emotion_classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")

    audio_data_per_second = {}
    print("Extracting acoustics and speech emotions per second...")
    for second in range(total_seconds):
        start = second * sr 
        end = min((second + 1) * sr, len(y))
        y_second = y[start:end]

        if len(y_second) < 400:
            audio_data_per_second[str(second)] = {
                "audio_acoustics": {"rms_energy": 0.0, "spectral_centroid": 0.0},
                "audio_emotions": {"neu": 0.0, "hap": 0.0, "ang": 0.0, "sad": 0.0},
            }
            continue

    
        rms_value = float(librosa.feature.rms(y=y_second).mean())
        centroid_value = float(librosa.feature.spectral_centroid(y=y_second, sr=sr).mean())

        emotion_dictionary = {"neu": 0.0, "hap": 0.0, "ang": 0.0, "sad": 0.0}

        if second in speech_seconds:
            results = emotion_classifier({"sampling_rate": sr, "raw": y_second})
            emotion_dictionary = {res["label"]: res["score"] for res in results}

        audio_data_per_second[str(second)] = {
            "audio_acoustics": {
                "rms_energy": rms_value,
                "spectral_centroid": centroid_value
            }, 
            "audio_emotions": emotion_dictionary
        }

    with open(output_path, "w") as f:
        json.dump(audio_data_per_second, f, indent=4)

    print("Successfully saved audio features.")
            
if __name__ == "__main__":
    main()