import os
import json
import cv2
import math
from transformers import pipeline

def main():
    base_directory = "/Users/bobby/Documents/consi-tribe-project"
    transcript_path = os.path.join(base_directory, "data", "transcript.json")
    output_path = os.path.join(base_directory, "data", "text_emotions.json")
    clip_path = os.path.join(base_directory, "data", "clip.mp4")

    with open(transcript_path, "r") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("No segments found.")
        return 
    
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps > 0:
        total_seconds = math.ceil(total_frames / fps)

    emotions_per_second = {str(second): {} for second in range(total_seconds)}

    print("Loading Huggingface GoEmotions model...")
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    
    print("Processing transcript segments...")
    for segment in segments:
        start = math.floor(segment["start"])
        end = math.ceil(segment["end"])
        text = segment["text"].strip()

        if not text:
            continue

        model_output = classifier(text)
        segment_results = model_output[0]

        segment_emotions = {}
        for result in segment_results:
            emotion_label = result["label"]
            emotion_score = result["score"]
            segment_emotions[emotion_label] = emotion_score

        for second in range(start, min(end + 1, total_seconds)):
            current_second_emotions = emotions_per_second[str(second)]

            if len(current_second_emotions) == 0:
                emotions_per_second[str(second)] = segment_emotions.copy()
            else:
                for label, score in segment_emotions.items():
                    if label in current_second_emotions:
                        existing_score = current_second_emotions[label]
                    else:
                        existing_score = 0.0
                        
                    new_average = (existing_score + score) / 2.0
                    current_second_emotions[label] = new_average

    with open(output_path, "w") as f:
        json.dump(emotions_per_second, f, indent=4)

    print("Successfully saved.")
    
if __name__ == "__main__":
    main()