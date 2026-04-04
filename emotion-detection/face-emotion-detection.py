from ultralytics import YOLO
from hsemotion.facial_emotions import HSEmotionRecognizer
from huggingface_hub import hf_hub_download

import cv2 
import pandas as pd
import torch

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})

device = torch.device("mps") # force the use of M3 GPU

""" MODEL SETUP
 -> Using YOLOv11n with face detection dataset to move from detecting entire persons to just faces.
This should clean the emotion data drastically as we were getting a ton of false positives
using vanilla YOLO26
"""
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
face_model = YOLO(model_path)
emotion_model = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl", device="mps")


""" CLIP PREP
-> filepath
-> capture 
-> get_fps
"""
clip = "../content/clip.mp4"
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)


data_log = [] # result array
face_buffer = [] # store faces in buffer to process in batches
time_buffer = [] # store time to make the faces in batch
BATCH_SIZE = 64
FRAME_SKIP = 1 # computing each frame, we'll average the 24 frames to match the TRIBEv2's 1Hz 

""" PROCESSING 
-> while new frame 
-> run face detection model 
-> for each detection -> move boxes to cpu so opencv can handle coordinates
-> for each box in boxes -> map cordinates of face -> crop frame to face -> add face and time to buffer
-> once batch is full -> run emotion model on buffer -> add emotion to data_log
-> reset buffers 
-> increment frame index
"""
frame_index = 0
while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = cap.read()
    if not success:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0                                  # current timestamp 
    detections = face_model.predict(frame, conf=0.5, device="mps", verbose=False)           # YOLO to detect faces

    for d in detections:
        boxes = d.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                face_buffer.append(face_crop)
                time_buffer.append(current_time)

    if len(face_buffer) >= BATCH_SIZE: 
        emotions, scores = emotion_model.predict_multi_emotions(face_buffer, logits=False) # HSEmotion Inference

        for i in range(len(emotions)):
            data_log.append({
                "timestamp": time_buffer[i], 
                "emotion": emotions[i],
                "valence": scores[i][8],
                "arousal": scores[i][9],
                "anger_idx": scores[i][0],
                "comtempt_idx": scores[i][1],
                "disgust_idx": scores[i][2],
                "fear_idx": scores[i][3],
                "happy_idx": scores[i][4],
                "neutral_idx": scores[i][5],
                "sad_idx": scores[i][6],
                "surprise_idx": scores[i][7]
            })
        
        face_buffer, time_buffer = [], []

    frame_index += FRAME_SKIP

""" FLUSH LEFTOVER FACES
-> Process any leftover faces in buffer (if batch_size doesn't break cleanly)
"""
if face_buffer:
    emotions, scores = emotion_model.predict_multi_emotions(face_buffer, logits=False)
    for i in range(len(emotions)):
        data_log.append({
            "timestamp": time_buffer[i], 
            "emotion": emotions[i],
            "valence": scores[i][8],
            "arousal": scores[i][9],
            "anger_idx": scores[i][0],
            "comtempt_idx": scores[i][1],
            "disgust_idx": scores[i][2],
            "fear_idx": scores[i][3],
            "happy_idx": scores[i][4],
            "neutral_idx": scores[i][5],
            "sad_idx": scores[i][6],
            "surprise_idx": scores[i][7]
        })

cap.release()

""" AVERAGING FRAMES 
-> Fit result array into dataframe 
-> Create a second column by flooring timestamp (e.g., 1.8s become 1.0s)
-> Group by the second column and aggregate
-> Valence/Arousal: Mean (Average Intensity) & Emotion: Mode (Most frequent label seen in 1 second window)
"""
df = pd.DataFrame(data_log)
df["timestamp_1hz"] = df["timestamp"].astype(int)
numeric_cols = df.select_dtypes(include="number").columns.difference(["timestamp_1hz"])

df_means = df.groupby("timestamp_1hz")[numeric_cols].mean()
df_mode = df.groupby("timestamp_1hz")["emotion"].agg(lambda x: x.mode().iloc[0])
df_averaged = df_means.join(df_mode)

# fill every second so length matches TRIBE v2 (which has a row per second)
clip_duration_sec = int(total_frames / fps)
full_index = pd.RangeIndex(0, clip_duration_sec, name="timestamp_1hz")
df_averaged = df_averaged.reindex(full_index)
df_averaged[numeric_cols] = df_averaged[numeric_cols].fillna(0)
df_averaged["emotion"] = df_averaged["emotion"].ffill().bfill().fillna("Neutral")

df_averaged.reset_index().to_csv("emotion_results_1hz_averaged.csv", index=False)

print("Done.")

