from ultralytics import YOLO
from hsemotion.facial_emotions import HSEmotionRecognizer
import cv2 
import pandas as pd
import torch
import torch.serialization
import timm.models.efficientnet
torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

# pip install ultralytics
# pip install hsemotion
# pip install opencv-python pandas matplotlib
# running this locally (M3 Pro) using miniforge 
# -> pip install torch torchvision torchaudio

# force the use of M3 GPU
device = torch.device("mps")

# model setup 
face_model = YOLO("yolo26n.pt")
emotion_model = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl", device="mps")

clip = "content/clip.mp4"

# loading video up
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS)


data_log = [] # result array
face_buffer = []
time_buffer = []
BATCH_SIZE = 32
FRAME_SKIP = 12 # Sampling 2 Hz -> Closely matches TRIBEv2 (1 Hz). Should reduce runtime and give cleaner pearson correlation test results

frame_index = 0
while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = cap.read()
    if not success:
        break

    # calculate current timestamp 
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # YOLO to detect faces
    detections = face_model.predict(frame, conf=0.5, device="mps", verbose=False)

    for d in detections:
        # move boxes to cpu so opencv can handle coordinates
        boxes = d.boxes.xyxy.cpu().numpy()

        for box in boxes:
            # extract coordinates
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                face_buffer.append(face_crop)
                time_buffer.append(current_time)

    # process batch once it's full
    if len(face_buffer) >= BATCH_SIZE:
        # HSEmotion Inference
        emotions, scores = emotion_model.predict_multi_emotions(face_buffer, logits=False)

        for i in range(len(emotions)):
            data_log.append({
                "timestamp": time_buffer[i], 
                "emotion": emotions[i],
                "valence": scores[i][8],
                "arousal": scores[i][9]
            })
        # reset buffers
        face_buffer, time_buffer = [], []

    frame_index += FRAME_SKIP

# process any leftover faces in buffer (if batch_size doesn't break cleanly)
if face_buffer:
    emotions, scores = emotion_model.predict_multi_emotions(face_buffer, logits=False)
    for i in range(len(emotions)):
        data_log.append({
            "timestamp": time_buffer[i], 
            "emotion": emotions[i],
            "valence": scores[i][8],
            "arousal": scores[i][9]
        })

cap.release()

pd.DataFrame(data_log).to_csv("emotion_results.csv", index=False)
print("Done.")

