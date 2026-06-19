import os
import cv2
import pandas as pd
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

TRAIN_DIR = r"D:\images\train"
VAL_DIR = r"D:\images\val"
OUT_CSV = "landmarks.csv"
MODEL_PATH = Path("hand_landmarker.task")

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH.resolve()))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2
)
detector = vision.HandLandmarker.create_from_options(options)

rows = []

def label_from_filename(fname):
    return fname.split("_")[0]

def process_folder(folder):
    if not os.path.exists(folder):
        return
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for i, img_name in enumerate(files, start=1):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
            rows.append(coords + [label_from_filename(img_name)])
        if i % 1000 == 0:
            print(f"{folder}: processed {i}/{len(files)}")

process_folder(TRAIN_DIR)
process_folder(VAL_DIR)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUT_CSV}")