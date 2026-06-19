from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import joblib
from tensorflow import keras
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

model = keras.models.load_model("mudra_model.keras")
le = joblib.load("label_encoder.pkl")

MODEL_PATH = Path("hand_landmarker.task")
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH.resolve()))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2
)
detector = vision.HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

latest_prediction = {"label": "---", "confidence": 0.0}

def confidence_level(conf):
    if conf >= 0.85:
        return "High"
    elif conf >= 0.60:
        return "Medium"
    return "Low"

def draw_info_box(frame, name, conf, level):
    x1, y1, x2, y2 = 15, 15, 500, 165
    cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 15, 15), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Bharatanatyam Mudra Detector", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Mudra: {name}", (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {conf * 100:.1f}%", (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Level: {level}", (25, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    bar_x1, bar_y1 = 320, 125
    bar_w, bar_h = 160, 18
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * conf)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), (0, 255, 0), -1)

def gen_frames():
    global latest_prediction
    timestamp = 0
    while True:
        success, frame = camera.read()
        if not success:
            continue

        timestamp += 1
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                features = []
                for lm in hand_landmarks:
                    features.extend([lm.x, lm.y, lm.z])

                x = np.array(features).reshape(1, -1)
                pred = model.predict(x, verbose=0)[0]
                idx = np.argmax(pred)
                name = le.inverse_transform([idx])[0]
                conf = float(pred[idx])
                level = confidence_level(conf)

                latest_prediction = {"label": name, "confidence": conf}
                draw_info_box(frame, name, conf, level)
        else:
            latest_prediction = {"label": "---", "confidence": 0.0}
            cv2.rectangle(frame, (15, 15), (500, 95), (15, 15, 15), -1)
            cv2.rectangle(frame, (15, 15), (500, 95), (0, 0, 255), 2)
            cv2.putText(frame, "Show your hand to detect mudra", (25, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Landmarks and prediction will appear here", (25, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/prediction")
def prediction():
    return jsonify(latest_prediction)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)