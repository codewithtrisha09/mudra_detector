# 🪷 MudraLens

**Real-time Bharatanatyam mudra recognition, powered by computer vision.**

MudraLens uses hand-landmark tracking and a trained classification model to identify Bharatanatyam hand mudras live through a webcam, scoring each prediction with a confidence level — making it a practical tool for dance students, teachers, and researchers exploring the intersection of classical Indian art and AI.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Why MudraLens

Bharatanatyam expresses meaning through 28 root hand gestures (mudras), each requiring precise finger and palm positioning. MudraLens brings that precision into a browser: point a webcam at your hand and get an instant, confidence-scored classification — no dance expert required to check your form.

---

## Key Features

- **Real-Time Hand Landmark Detection** — MediaPipe tracks 21 keypoints per hand at live video speed
- **Bharatanatyam Mudra Classification** — a trained Keras model maps landmark geometry to named mudra classes
- **Confidence Scoring** — every prediction ships with a probability score, so users know how certain the model is
- **Webcam-Native Interface** — works directly in the browser via a lightweight Flask backend, no installation beyond setup
- **Modular Pipeline** — landmark extraction, training, and inference are cleanly separated, so new mudra classes can be added without rewriting the app
- **89% Classification Accuracy** — validated on a large, diverse image dataset spanning multiple mudra classes

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.x |
| **Computer Vision** | OpenCV, MediaPipe |
| **Machine Learning** | scikit-learn, TensorFlow / Keras |
| **Web Framework** | Flask |
| **Data Processing** | NumPy, Pandas |
| **Frontend** | HTML, CSS, JavaScript |

---

## How It Works

```
Webcam Feed → OpenCV Frame Capture → MediaPipe Landmark Detection
     → Feature Normalization → Keras Classifier → Mudra + Confidence Score
```

1. **Video Capture** — OpenCV pulls continuous frames from the connected webcam
2. **Landmark Detection** — MediaPipe locates 21 hand keypoints per frame
3. **Feature Extraction** — landmark coordinates are normalized into a feature vector
4. **Classification** — the trained model predicts the mudra class from that vector
5. **Result Display** — the predicted mudra and its confidence score render live in the browser

---

## Project Structure

```
mudralens/
├── app.py                   # Flask application entry point
├── extract_landmarks.py     # Hand landmark extraction from images/video
├── train_model.py           # Model training script
├── requirements.txt         # Project dependencies
├── hand_landmarker.task     # MediaPipe hand landmark model
├── mudra_model.keras        # Trained classification model
├── label_encoder.pkl        # Label encoder for mudra classes
├── landmarks.csv            # Training dataset (extracted landmark coordinates)
├── static/                  # CSS and JavaScript assets
└── templates/                # HTML templates
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/codewithtrisha09/mudralens.git
cd mudralens
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate it

| OS | Command |
|---|---|
| Windows (PowerShell) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` then `.\.venv\Scripts\Activate.ps1` |
| Windows (CMD) | `.venv\Scripts\activate` |
| macOS / Linux | `source .venv/bin/activate` |

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
python app.py
```

Open the local URL printed in the terminal (typically `http://127.0.0.1:5000`), grant webcam access when prompted, and hold a mudra up to the camera to see it classified live.

---

## Training Your Own Model

**Extract landmarks from your dataset:**
```bash
python extract_landmarks.py
```
Outputs normalized hand-landmark coordinates to `landmarks.csv`.

**Train the classifier:**
```bash
python train_model.py
```
Saves the trained model to `mudra_model.keras` along with its label encoder.

---

## Roadmap

- [ ] Expand the mudra library to cover the full set of 28 root gestures
- [ ] Posture-correction feedback for learners
- [ ] AR overlays showing correct hand positioning
- [ ] Two-hand (samyukta) mudra recognition
- [ ] Cloud deployment for browser-only access, no local setup
- [ ] Mobile-compatible build (iOS/Android)

---

## Author

**Trisha Shetty**
B.Tech CSE (AI & ML), MIT Manipal · 2024–2028
[github.com/codewithtrisha09](https://github.com/codewithtrisha09)

---

## Contributing

Issues and pull requests are welcome — especially new labeled mudra data, model improvements, or UI polish. Please open an issue first for any significant feature so we can align on scope before you start coding.


