## 🪷 MudraLens

**Real-time Bharatanatyam mudra recognition powered by computer vision.**

MudraLens uses hand-landmark tracking and a trained classification model to identify Bharatanatyam hand mudras live through a webcam, scoring each prediction with a confidence level. It is designed as a practical tool for dance students, teachers, and researchers exploring the intersection of classical Indian art and artificial intelligence.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Bharatanatyam expresses meaning through 28 root hand gestures (mudras), each requiring precise finger and palm positioning. MudraLens brings that precision into the browser: point a webcam at a hand and receive an instant, confidence-scored classification, enabling practitioners to verify their form without requiring an expert in the room.

---

## Key Features

- **Real-Time Hand Landmark Detection** — MediaPipe tracks 21 keypoints per hand at live video speed
- **Bharatanatyam Mudra Classification** — a trained Keras model maps landmark geometry to named mudra classes
- **Confidence Scoring** — every prediction includes a probability score indicating model certainty
- **Webcam-Native Interface** — runs directly in the browser via a lightweight Flask backend
- **Modular Pipeline** — landmark extraction, training, and inference are cleanly separated, allowing new mudra classes to be added without modifying the core application
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

1. **Video Capture** — OpenCV captures continuous frames from the connected webcam
2. **Landmark Detection** — MediaPipe locates 21 hand keypoints per frame
3. **Feature Extraction** — landmark coordinates are normalized into a feature vector
4. **Classification** — the trained model predicts the mudra class from that vector
5. **Result Display** — the predicted mudra and its confidence score are rendered live in the browser

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

### Prerequisites
- Python 3.x
- A webcam-enabled device

### 1. Clone the repository
```bash
git clone https://github.com/codewithtrisha09/mudralens.git
cd mudralens
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the environment

| OS | Command |
|---|---|
| Windows (PowerShell) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` then `.\.venv\Scripts\Activate.ps1` |
| Windows (CMD) | `.venv\Scripts\activate` |
| macOS / Linux | `source .venv/bin/activate` |

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the application
```bash
python app.py
```

Open the local URL printed in the terminal (typically `http://127.0.0.1:5000`), grant webcam access when prompted, and present a mudra to the camera to see it classified in real time.

---

## Training a Custom Model

**Extract landmarks from a dataset:**
```bash
python extract_landmarks.py
```
This outputs normalized hand-landmark coordinates to `landmarks.csv`.

**Train the classifier:**
```bash
python train_model.py
```
This saves the trained model to `mudra_model.keras` along with its corresponding label encoder.

---

## Roadmap

- [ ] Expand the mudra library to cover the complete set of 28 root gestures
- [ ] Add posture-correction feedback for learners
- [ ] Introduce AR overlays to guide correct hand positioning
- [ ] Support two-hand (samyukta) mudra recognition
- [ ] Enable cloud deployment for browser-only access, eliminating local setup
- [ ] Deliver a mobile-compatible build (iOS/Android)

---

## Author

**Trisha Shetty**
B.Tech CSE (AI & ML), MIT Manipal · 2024–2028
[github.com/codewithtrisha09](https://github.com/codewithtrisha09)

---

## Contributing

Contributions are welcome, particularly labeled mudra data, model improvements, and UI enhancements. For significant feature proposals, please open an issue first to align on scope before development begins.
