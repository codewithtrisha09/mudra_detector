# Mudra Detector

Mudra Detector is a computer vision project that recognizes Bharatanatyam hand mudras in real time using OpenCV, MediaPipe, and machine learning. It detects hand landmarks from a webcam feed, classifies the mudra, and displays the prediction through a Flask web interface.

## Features

- Real-time hand landmark detection.
- Bharatanatyam mudra classification.
- Webcam-based prediction.
- Confidence score display.
- Flask-based web interface.
- Easy to extend with more mudra classes.

## Tech Stack

- Python
- OpenCV
- MediaPipe
- Flask
- NumPy
- Pandas
- scikit-learn / TensorFlow
- HTML
- CSS

## Project Structure

```text
mudra_detector/
├── app.py
├── extract_landmarks.py
├── train_model.py
├── requirements.txt
├── hand_landmarker.task
├── label_encoder.pkl
├── labelencoder.pk1
├── landmarks.csv
├── mudra_model.keras
├── static/
└── templates/
```

## How It Works

1. The webcam captures live video.
2. OpenCV reads each frame.
3. MediaPipe detects hand landmarks.
4. Landmark data is passed to the trained model.
5. The model predicts the mudra.
6. The result is shown on the web page.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/codewithtrisha09/mudra_detector.git
cd mudra_detector
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### Command Prompt
```bat
.venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the app
```bash
python app.py
```

Then open the local address shown in the terminal.

## Future Improvements

- Add more mudra classes.
- Improve prediction accuracy.
- Add posture correction feedback.
- Add AR-style overlays.
- Deploy the app online.



## Author

Trisha Shetty  
CSE (AI & ML), MIT Manipal  
2024–2028