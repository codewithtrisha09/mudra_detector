# Mudra Detector

## Overview

Mudra Detector is a sophisticated computer vision application designed to recognize and classify Bharatanatyam hand mudras in real time. Leveraging advanced technologies including OpenCV, MediaPipe, and machine learning, the system detects hand landmarks from live webcam feeds, performs mudra classification, and presents results through an intuitive Flask-based web interface with confidence scoring.

---

## Key Features

- **Real-time Hand Landmark Detection**: Utilizes MediaPipe for accurate and rapid identification of hand keypoints from live video streams
- **Bharatanatyam Mudra Classification**: Applies trained machine learning models to classify detected hand positions into recognized mudra categories
- **Webcam Integration**: Seamless integration with standard webcam hardware for immediate real-time processing
- **Confidence Scoring**: Displays classification confidence metrics to indicate prediction reliability
- **Web-Based Interface**: User-friendly Flask application accessible through a web browser
- **Modular Architecture**: Easily extensible design to accommodate additional mudra classes and classification models

---

## Technology Stack

| Category | Technologies |
|----------|---|
| **Language** | Python 3.x |
| **Computer Vision** | OpenCV, MediaPipe |
| **Machine Learning** | scikit-learn, TensorFlow/Keras |
| **Web Framework** | Flask |
| **Data Processing** | NumPy, Pandas |
| **Frontend** | HTML, CSS, JavaScript |

---

## Project Structure

```
mudra_detector/
├── app.py                      # Flask application entry point
├── extract_landmarks.py        # Hand landmark extraction module
├── train_model.py              # Model training script
├── requirements.txt            # Project dependencies
├── hand_landmarker.task        # MediaPipe hand landmark model
├── mudra_model.keras           # Trained classification model
├── label_encoder.pkl           # Label encoding for mudra classes
├── labelencoder.pkl            # Backup label encoder
├── landmarks.csv               # Training dataset with landmark coordinates
├── static/                     # Static web assets
│   └── [CSS and JavaScript files]
└── templates/                  # HTML templates
    └── [Web interface templates]
```

---

## How It Works

The Mudra Detector operates through a well-defined pipeline:

1. **Video Capture**: The application captures continuous video frames from the connected webcam
2. **Frame Processing**: OpenCV processes each frame for analysis
3. **Landmark Detection**: MediaPipe's hand detection model identifies 21 hand keypoints in each frame
4. **Feature Extraction**: Extracted landmark coordinates are normalized and prepared as feature vectors
5. **Classification**: The trained machine learning model processes the feature vectors and predicts the mudra class
6. **Result Display**: Predictions along with confidence scores are rendered in real time on the web interface

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/codewithtrisha09/mudra_detector.git
cd mudra_detector
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

#### On Windows (PowerShell)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt)
```cmd
.venv\Scripts\activate
```

#### On macOS and Linux
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure all required packages are installed before proceeding to the next section.

---

## Usage

### Running the Application

Start the Flask development server:

```bash
python app.py
```

After initialization, the terminal will display a local URL (typically `http://127.0.0.1:5000`). Open this address in your web browser to access the application interface.

### Using the Interface

1. Grant webcam permissions when prompted by your browser
2. Position your hand within the camera's field of view
3. The system will display the detected mudra classification and associated confidence score in real time

---

## Training and Model Development

### Extract Landmarks

To generate training data from recorded hand positions:

```bash
python extract_landmarks.py
```

This script processes video or image data and outputs landmark coordinates to `landmarks.csv`.

### Train the Classification Model

To train the mudra classification model:

```bash
python train_model.py
```

The trained model will be saved as `mudra_model.keras` with corresponding label encodings.

---

## Future Development Roadmap

- **Expanded Mudra Library**: Incorporate additional Bharatanatyam mudra classes for comprehensive coverage
- **Enhanced Accuracy**: Implement advanced feature engineering and model optimization techniques
- **Posture Correction Feedback**: Add real-time guidance for users learning mudra positions
- **Augmented Reality Overlays**: Develop AR visualizations for enhanced user engagement and learning
- **Cloud Deployment**: Deploy the application to cloud platforms for broader accessibility
- **Mobile Support**: Create mobile-compatible versions for iOS and Android platforms
- **Multi-hand Detection**: Extend functionality to recognize complex two-hand mudra combinations

---

## Author

**Trisha Shetty**  
Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
MIT Manipal  
2024–2028



## Contributing

Contributions, feedback, and suggestions are welcome. Please feel free to open issues or submit pull requests to improve the project.
