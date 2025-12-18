# Deployment Module Overview

This document provides an overview of the `deployment` folder in the `msi-system` project. It explains the purpose of each file and its relationship to other modules in the system.

---

## Folder Structure



The `deployment` folder contains the following files and subdirectories:

```
deployment/
├── app.py
├── requirements.txt
├── camera/
│   ├── camera.py
│   └── __pycache__/
│       ├── camera.cpython-311.pyc
│       └── camera.cpython-312.pyc
├── feature_extraction/
│   ├── extractor.py
│   └── __pycache__/
│       └── extractor.cpython-311.pyc
└── inference/
    ├── predictor.py
    └── __pycache__/
        └── predictor.cpython-311.pyc
```

---

## File Descriptions

### 1. `app.py`

**Purpose**:  
The `app.py` file serves as the entry point for the deployment system. It orchestrates the camera feed, feature extraction, and inference processes to perform real-time classification.

**Key Responsibilities**:
- Initializes the `Camera`, `FeatureExtractor`, and `Predictor` modules.
- Captures frames from the camera in real-time.
- Extracts features from the captured frames using the `FeatureExtractor`.
- Uses the `Predictor` to classify the extracted features.
- Displays the classification results, confidence scores, and FPS on the live video feed.

**Relationships**:
- **`camera/camera.py`**: Provides the `Camera` class for capturing video frames.
- **`feature_extraction/extractor.py`**: Provides the `FeatureExtractor` class for extracting features from frames.
- **`inference/predictor.py`**: Provides the `Predictor` class for making predictions based on extracted features.

---

### 2. `requirements.txt`

**Purpose**:  
Lists the Python dependencies required to run the deployment system.

**Key Responsibilities**:
- Ensures all necessary libraries (e.g., `opencv-python`, `numpy`) are installed in the environment.

**Relationships**:
- Used during the setup process to install dependencies for all modules in the `deployment` folder.

---

### 3. `camera/camera.py`

**Purpose**:  
Handles camera initialization and frame capture.

**Key Responsibilities**:
- Connects to the camera device and sets the resolution.
- Captures frames in real-time.
- Provides utility functions to test and find optimal camera resolutions.

**Relationships**:
- **`app.py`**: The `Camera` class is instantiated in `app.py` to provide real-time video frames for processing.

---

### 4. `feature_extraction/extractor.py`

**Purpose**:  
Extracts meaningful features from video frames for classification.

**Key Responsibilities**:
- Processes raw frames to extract numerical features.
- Prepares the data for the inference module.

**Relationships**:
- **`app.py`**: The `FeatureExtractor` class is used in `app.py` to process frames before passing them to the `Predictor`.

---

### 5. `inference/predictor.py`

**Purpose**:  
Performs inference using pre-trained models to classify the extracted features.

**Key Responsibilities**:
- Loads pre-trained models from the `saved_models` directory.
- Predicts the class label and confidence score for the given features.
- Maps the predicted class ID to a human-readable label.

**Relationships**:
- **`app.py`**: The `Predictor` class is used in `app.py` to classify the features extracted from video frames.
- **`saved_models/`**: Loads models from the `saved_models` directory in the project root.

---

## Workflow Overview

1. **Camera Initialization**:
   - The `Camera` module initializes the camera and starts capturing frames.

2. **Feature Extraction**:
   - The `FeatureExtractor` processes each frame to extract numerical features.

3. **Inference**:
   - The `Predictor` uses the extracted features to predict the class label and confidence score.

4. **Visualization**:
   - The results (class label, confidence, and FPS) are displayed on the live video feed.

---

## Relationships to Other Modules

- **`src/models/unified_predictor.py`**:
  - The `Predictor` module depends on the `UnifiedPredictor` class from the `src` directory to load and use pre-trained models.

- **`saved_models/`**:
  - The `Predictor` module loads models from the `saved_models` directory.

- **`docs/`**:
  - The `docs` folder provides detailed documentation for the models, APIs, and training processes referenced in the `deployment` folder.

---

## Notes

- Ensure that the `saved_models` directory contains the required pre-trained models before running the system.
- The `requirements.txt` file must be used to install all dependencies in the environment.
- The `__pycache__` directories contain compiled Python files and can be ignored.

---

This concludes the overview of the `deployment` folder. For more details, refer to the specific `.md` files in the `docs` folder.