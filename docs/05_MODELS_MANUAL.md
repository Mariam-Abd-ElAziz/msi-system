# User Manual: Making Predictions with MSI Models

## Overview

This manual explains how to use the trained MSI models to make predictions on new material images. After training is complete (Phase 3), you have two ways to use the models:

1. **Interactive Prediction** - Predict single images
2. **Batch Processing** - Predict multiple images
3. **Ensemble Voting** - Combine predictions from multiple models

---

## Quick Start

### Basic Single Image Prediction

```python
import numpy as np
import cv2
from src.models.unified_predictor import UnifiedPredictor
from src.preprocessing.feature_extractor import extract_combined_features

# Load trained models
predictor = UnifiedPredictor(model_dir="saved_models")

# Load and process image
image = cv2.imread("path/to/image.jpg")
image = cv2.resize(image, (128, 128))

# Extract features
features = extract_combined_features(image)

# Get prediction
prediction = predictor.predict_single(features, model_name='svm')
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
print(f"Predicted class: {class_names[prediction]}")
```

---

## Installation & Setup

### Prerequisites

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Or install individually
pip install opencv-python numpy scikit-learn scikit-image matplotlib pillow
```

### Import Required Modules

```python
import numpy as np
import cv2
from pathlib import Path
from src.models.unified_predictor import UnifiedPredictor
from src.preprocessing.feature_extractor import (
    extract_combined_features,
    extract_hog_features,
    extract_color_histogram_features,
    extract_lbp_features,
    preprocess_image
)
```

---

## Loading Trained Models

### Method 1: Automatic Loading

```python
from src.models.unified_predictor import UnifiedPredictor

# Automatically loads both SVM and KNN from default location
predictor = UnifiedPredictor()

# Check loaded models
print(f"Loaded models: {list(predictor.models.keys())}")
# Output: Loaded models: ['svm', 'knn']
```

### Method 2: Custom Model Directory

```python
# If models are in different location
predictor = UnifiedPredictor(model_dir="path/to/models")
```

### Verify Models Loaded

```python
# Check if models are available
if 'svm' in predictor.models:
    print("✓ SVM model loaded")
else:
    print("✗ SVM model NOT found")

# Get model configuration
config = predictor.get_model_configs()
print(config)
```

---

## Making Single Predictions

### Step 1: Load Image

```python
import cv2

# Read image from file
image_path = "glass_bottle.jpg"
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    print(f"Image shape: {image.shape}")  # (height, width, 3)
```

### Step 2: Preprocess Image

```python
# Resize to standard size (128×128)
image = cv2.resize(image, (128, 128))
print(f"Resized: {image.shape}")  # (128, 128, 3)
```

### Step 3: Extract Features

```python
from src.preprocessing.feature_extractor import extract_combined_features

# Extract all features (HOG + Color + LBP)
features = extract_combined_features(image)
print(f"Feature vector shape: {features.shape}")  # (8336,)

# Ensure 2D shape for prediction
features = features.reshape(1, -1)  # (1, 8336)
```

### Step 4: Make Prediction

```python
# Get prediction from SVM
prediction_svm = predictor.predict_single(features, model_name='svm')
print(f"SVM prediction: {prediction_svm}")  # 0-6 (class index)

# Get prediction from KNN
prediction_knn = predictor.predict_single(features, model_name='knn')
print(f"KNN prediction: {prediction_knn}")

# Map to class name
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
print(f"Predicted material: {class_names[prediction_svm]}")
```

### Step 5: Get Confidence Scores (SVM)

```python
# SVM provides probability estimates
probabilities_svm = predictor.models['svm'].predict_proba(features)
print(f"Probabilities shape: {probabilities_svm.shape}")  # (1, 7)

# Get confidence for predicted class
prediction = predictor.predict_single(features, model_name='svm')
confidence = probabilities_svm[0][prediction]
print(f"Confidence: {confidence:.2%}")  # e.g., 92.34%

# Show all class probabilities
for class_name, prob in zip(class_names, probabilities_svm[0]):
    print(f"  {class_name:12} : {prob:>6.2%}")
```

### Complete Example

```python
import cv2
from src.models.unified_predictor import UnifiedPredictor
from src.preprocessing.feature_extractor import extract_combined_features

# Load models
predictor = UnifiedPredictor()

# Load and process image
image = cv2.imread("glass_bottle.jpg")
image = cv2.resize(image, (128, 128))

# Extract features
features = extract_combined_features(image)
features = features.reshape(1, -1)

# Make prediction
pred_class = predictor.predict_single(features, model_name='svm')

# Get confidence
probabilities = predictor.models['svm'].predict_proba(features)[0]
confidence = probabilities[pred_class]

# Display results
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
print(f"Material: {class_names[pred_class]}")
print(f"Confidence: {confidence:.1%}")
```

---

## Batch Predictions (Multiple Images)

### Processing Image Directory

```python
import cv2
import numpy as np
from src.models.unified_predictor import UnifiedPredictor
from src.preprocessing.feature_extractor import extract_combined_features
from pathlib import Path

# Load models
predictor = UnifiedPredictor()
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']

# Get all images from folder
image_folder = Path("data/test_images")
image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))

# Process each image
results = []
for image_path in image_files:
    # Load and preprocess
    image = cv2.imread(str(image_path))
    if image is None:
        continue
    
    image = cv2.resize(image, (128, 128))
    
    # Extract features
    features = extract_combined_features(image)
    features = features.reshape(1, -1)
    
    # Predict
    pred = predictor.predict_single(features, model_name='svm')
    
    # Get confidence
    probs = predictor.models['svm'].predict_proba(features)[0]
    confidence = probs[pred]
    
    # Store result
    results.append({
        'image': image_path.name,
        'material': class_names[pred],
        'confidence': confidence
    })
    
    print(f"{image_path.name}: {class_names[pred]} ({confidence:.1%})")

# Print summary
print(f"\nProcessed {len(results)} images")
```

### Optimized Batch Processing

For better performance with many images:

```python
import cv2
import numpy as np
from pathlib import Path
from src.models.unified_predictor import UnifiedPredictor
from src.preprocessing.feature_extractor import extract_combined_features

# Load models once
predictor = UnifiedPredictor()
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']

# Get all images
image_folder = Path("data/test_images")
image_files = sorted(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")))

# Extract all features first
all_features = []
valid_files = []

for image_path in image_files:
    image = cv2.imread(str(image_path))
    if image is None:
        continue
    
    image = cv2.resize(image, (128, 128))
    features = extract_combined_features(image)
    all_features.append(features)
    valid_files.append(image_path.name)

# Batch predict (faster than one-by-one)
if all_features:
    features_array = np.array(all_features)  # Shape: (n_images, 8336)
    predictions = predictor.batch_predict(features_array, model_name='svm')
    
    # Get confidences
    probabilities = predictor.models['svm'].predict_proba(features_array)
    confidences = probabilities[np.arange(len(predictions)), predictions]
    
    # Print results
    for filename, pred, conf in zip(valid_files, predictions, confidences):
        print(f"{filename}: {class_names[pred]} ({conf:.1%})")
```

---

## Ensemble Predictions

### Voting Predictions

Combine SVM and KNN for potentially better accuracy:

```python
# Single image
features = features.reshape(1, -1)

# Ensemble voting
result = predictor.predict_ensemble(features, method='voting')

print(f"SVM prediction: {result['individual_predictions']['svm']}")
print(f"KNN prediction: {result['individual_predictions']['knn']}")
print(f"Ensemble prediction: {result['ensemble_prediction']}")

class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
print(f"Predicted material: {class_names[result['ensemble_prediction']]}")
```

### How Voting Works

```
Sample Feature Vector
        │
        ├──→ SVM Model ──→ Prediction: 0 (glass)
        │
        └──→ KNN Model ──→ Prediction: 0 (glass)

Votes: glass=2, paper=0, ...
Winner: glass (unanimous)

Final Prediction: 0 (glass)
Confidence: VERY HIGH (both agree)
```

### When to Use Ensemble

```
Case 1 - Both Agree:
  SVM: glass, KNN: glass
  Result: glass (CONFIDENT)
  Use: YES

Case 2 - Disagree:
  SVM: glass, KNN: paper
  Result: glass or paper (50-50 tie)
  Use: MAYBE (check confidence)

Case 3 - Confidence Check:
  SVM: glass (92%), KNN: glass (78%)
  Result: glass (both confident)
  Use: YES (ensemble better)
```

---

## Prediction Confidence & Thresholding

### Understanding Confidence Scores

```python
# Get all probabilities
probabilities = predictor.models['svm'].predict_proba(features)[0]

# Default prediction
default_pred = np.argmax(probabilities)
default_conf = probabilities[default_pred]

print(f"Default prediction: {class_names[default_pred]}")
print(f"Confidence: {default_conf:.2%}")

# Full breakdown
for class_name, prob in zip(class_names, probabilities):
    bar = "█" * int(prob * 50)
    print(f"{class_name:12} {bar} {prob:.1%}")
```

### Confidence Thresholding

Reject predictions below confidence threshold:

```python
# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.75  # 75%

# Make prediction
pred_class = predictor.predict_single(features, model_name='svm')
probabilities = predictor.models['svm'].predict_proba(features)[0]
confidence = probabilities[pred_class]

# Check threshold
if confidence >= CONFIDENCE_THRESHOLD:
    result = class_names[pred_class]
    status = "CONFIDENT"
else:
    result = "unknown"  # Reject prediction
    status = "UNCERTAIN - Rejected"

print(f"Result: {result} ({status})")
print(f"Confidence: {confidence:.1%}")
```

### When to Use Thresholding

```
Use Case 1: Quality Control
  - Reject items with <90% confidence
  - Manually review uncertain cases
  - Threshold: 0.90

Use Case 2: Recycling Automation
  - Confidence <70%? Sort to "unknown"
  - Confidence ≥70%? Automatic sorting
  - Threshold: 0.70

Use Case 3: High Safety Requirements
  - Only accept >95% confidence
  - Everything else manual inspection
  - Threshold: 0.95
```

---

## Working with Images from Different Sources

### Handling Various Image Formats

```python
import cv2
from pathlib import Path

# Supported formats
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

image_path = "path/to/image.jpg"
image = cv2.imread(image_path)

# Handle read errors
if image is None:
    print(f"Error: Cannot read {image_path}")
    # Try alternative approach
    from PIL import Image
    pil_img = Image.open(image_path)
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Check image format
print(f"Image type: {image.dtype}")
print(f"Image shape: {image.shape}")
print(f"Image range: [{image.min()}, {image.max()}]")
```

### Handling Different Image Sizes

```python
import cv2

# All images are resized to (128, 128) automatically
image = cv2.imread("small_image.jpg")  # Might be 64×64
image = cv2.resize(image, (128, 128))  # Always 128×128

# For very small images (< 64×64), consider interpolation
if image.shape[0] < 64 or image.shape[1] < 64:
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
else:
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
```

### Handling Video Frames

```python
import cv2

# Open video
cap = cv2.VideoCapture("material_stream.mp4")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process every Nth frame (e.g., every 5th for speed)
    if frame_count % 5 != 0:
        continue
    
    # Make prediction
    frame_resized = cv2.resize(frame, (128, 128))
    features = extract_combined_features(frame_resized)
    features = features.reshape(1, -1)
    
    pred = predictor.predict_single(features, model_name='svm')
    print(f"Frame {frame_count}: {class_names[pred]}")

cap.release()
```

---

## Error Handling

### Common Errors & Solutions

#### Error 1: Model Files Not Found

```python
try:
    predictor = UnifiedPredictor()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Solution: Run main_train.py first to train models")
```

#### Error 2: Image Load Failure

```python
image_path = "path/to/image.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Cannot read image from {image_path}")
    print("Check:")
    print(f"  - File exists: {Path(image_path).exists()}")
    print(f"  - Path is correct")
    print(f"  - File is valid image format")
```

#### Error 3: Wrong Feature Dimensions

```python
# If extracted features have wrong shape
features = extract_combined_features(image)
print(f"Feature shape: {features.shape}")  # Should be (8336,)

if len(features) != 8336:
    print("Error: Feature dimension mismatch!")
    print(f"Expected 8336, got {len(features)}")
```

### Debugging Predictions

```python
import cv2
import numpy as np
from src.models.unified_predictor import UnifiedPredictor

# Enable detailed output
def predict_with_debug(image_path, predictor, class_names):
    print(f"\n{'='*50}")
    print(f"DEBUG: Predicting {image_path}")
    print(f"{'='*50}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Failed to load image")
        return
    print(f"✓ Loaded image: {image.shape}")
    
    # Preprocess
    image = cv2.resize(image, (128, 128))
    print(f"✓ Resized image: {image.shape}")
    
    # Extract features
    from src.preprocessing.feature_extractor import extract_combined_features
    features = extract_combined_features(image)
    print(f"✓ Extracted features: {features.shape}")
    
    features = features.reshape(1, -1)
    print(f"✓ Reshaped for prediction: {features.shape}")
    
    # Predict
    try:
        pred = predictor.predict_single(features, model_name='svm')
        probs = predictor.models['svm'].predict_proba(features)[0]
        
        print(f"\n✓ Predictions:")
        print(f"   Top prediction: {class_names[pred]} ({probs[pred]:.2%})")
        
        # Show top 3
        top_3_idx = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(top_3_idx, 1):
            print(f"   {rank}. {class_names[idx]:12} {probs[idx]:>6.2%}")
    
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()

# Use it
predictor = UnifiedPredictor()
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
predict_with_debug("test_image.jpg", predictor, class_names)
```

---

## Production Deployment

### Creating a Simple API

```python
from flask import Flask, request, jsonify
from src.models.unified_predictor import UnifiedPredictor
from src.preprocessing.feature_extractor import extract_combined_features
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
predictor = UnifiedPredictor()
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict material from image
    
    Request: POST /predict
    Data: {"image": base64_encoded_image}
    Response: {"material": "glass", "confidence": 0.92}
    """
    try:
        # Get image from request
        image_data = request.json['image']
        image = Image.open(BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess
        image = cv2.resize(image, (128, 128))
        
        # Extract features
        features = extract_combined_features(image)
        features = features.reshape(1, -1)
        
        # Predict
        pred = predictor.predict_single(features, model_name='svm')
        prob = predictor.models['svm'].predict_proba(features)[0][pred]
        
        return jsonify({
            'material': class_names[pred],
            'confidence': float(prob),
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

---

## Performance Tips

### Optimize for Speed

```python
# Batch processing is faster than one-by-one
# Process in batches of 32-128 images

import numpy as np
from pathlib import Path

image_files = sorted(Path("images/").glob("*.jpg"))
batch_size = 32

for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i+batch_size]
    
    # Extract features for batch
    batch_features = []
    for path in batch_files:
        image = cv2.imread(str(path))
        image = cv2.resize(image, (128, 128))
        features = extract_combined_features(image)
        batch_features.append(features)
    
    # Predict for entire batch at once
    features_array = np.array(batch_features)
    predictions = predictor.batch_predict(features_array, model_name='svm')
    
    print(f"Processed batch {i//batch_size + 1}")
```

### Memory Efficiency

```python
# Don't load all images at once for very large datasets
from pathlib import Path
from src.models.unified_predictor import UnifiedPredictor

predictor = UnifiedPredictor()  # Load once

for image_path in Path("images/").glob("*.jpg"):
    # Load one at a time, predict, save result, discard
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (128, 128))
    features = extract_combined_features(image).reshape(1, -1)
    
    pred = predictor.predict_single(features, model_name='svm')
    
    # Save result to file instead of keeping in memory
    with open("results.txt", "a") as f:
        f.write(f"{image_path.name},{class_names[pred]}\n")
```

---

## Summary

| Task | Function | Example |
|------|----------|---------|
| Load models | `UnifiedPredictor()` | `predictor = UnifiedPredictor()` |
| Single prediction | `predict_single()` | `pred = predictor.predict_single(features, 'svm')` |
| Batch prediction | `batch_predict()` | `preds = predictor.batch_predict(features_array)` |
| Get confidence | `predict_proba()` | `conf = model.predict_proba(features)[0]` |
| Ensemble vote | `predict_ensemble()` | `result = predictor.predict_ensemble(features)` |

---

**Previous**: [Phase 3: Model Training](04_PHASE_3_TRAINING.md)  
**Next**: [API Reference](06_API_REFERENCE.md)
