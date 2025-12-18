# API Reference

Comprehensive documentation for all MSI System functions and classes.

---

## Table of Contents

1. [Feature Extraction](#feature-extraction)
2. [Model Training](#model-training)
3. [Unified Predictor](#unified-predictor)
4. [Analysis Tools](#analysis-tools)
5. [Data Structures](#data-structures)

---

## Feature Extraction

### `extract_hog_features(image)`

Extracts Histogram of Oriented Gradients (HOG) features.

**Parameters:**
- `image` (ndarray): Input image in BGR format (from cv2.imread)

**Returns:**
- `features` (ndarray): HOG feature vector of shape (8181,)

**Example:**
```python
import cv2
from src.preprocessing.feature_extractor import extract_hog_features

image = cv2.imread("glass.jpg")
hog = extract_hog_features(image)
print(hog.shape)  # (8181,)
```

**Technical Details:**
- Uses 9 orientations
- 8×8 pixel cells
- 2×2 cell blocks
- L2-Hys normalization

---

### `extract_color_histogram_features(image)`

Extracts color histogram features from RGB channels.

**Parameters:**
- `image` (ndarray): Input image in BGR format

**Returns:**
- `features` (ndarray): Color histogram vector of shape (96,)

**Example:**
```python
from src.preprocessing.feature_extractor import extract_color_histogram_features

image = cv2.imread("cardboard.jpg")
color_hist = extract_color_histogram_features(image)
print(color_hist.shape)  # (96,)
print(color_hist.sum())  # ~1.0 (normalized)
```

**Technical Details:**
- 32 bins per RGB channel (32 × 3 = 96)
- Normalized to sum to 1
- Captures color distribution

---

### `extract_lbp_features(image)`

Extracts Local Binary Patterns (LBP) features.

**Parameters:**
- `image` (ndarray): Input image in BGR format

**Returns:**
- `features` (ndarray): LBP histogram vector of shape (59,)

**Example:**
```python
from src.preprocessing.feature_extractor import extract_lbp_features

image = cv2.imread("paper.jpg")
lbp = extract_lbp_features(image)
print(lbp.shape)  # (59,)
print(lbp.sum())  # ~1.0 (normalized)
```

**Technical Details:**
- 8 points, radius 1
- Uniform patterns only
- 59 unique uniform patterns
- Normalized histogram

---

### `extract_combined_features(image)`

Extracts and concatenates all features (HOG + Color + LBP).

**Parameters:**
- `image` (ndarray): Input image in BGR format, shape (H, W, 3)

**Returns:**
- `features` (ndarray): Combined feature vector of shape (8336,)

**Example:**
```python
from src.preprocessing.feature_extractor import extract_combined_features

image = cv2.imread("material.jpg")
features = extract_combined_features(image)
print(features.shape)  # (8336,)

# Decompose features
hog_features = features[:8181]
color_features = features[8181:8277]
lbp_features = features[8277:]
```

**Feature Breakdown:**
```
Combined Vector = [HOG | Color Histogram | LBP]
Index Range:     [0-8180 | 8181-8276 | 8277-8335]
Dimensions:      [8181 | 96 | 59]
```

---

### `preprocess_image(image_path)`

Loads and preprocesses an image file.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- `image` (ndarray): Resized image (128, 128, 3) or None if error

**Example:**
```python
from src.preprocessing.feature_extractor import preprocess_image

image = preprocess_image("path/to/image.jpg")
if image is not None:
    print(image.shape)  # (128, 128, 3)
else:
    print("Failed to load image")
```

**Preprocessing Steps:**
1. Read image from file
2. Resize to (128, 128)
3. Return or None on failure

---

### `extract_features_from_dataset(data_dir)`

Batch extracts features from all images in a directory.

**Parameters:**
- `data_dir` (str): Path to folder with class subfolders

**Returns:**
- `features` (ndarray): Shape (n_samples, 8336)
- `labels` (ndarray): Shape (n_samples,)
- `image_paths_list` (list): Original image paths

**Example:**
```python
from src.preprocessing.feature_extractor import extract_features_from_dataset

X, y, paths = extract_features_from_dataset("data/augmented")
print(f"Features: {X.shape}")
print(f"Labels: {y.shape}")
print(f"Classes: {set(y)}")
```

**Expected Directory Structure:**
```
data/augmented/
├── glass/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── paper/
├── cardboard/
├── plastic/
├── metal/
├── trash/
└── unknown/
```

---

## Model Training

### `SVMTrainer` Class

Support Vector Machine trainer with hyperparameter tuning.

**Location:** `src/models/svm_training.py`

#### Methods

##### `__init__(data_dir, model_dir)`

Initialize trainer.

```python
from src.models.svm_training import SVMTrainer

trainer = SVMTrainer(
    data_dir="data/features",
    model_dir="saved_models"
)
```

##### `train(quick_mode=False)`

Complete training pipeline.

```python
trainer.train(quick_mode=False)
```

**Parameters:**
- `quick_mode` (bool): If True, reduces parameter grid for faster tuning

**Outputs:**
- Saves `svm_model.pkl`
- Saves `svm_config.json`
- Generates analysis plots

##### `tune_hyperparameters(X_train, y_train)`

Hyperparameter optimization with GridSearchCV.

```python
import numpy as np

X_train = np.load("data/features/X_train.npy")
y_train = np.load("data/features/y_train.npy")

best_params = trainer.tune_hyperparameters(X_train, y_train)
print(best_params)
# {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
```

---

### `KNNTrainer` Class

K-Nearest Neighbors trainer.

**Location:** `src/models/knn_training.py`

#### Methods

##### `__init__(data_dir, model_dir)`

Initialize trainer.

```python
from src.models.knn_training import KNNTrainer

trainer = KNNTrainer(
    data_dir="data/features",
    model_dir="saved_models"
)
```

##### `train()`

Complete training pipeline.

```python
trainer.train()
```

**Outputs:**
- Saves `knn_model.pkl`
- Saves `knn_config.json`
- Prints accuracy metrics

##### `train_with_grid_search(X_train, y_train)`

Hyperparameter tuning.

```python
model, best_params = trainer.train_with_grid_search(X_train, y_train)
print(best_params)
# {'n_neighbors': 7, 'weights': 'distance', 'metric': 'euclidean'}
```

---

## Unified Predictor

### `UnifiedPredictor` Class

Unified interface for SVM and KNN predictions.

**Location:** `src/models/unified_predictor.py`

#### Initialization

```python
from src.models.unified_predictor import UnifiedPredictor

# Default model directory
predictor = UnifiedPredictor()

# Custom directory
predictor = UnifiedPredictor(model_dir="path/to/models")
```

#### Methods

##### `predict_single(features, model_name='svm')`

Predict class for single sample.

**Parameters:**
- `features` (ndarray): Feature vector, shape (8336,) or (1, 8336)
- `model_name` (str): 'svm' or 'knn'

**Returns:**
- `prediction` (int): Predicted class index (0-6)

**Example:**
```python
features = extract_combined_features(image)
pred = predictor.predict_single(features, model_name='svm')
print(pred)  # 0 (glass)
```

---

##### `predict_probability(features, model_name='svm')`

Get prediction probabilities.

**Parameters:**
- `features` (ndarray): Feature vector, shape (8336,) or (1, 8336)
- `model_name` (str): 'svm' or 'knn'

**Returns:**
- `probabilities` (ndarray): Probability for each class, shape (7,)

**Example:**
```python
features = extract_combined_features(image)
probs = predictor.predict_probability(features, model_name='svm')
print(probs)  # [0.05, 0.82, 0.03, 0.05, 0.02, 0.02, 0.01]
print(probs.sum())  # 1.0
```

---

##### `predict_ensemble(features, method='voting')`

Ensemble prediction combining multiple models.

**Parameters:**
- `features` (ndarray): Feature vector
- `method` (str): 'voting' or 'average'

**Returns:**
- `result` (dict): Dictionary with keys:
  - `ensemble_prediction`: Final class prediction
  - `ensemble_method`: Method used
  - `individual_predictions`: Dict of each model's prediction

**Example:**
```python
result = predictor.predict_ensemble(features, method='voting')
print(result)
# {
#   'ensemble_prediction': 0,
#   'ensemble_method': 'voting',
#   'individual_predictions': {
#       'svm': 0,
#       'knn': 0
#   }
# }
```

**How It Works:**
```python
# Voting: Majority vote
votes = [svm_pred, knn_pred]
ensemble_pred = most_common(votes)

# Example:
# SVM: 0 (glass), KNN: 0 (glass) → Result: 0
# SVM: 0 (glass), KNN: 2 (cardboard) → Result: 0 (tie-break by order)
```

---

##### `batch_predict(features_array, model_name='svm')`

Predict on multiple samples efficiently.

**Parameters:**
- `features_array` (ndarray): 2D array of features, shape (n_samples, 8336)
- `model_name` (str): 'svm' or 'knn'

**Returns:**
- `predictions` (ndarray): Predicted classes, shape (n_samples,)

**Example:**
```python
# Process 100 images
features_array = np.array([...])  # Shape: (100, 8336)
predictions = predictor.batch_predict(features_array, model_name='svm')
print(predictions.shape)  # (100,)
```

---

##### `evaluate_on_test_set(X_test, y_test)`

Evaluate models on test set.

**Parameters:**
- `X_test` (ndarray): Test features, shape (n_samples, 8336)
- `y_test` (ndarray): Test labels, shape (n_samples,)

**Returns:**
- `results` (dict): Dictionary with accuracy and predictions for each model

**Example:**
```python
X_test = np.load("data/features/X_test.npy")
y_test = np.load("data/features/y_test.npy")

results = predictor.evaluate_on_test_set(X_test, y_test)

for model_name, metrics in results.items():
    print(f"{model_name}: {metrics['accuracy']:.2%}")
```

---

##### `get_model_configs()`

Get model configurations.

**Returns:**
- `configs` (dict): Configuration dictionaries for each loaded model

**Example:**
```python
configs = predictor.get_model_configs()
print(configs['svm'])
# {
#   'model_type': 'SVM',
#   'C': 10,
#   'gamma': 0.01,
#   'kernel': 'rbf',
#   'metrics': {...}
# }
```

---

## Analysis Tools

### `KNNAnalyzer` Class

KNN model analysis and visualization.

**Location:** `src/models/knn_analysis_helper.py`

#### Methods

##### `__init__(model_dir)`

Initialize analyzer.

```python
from src.models.knn_analysis_helper import KNNAnalyzer

analyzer = KNNAnalyzer(model_dir="saved_models")
```

##### `plot_confusion_matrix(y_true, y_pred, class_names, save=True)`

Create confusion matrix visualization.

**Parameters:**
- `y_true` (ndarray): True labels
- `y_pred` (ndarray): Predicted labels
- `class_names` (list): Class names
- `save` (bool): Save to file

**Example:**
```python
y_pred = analyzer.model.predict(X_test)
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
analyzer.plot_confusion_matrix(y_test, y_pred, class_names)
```

---

##### `get_model_info()`

Print model information.

```python
analyzer.get_model_info()
# [KNN MODEL INFORMATION]
# Model Type: KNN
# Feature Dimension: 8336
# Best Parameters:
#   n_neighbors: 7
#   weights: distance
#   metric: euclidean
# Metrics:
#   train_accuracy: 0.8954
#   val_accuracy: 0.8421
#   test_accuracy: 0.8312
```

---

##### `compare_models(svm_config_path)`

Compare KNN vs SVM performance.

**Parameters:**
- `svm_config_path` (str): Path to SVM config file

**Example:**
```python
analyzer.compare_models("saved_models/svm_config.json")
```

Generates comparison visualization and prints metrics table.

---

### `SVMAnalyzer` Functions

SVM analysis and error investigation.

**Location:** `src/models/svm_analysis.py`

#### `analyze_misclassifications(model, X_test, y_test, threshold)`

Analyze misclassified samples.

**Parameters:**
- `model`: Trained SVM model
- `X_test`: Test features
- `y_test`: Test labels
- `threshold`: Confidence threshold (0.0-1.0)

**Returns:**
- `misclassified_indices`: Indices of misclassified samples
- `predictions`: Adjusted predictions
- `confidences`: Prediction confidences

**Example:**
```python
from src.models.svm_analysis import analyze_misclassifications
import pickle

model = pickle.load(open("saved_models/svm_model.pkl", "rb"))
X_test = np.load("data/features/X_test.npy")
y_test = np.load("data/features/y_test.npy")

indices, preds, confs = analyze_misclassifications(model, X_test, y_test, 0.5)
print(f"Misclassified: {len(indices)}/{len(y_test)}")
```

---

#### `analyze_per_class_performance(model, X_test, y_test, threshold)`

Analyze performance metrics per class.

**Returns:**
- Dictionary with per-class metrics (precision, recall, F1)

---

## Data Structures

### Configuration JSON Format

**SVM Config:**
```json
{
  "model_type": "SVM",
  "kernel": "rbf",
  "C": 10,
  "gamma": 0.01,
  "probability": true,
  "cache_size": 1000,
  "random_state": 42,
  "metrics": {
    "train_accuracy": 0.9234,
    "val_accuracy": 0.8756,
    "test_accuracy": 0.8642,
    "precision": {
      "glass": 0.92,
      "paper": 0.85,
      ...
    },
    "recall": {...},
    "f1_score": {...}
  },
  "class_names": [
    "glass", "paper", "cardboard",
    "plastic", "metal", "trash", "unknown"
  ],
  "confusion_matrix": [[45, 2, ...], ...]
}
```

**KNN Config:**
```json
{
  "model_type": "KNN",
  "n_neighbors": 7,
  "weights": "distance",
  "metric": "euclidean",
  "algorithm": "auto",
  "metrics": {
    "train_accuracy": 0.8954,
    "val_accuracy": 0.8421,
    "test_accuracy": 0.8312
  },
  "class_names": [
    "glass", "paper", "cardboard",
    "plastic", "metal", "trash", "unknown"
  ],
  "feature_dimension": 8336
}
```

---

### Features Info JSON

```json
{
  "total_samples": 3400,
  "train_samples": 2380,
  "val_samples": 510,
  "test_samples": 510,
  "feature_dimension": 8336,
  "features": {
    "hog": {
      "dimension": 8181,
      "orientations": 9,
      "pixels_per_cell": [8, 8],
      "cells_per_block": [2, 2]
    },
    "color_histogram": {
      "dimension": 96,
      "bins": 32,
      "channels": 3
    },
    "lbp": {
      "dimension": 59,
      "radius": 1,
      "points": 8,
      "method": "uniform"
    }
  },
  "class_distribution": {
    "glass": 500,
    "paper": 500,
    "cardboard": 500,
    "plastic": 500,
    "metal": 500,
    "trash": 500,
    "unknown": 400
  }
}
```

---

### NumPy Array Formats

**Feature Arrays:**
```python
# X_train.npy
X_train = np.load("data/features/X_train.npy")
print(X_train.shape)  # (2380, 8336)
print(X_train.dtype)  # float32

# Each row is a feature vector for one image
# Features are normalized to zero mean and unit variance

# y_train.npy
y_train = np.load("data/features/y_train.npy")
print(y_train.shape)  # (2380,)
print(y_train.dtype)  # int64
print(np.unique(y_train))  # [0 1 2 3 4 5 6]
```

---

## Constants & Defaults

### Class Names

```python
CLASS_NAMES = [
    'glass',      # Index 0
    'paper',      # Index 1
    'cardboard',  # Index 2
    'plastic',    # Index 3
    'metal',      # Index 4
    'trash',      # Index 5
    'unknown'     # Index 6
]
```

### Feature Dimensions

```python
HOG_DIMENSION = 8181
COLOR_HISTOGRAM_DIMENSION = 96
LBP_DIMENSION = 59
TOTAL_DIMENSION = 8336
```

### Image Processing

```python
IMAGE_SIZE = (128, 128)  # Target size for all images
COLOR_SPACE = 'BGR'       # OpenCV default (not RGB!)
```

### Hyperparameter Defaults

**SVM:**
```python
default_svm_params = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'probability': True,
    'cache_size': 200,
    'random_state': 42
}
```

**KNN:**
```python
default_knn_params = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'metric': 'minkowski',
    'algorithm': 'auto'
}
```

---

## Error Handling

### Common Exceptions

```python
# FileNotFoundError
# Raised when model files don't exist
try:
    predictor = UnifiedPredictor()
except FileNotFoundError as e:
    print(f"Models not found: {e}")

# ValueError
# Raised with invalid parameters
try:
    pred = predictor.predict_single(features, model_name='invalid')
except ValueError as e:
    print(f"Invalid model: {e}")

# RuntimeError
# Raised during feature extraction
try:
    features = extract_combined_features(invalid_image)
except RuntimeError as e:
    print(f"Feature extraction error: {e}")
```

---

## Performance & Complexity

### Time Complexity

```python
# Single prediction
extract_combined_features(): O(H × W)  # H=128, W=128
predict_single():            O(n_features) for SVM
                             O(n_train) for KNN

# Batch prediction (N samples)
batch_predict():             O(N × n_features)

# Feature extraction (M images)
extract_features_from_dataset(): O(M × H × W)
```

### Space Complexity

```python
# Memory usage
trained_svm:  ~50 MB
trained_knn:  ~100 MB
X_train:      ~2GB (2380 × 8336 × 4 bytes)
All models:   ~2.5 GB total
```

---

## Advanced Usage

### Custom Model Evaluation

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)

# Get predictions
y_pred = predictor.models['svm'].predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ROC-AUC (requires probabilities)
y_proba = predictor.models['svm'].predict_proba(X_test)
# For multiclass: use 'ovr' or 'ovo'
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
```

---

## Version Information

```python
import src
print(src.__version__)  # 0.1.0

# Dependencies
import cv2; print(cv2.__version__)
import numpy; print(numpy.__version__)
import sklearn; print(sklearn.__version__)
```

---

## Support

For issues or questions:
1. Check logs in `results/` directory
2. Review error messages
3. Refer to phase-specific documentation
4. Check example scripts in `src/` directory

**Last Updated**: December 2024  
**Version**: 0.1.0
