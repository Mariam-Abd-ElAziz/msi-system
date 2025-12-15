# Phase 3: Model Training & Optimization

## Overview

**Phase 3** trains the machine learning classification models using the features extracted in Phase 2. This phase includes hyperparameter tuning, model training, evaluation, and comparative analysis.

## System Architecture

Phase 3 trains **two complementary models**:

```
Feature Vectors (8,336 dims)
     │
     ├──────────────┬──────────────┐
     │              │              │
     ▼              ▼              ▼
  Training      Validation       Test
  (70%)         (15%)           (15%)
     │              │              │
     │              │              │
     ├──────────────┼──────────────┤
     │              │
     ▼              ▼
  ┌───────────────────────────┐
  │  Model Training           │
  ├───────────────────────────┤
  │ 1. SVM (Support Vector    │
  │    Machine)               │
  │ 2. KNN (K-Nearest         │
  │    Neighbors)             │
  └───────────────────────────┘
     │              │
     ▼              ▼
┌──────────────┐ ┌──────────────┐
│ SVM Model    │ │ KNN Model    │
│ (trained)    │ │ (trained)    │
└──────────────┘ └──────────────┘
     │              │
     └──────┬───────┘
            ▼
    ┌──────────────────┐
    │ Ensemble Voting  │
    │ & Analysis       │
    └──────────────────┘
```

---

## File Locations

```
src/models/svm_training.py          # SVM model training
src/models/knn_training.py          # KNN model training
src/models/svm_analysis.py          # SVM analysis tools
src/models/knn_analysis_helper.py   # KNN analysis tools
src/models/unified_predictor.py     # Unified inference
main_train.py                        # Orchestrator script
```

---

## Running Phase 3

### Full Training Pipeline

```bash
# From project root
python main_train.py
```

This automatically:
1. Trains SVM with hyperparameter tuning
2. Trains KNN with hyperparameter tuning
3. Generates analysis plots
4. Creates model comparisons
5. Saves all models and configurations

### Expected Output

```
============================================================
  MATERIAL STREAM IDENTIFICATION (MSI) - TRAINING PIPELINE
============================================================

[OK] Phase 1: Data Augmentation
  (Run src/preprocessing/augmentation.py separately if needed)

[OK] Phase 2: Feature Extraction
  (Run src/preprocessing/feature_extractor.py separately if needed)

============================================================
  PHASE 3: MODEL TRAINING
============================================================

Training SVM Model...
...
Training KNN Model...
...
Model Analysis
...
Unified Predictor Setup
[OK] Unified predictor ready for inference
[OK] Loaded models: ['svm', 'knn']

============================================================
  TRAINING COMPLETE
============================================================
Next steps:
  1. Review saved models in 'saved_models/' directory
  2. Check analysis plots (confusion matrices, comparisons)
  3. Use UnifiedPredictor for inference on new data
```

**Execution Time**: 15-30 minutes (depending on system and GridSearchCV settings)

---

## Model 1: Support Vector Machine (SVM)

### What is SVM?

SVM is a powerful classification algorithm that:
1. Finds an optimal **decision boundary** between classes
2. Maximizes the **margin** (distance) between classes
3. Can handle **non-linear** patterns via kernels

### How SVM Works

#### Basic Concept

```
Feature Space (simplified 2D):

Glass ●●●    Decision Boundary
       ●●      ↑
        ●      │
    ─────────┼─────  ← Optimal hyperplane
        ○      │
       ○○      ↓
Paper ○○○

Margin: Distance from boundary to nearest points
Goal: Maximize margin (more confident predictions)
```

#### Training Process

1. **Data Loading**: Loads feature vectors from Phase 2
2. **Hyperparameter Tuning**: Uses GridSearchCV to find optimal parameters
3. **Cross-Validation**: Tests parameters using 5-fold CV
4. **Model Training**: Trains final model with best parameters
5. **Evaluation**: Measures performance on test set

### SVM Hyperparameter Tuning

The script performs automatic hyperparameter optimization:

```python
param_grid = {
    'C': [0.1, 1, 10, 50, 100],          # Regularization strength
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Kernel coefficient
    'kernel': ['rbf', 'poly', 'linear']   # Decision boundary shape
}
```

#### Parameter Explanations

**C (Regularization Parameter)**
- Controls how much misclassification is tolerated
- **Low C** (0.1): More tolerance for errors, smoother boundary
- **High C** (100): Stricter on errors, more complex boundary
- **Typical**: 1-10 for most problems

```
C = 0.1 (loose):
    ●●●        ○○○         Allows some misclassifications
     ●●  ─────────  ○○    More generalizable
      ●           ○

C = 100 (strict):
    ●●●        ○○○         Fits training data tightly
     ●●────┐┌────○○    Less generalizable (overfitting)
      ●    ││    ○
```

**Gamma (Kernel Coefficient)**
- Determines the reach of a single training example
- **Low gamma** (0.001): Each point has far reach, smooth boundary
- **High gamma** (0.1): Each point has near reach, complex boundary
- **'scale'**: 1/(n_features * X.var()) - usually optimal

```
Gamma = 0.001 (smooth):
    ●●●        ○○○
     ●●  ─────────  ○○    Smooth decision boundary
      ●           ○

Gamma = 0.1 (complex):
    ●●●        ○○○
     ●● ─┐┌┐┌─  ○○    Wiggly, complex boundary
      ● ││││ ○
```

**Kernel (Decision Boundary Shape)**
- **'linear'**: Straight line boundary (works if classes are linearly separable)
- **'rbf'**: Non-linear, circular/curved boundary (most flexible)
- **'poly'**: Polynomial boundary (between linear and RBF)

```
Linear:        RBF:              Polynomial:
 ●●●            ●●●              ●●●
  ●●  ─────────  ○○   ●●  ╱─────╲  ○○
   ●           ○     ●  │Curved  │  ○
```

#### Tuning Output

The script tests all combinations and reports results:

```
[COUNT] Total combinations to test: 75
   With 5-fold CV: 375 model fits
   Estimated time: 15-30 minutes

[START] Starting grid search...
[Progress: CV fold 1/5...]
[Progress: CV fold 2/5...]
...

[OK] Grid search complete!
   Time elapsed: 22.4 minutes

[BEST] Best Parameters:
   kernel       rbf
   C            10
   gamma        0.01

[SCORE] Best Cross-Validation Score: 0.8756

[TOP] Top 5 Parameter Combinations:
   1. Score: 0.8756 (+/- 0.0234)
      Params: C=10, gamma=0.01, kernel=rbf
   2. Score: 0.8698 (+/- 0.0198)
      Params: C=50, gamma=0.01, kernel=rbf
   3. Score: 0.8654 (+/- 0.0276)
      Params: C=10, gamma=0.001, kernel=rbf
   ...
```

### SVM Training

Once best parameters are found, the final model is trained:

```python
svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01,
    probability=True,      # Enable confidence scores
    random_state=42
)
svm_model.fit(X_train, y_train)
```

### SVM Advantages & Disadvantages

| Aspect | Details |
|--------|---------|
| **Advantages** | ✓ Works well with high-dimensional data (8,336 features) |
| | ✓ Robust to outliers |
| | ✓ Effective at finding complex boundaries |
| | ✓ Well-understood and stable |
| **Disadvantages** | ✗ Slower training on very large datasets |
| | ✗ Requires feature scaling (we do this) |
| | ✗ Less interpretable than simpler models |

---

## Model 2: K-Nearest Neighbors (KNN)

### What is KNN?

KNN is a simple but effective algorithm that:
1. Stores all training examples
2. For new samples, finds K nearest neighbors
3. Predicts using majority vote among neighbors

### How KNN Works

#### Basic Concept

```
New Sample X:          Training Data:
  ? (unknown)            ● Glass
                         ● Glass
  X ──check nearest──→  ● Paper
                        ○ Paper
                        ○ Plastic
                        
For K=3: Neighbors are {●, ●, ●}
Result: Majority vote → Glass (2/3)

For K=5: Neighbors are {●, ●, ●, ○, ○}
Result: Majority vote → Glass (3/5)
```

#### Training Process

Unlike SVM, KNN doesn't really "train":
1. Stores all training examples and their labels
2. Uses examples as-is during prediction
3. Tunes hyperparameters (K value, distance metric)

### KNN Hyperparameter Tuning

The script automatically finds optimal parameters:

```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
```

#### Parameter Explanations

**n_neighbors (K)**
- Number of neighbors to consider
- **Low K** (3): Sensitive to noise, can overfit
- **Medium K** (5-7): Good balance (usually best)
- **High K** (20): Smoother decisions, can underfit

```
K=3 (low):          K=7 (medium):       K=15 (high):
?    ?  ●            ?    ?  ●           ?    ?  ●
  ○ X ●               ○ X ●              ○ X ●
   ○●                  ○●                 ○●
Pred: ○             Pred: ●             Pred: ● (majority)
(majority)
```

**weights**
- How to weight neighbors in voting
- **'uniform'**: All neighbors count equally
- **'distance'**: Closer neighbors count more

```
Uniform (all equal):      Distance (closer = more):
     ○ ─ ─ ─ ─ ─ ─ ─ ─ ─ ●      ○ ═════════════════ ●
              X                              X
     ● ─ ─ ─ ─ ─ ─ ─ ─ ─ ○      ● ════════════════ ○

Vote: 2 ●, 2 ○ (tie)      Vote: ● gets 2 votes (closer)
                                 ○ gets 1 vote (farther)
                                 Result: ●
```

**metric** (Distance Measure)
- **'euclidean'**: Straight-line distance (most common)
- **'manhattan'**: Block distance (better for some data types)

```
Euclidean:          Manhattan:
  3 │     ●          3 │     ●
    │    /│           │    ──
  2 │   / │           │     │
    │  /  │           │     │
  1 │ /   X           │     X
    │/    /           │     │
  0 └────────         │     └────────
    0 1 2 3           0 1 2 3

Distance = √((2-0)² + (3-1)²) = 2.83    Distance = |2-0| + |3-1| = 4
```

#### Tuning Output

```
[TUNING] Tuning KNN hyperparameters...

[OK] Best Parameters: 
{'n_neighbors': 7, 'weights': 'distance', 'metric': 'euclidean'}

[OK] Best CV Score: 0.8542
```

### KNN Advantages & Disadvantages

| Aspect | Details |
|--------|---------|
| **Advantages** | ✓ Simple and intuitive |
| | ✓ No training phase (fast setup) |
| | ✓ Provides natural probability estimates |
| | ✓ Often competitive with complex models |
| **Disadvantages** | ✗ Slower prediction (must search all training data) |
| | ✗ Memory-intensive (stores all examples) |
| | ✗ Sensitive to feature scaling (we handle this) |
| | ✗ Sensitive to irrelevant features |

---

## Model Evaluation

Both models are evaluated using multiple metrics:

### Accuracy

Simple: (Correct Predictions) / (Total Predictions)

```python
accuracy = (y_pred == y_test).mean()
# E.g., 87 correct out of 100 = 0.87 accuracy
```

### Per-Class Metrics

Computed for each class individually:

#### Precision
"Of items we predicted as Class X, how many were actually Class X?"

```
       Predicted Positive
            ↓
True  ┌──────┬──────┐
Pos.  │ TP=6 │ FP=1 │ Precision = TP / (TP + FP) = 6/7 = 0.86
      └──────┴──────┘
True  ┌──────┬──────┐
Neg.  │ FN=2 │ TN=91│
      └──────┴──────┘
```

#### Recall
"Of items that actually are Class X, how many did we catch?"

```
Recall = TP / (TP + FN) = 6/8 = 0.75
```

#### F1-Score
Harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.86 × 0.75) / (0.86 + 0.75)
   = 0.80
```

### Confusion Matrix

Shows misclassification patterns:

```
              Predicted Class
                Glass Paper Card. Plastic Metal Trash Unknown
True    Glass    45    2    0      1      0     1    1
Class   Paper     1   48    2      1      0     1    2
        Card.     0    3   46      1      1     2    2
        Plastic   1    1    1     46      0     2    4
        Metal     0    0    1      0     48     1    0
        Trash     2    1    2      1      0    43    1
        Unknown   1    2    1      2      0     1   53
```

**Diagonal (correct predictions)**: 45, 48, 46, 46, 48, 43, 53

Interpreting patterns:
- Cardboard sometimes confused with Paper (makes sense - similar materials)
- Metal almost never confused with others (very distinct)

### ROC-AUC Curves

Measures discrimination ability across different thresholds.

---

## Typical Performance Results

### Example Results

```
╔════════════════════════════════════════════════════════════╗
║             MODEL PERFORMANCE COMPARISON                  ║
╠════════════════════════════════════════════════════════════╣
║                    Training   Validation   Test            ║
║  SVM (RBF):        0.9234     0.8756      0.8642          ║
║  KNN (K=7):        0.8954     0.8421      0.8312          ║
║  Ensemble:         0.9123     0.8642      0.8754          ║
╚════════════════════════════════════════════════════════════╝
```

Interpretation:
- **SVM** achieves 86.42% accuracy on test set
- **KNN** achieves 83.12% accuracy on test set
- **Ensemble** (voting both) achieves 87.54% (best)
- All models have similar train/test performance (good generalization)

---

## Saved Artifacts

### Model Files

```
saved_models/
├── svm_model.pkl          # Trained SVM (sklearn format)
├── svm_config.json        # SVM parameters and metrics
├── knn_model.pkl          # Trained KNN (sklearn format)
├── knn_config.json        # KNN parameters and metrics
└── scaler.pkl             # Feature normalization
```

### Configuration Files

```json
// svm_config.json
{
  "model_type": "SVM",
  "kernel": "rbf",
  "C": 10,
  "gamma": 0.01,
  "metrics": {
    "train_accuracy": 0.9234,
    "val_accuracy": 0.8756,
    "test_accuracy": 0.8642
  },
  "class_names": [
    "glass", "paper", "cardboard", 
    "plastic", "metal", "trash", "unknown"
  ]
}
```

### Analysis Plots

```
results/
├── svm_confusion_matrix.png           # Confusion matrix
├── svm_roc_curves.png                 # ROC-AUC curves
├── hyperparameter_heatmap.png         # Parameter impact
├── misclassified_examples.png         # Examples of errors
├── knn_vs_svm_comparison.png          # Model comparison
└── feature_analysis.png               # Feature visualizations
```

---

## Customizing Training

### Quick Training (Reduced GridSearchCV)

Modify `main_train.py`:

```python
# In svm_training.py:
best_params, grid_search = tune_svm_hyperparameters(
    X_train, y_train, quick_mode=True  # ← Set to True
)

# Reduces from 75 to ~20 combinations
# Time: 5-10 minutes instead of 15-30
```

### More Aggressive Parameter Search

```python
param_grid = {
    'C': [0.01, 0.1, 1, 10, 50, 100, 200],    # More C values
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
}

# Total: 7 × 7 × 4 = 196 combinations
# Time: 45-60 minutes
```

### Custom KNN Parameters

```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Fewer K values
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean']  # Single metric
}
```

---

## Troubleshooting

### Problem: "No such file or directory: X_train.npy"

**Cause**: Phase 2 didn't complete or is in wrong location

**Solution**:
```bash
# Re-run feature extraction
python src/preprocessing/feature_extractor.py

# Verify files exist
ls data/features/
```

### Problem: Training extremely slow

**Cause**: Large GridSearchCV parameter grid or system resources

**Solutions**:
1. Use `quick_mode=True` for SVM
2. Reduce parameter grid size
3. Close other applications
4. Use `-1` for `n_jobs` (parallel processing)

### Problem: Model accuracy is very low (<60%)

**Cause**: Features might be poor quality, or phases not completed correctly

**Solutions**:
1. Verify Phase 1 augmentation completed
2. Verify Phase 2 feature extraction completed
3. Check feature dimensions are 8,336
4. Visualize features: `python src/pipeline/feature_analysis.py`
5. Check for class imbalance in training data

### Problem: Overfitting (high train accuracy, low test accuracy)

**Cause**: Model learning training data too specifically

**Solutions**:
1. Increase regularization: Lower `C` parameter for SVM
2. Increase `K` for KNN
3. Add more training data via augmentation
4. Reduce feature dimensionality

---

## Key Concepts

### **Hyperparameter**
Parameter that controls learning process (not learned from data). Examples: C, gamma, K.

### **GridSearchCV**
Automated tool that tests many parameter combinations and finds best.

### **Cross-Validation**
Technique that splits data into K folds, trains on K-1, tests on 1, repeats K times.

### **Overfitting**
Model learns training data too specifically, performs poorly on new data.

### **Generalization**
Model's ability to perform well on unseen data (important!).

---

## Performance Optimization

### If Test Accuracy is Still Low:

1. **Try different kernel**: Change from RBF to Poly or Linear
2. **Use ensemble**: Combine SVM and KNN predictions (voting)
3. **Collect more data**: More training examples helps
4. **Improve features**: Add SIFT, SURF, or deep learning features
5. **Adjust class weights**: Give more weight to difficult classes

### If Training is Too Slow:

1. Use `quick_mode=True` for faster tuning
2. Reduce parameter grid size
3. Use parallel processing (`n_jobs=-1`)
4. Reduce training data size
5. Simplify features (fewer HOG orientations)

---

## Next Steps

After Phase 3 completes:

1. **Review results**: Check confusion matrices and accuracy scores
2. **Analyze errors**: Look at misclassified examples
3. **Use for prediction**: See [User Manual](05_MODELS_MANUAL.md)
4. **Deploy model**: Use UnifiedPredictor for production

---

## Summary

| Aspect | Details |
|--------|---------|
| **Input** | Feature vectors from Phase 2 |
| **Models** | SVM + KNN |
| **Output** | Trained models in `saved_models/` |
| **Training Time** | 15-30 minutes |
| **Hyperparameter Tuning** | Automatic GridSearchCV |
| **Evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix |
| **Typical Accuracy** | 83-88% on test set |

**Key Takeaway**: Phase 3 automatically trains and tunes two complementary models, selecting optimal parameters and providing comprehensive evaluation. The combination of SVM and KNN often outperforms either model alone.

---

**Previous**: [Phase 2: Feature Extraction](03_PHASE_2_FEATURES.md)  
**Next**: [User Manual: Using the Models](05_MODELS_MANUAL.md)
