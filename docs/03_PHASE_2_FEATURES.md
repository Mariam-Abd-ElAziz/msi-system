# Phase 2: Feature Extraction & Representation

## Overview

**Phase 2** is the **feature engineering** phase. It transforms raw images into numerical feature vectors that machine learning models can learn from.

## Why Feature Extraction?

Machine learning models don't work well with raw pixel values:
- **Raw pixels** (128×128×3): 49,152 dimensions - too high-dimensional
- **Noisy**: Pixel values change with lighting, shadows, etc.
- **Inefficient**: Contains redundant information

**Feature extraction** solves this by:
1. Computing meaningful descriptors from images
2. Reducing dimensionality while preserving important information
3. Creating representations that capture material properties

---

## File Location

```
src/preprocessing/feature_extractor.py
```

## Configuration

Key parameters at the top of the file:

```python
# Paths
AUGMENTED_DATA_DIR = 'data/augmented'    # Input from Phase 1
OUTPUT_DIR = 'data/features'             # Output features

# Image preprocessing
IMAGE_SIZE = (128, 128)  # All images resized to this

# HOG parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Color Histogram parameters
COLOR_BINS = 32  # Bins per channel (RGB)

# LBP parameters
LBP_RADIUS = 1
LBP_POINTS = 8
LBP_METHOD = 'uniform'
```

---

## Feature Types

The system extracts **three complementary feature types**. Each captures different information:

### 1. HOG (Histogram of Oriented Gradients)

**What it captures**: Edge directions and shapes (structural information)

#### How It Works

1. **Grayscale conversion**: Color → grayscale
2. **Gradient computation**: Calculate edge magnitude and direction at each pixel
3. **Orientation binning**: Group orientations into 9 bins (0°-180° in 20° steps)
4. **Cell-based histograms**: Divide image into 8×8 pixel cells, compute histogram in each
5. **Block normalization**: Group cells into 2×2 blocks, normalize to handle lighting

```
Image (128×128)
     ↓
Gradients (magnitude + direction)
     ↓
Grid of 8×8 cells (16×16 cells total)
     ↓
Orientation histograms per cell
     ↓
HOG Feature Vector (8,181 dimensions)
```

#### Visual Interpretation

HOG captures **edge structure**:
- Glass bottle → smooth cylindrical edges
- Cardboard box → sharp rectangular edges
- Crumpled paper → irregular edges

**Dimensions**: 8,181 features

#### Example

```python
from src.preprocessing.feature_extractor import extract_hog_features
import cv2

image = cv2.imread('glass_001.jpg')
hog_features = extract_hog_features(image)
# Output: array of 8,181 values

# These values represent the distribution of edge orientations
# in different parts of the image
```

### 2. Color Histogram

**What it captures**: Color distribution (color properties)

#### How It Works

1. **Channel splitting**: BGR image → separate Red, Green, Blue channels
2. **Histogram computation**: For each channel:
   - Divide color range (0-255) into `COLOR_BINS` (32) bins
   - Count pixels falling into each bin
3. **Normalization**: Divide by total pixels to get probabilities
4. **Concatenation**: Combine R, G, B histograms

```
Image (128×128, 3 channels)
     ↓
Red channel histogram (32 bins)
Green channel histogram (32 bins)
Blue channel histogram (32 bins)
     ↓
Color Feature Vector (96 dimensions)
```

#### Visual Interpretation

Color histogram captures **color composition**:
- Glass → Often has blues/greens (from environment reflections)
- Cardboard → Browns, tans, grays
- Metal → Grays, silvers
- Plastic → Varies (red, blue, clear, etc.)

**Dimensions**: 96 features (32 bins × 3 channels)

#### Example

```python
from src.preprocessing.feature_extractor import extract_color_histogram_features

image = cv2.imread('glass_001.jpg')
color_features = extract_color_histogram_features(image)
# Output: array of 96 values

# First 32: Red channel histogram
# Next 32: Green channel histogram
# Last 32: Blue channel histogram
```

### 3. LBP (Local Binary Patterns)

**What it captures**: Texture properties (surface characteristics)

#### How It Works

1. **Grayscale conversion**: Color → grayscale
2. **Local binary patterns**: For each pixel:
   - Compare with 8 neighbors in a circle (radius = 1)
   - Create 8-bit binary number based on comparisons
   - Map to texture pattern
3. **Uniform patterns**: Keep only "uniform" patterns (smooth transitions)
4. **Histogram**: Count occurrences of each LBP pattern

```
Grayscale image (128×128)
     ↓
For each pixel, compute LBP:
    Neighbors > center? → 1
    Neighbors ≤ center? → 0
     ↓
Binary patterns (0-255 range)
     ↓
Histogram of uniform patterns
     ↓
LBP Feature Vector (59 dimensions)
```

#### Visual Interpretation

LBP captures **texture/surface properties**:
- Glass → Smooth texture (few pattern changes)
- Paper → Granular texture (many small patterns)
- Metal → Reflective texture (smooth with highlights)
- Plastic → Varies (smooth or bumpy)

**Dimensions**: 59 features (uniform patterns for P=8, R=1)

#### Example

```python
from src.preprocessing.feature_extractor import extract_lbp_features

image = cv2.imread('cardboard_001.jpg')
lbp_features = extract_lbp_features(image)
# Output: array of 59 values

# Each value = count of a specific texture pattern
```

---

## Combined Feature Vector

The final feature vector **concatenates all three**:

```
Feature Vector = [HOG (8,181) + Color Histogram (96) + LBP (59)]
               = 8,336 dimensions total

┌─────────────────────────────────────────────┐
│  HOG Features (0-8180)                      │ 8,181 dims
│  Edge/Shape information                     │
├─────────────────────────────────────────────┤
│  Color Histogram Features (8181-8276)       │   96 dims
│  Color distribution                         │
├─────────────────────────────────────────────┤
│  LBP Features (8277-8335)                   │   59 dims
│  Texture information                        │
└─────────────────────────────────────────────┘
```

### Why Combine?

**Complementary information**:
- HOG: Captures shape (glass bottle looks different from crumpled paper)
- Color: Captures color (metal is typically gray/silver)
- LBP: Captures texture (glass is smooth, paper is textured)

**Together**: Much more discriminative than any single feature type

---

## Running Phase 2

### Basic Usage

```bash
python src/preprocessing/feature_extractor.py
```

### Expected Output

```
======================================================================
FEATURE EXTRACTION PIPELINE
======================================================================

Analyzing feature dimensions...

Feature Dimensions:
   HOG features:           8,181 dimensions
   Color Histogram:           96 dimensions
   LBP features:              59 dimensions
   ----------------------------------------
   TOTAL combined:         8,336 dimensions

Extracting features from all images...
   Processing images ████████████████████████████████████ 100% | 3400/3400

[OK] Feature extraction complete!
   Total samples: 3,400
   Features shape: (3400, 8336)

Splitting into train/validation/test sets...
   Training set:   2,380 samples (70%)
   Validation set:    510 samples (15%)
   Test set:          510 samples (15%)

[OK] Data normalized using StandardScaler

[SAVE] Saving processed data...
   [OK] Saved: X_train.npy (2380, 8336)
   [OK] Saved: y_train.npy (2380,)
   [OK] Saved: X_val.npy (510, 8336)
   [OK] Saved: y_val.npy (510,)
   [OK] Saved: X_test.npy (510, 8336)
   [OK] Saved: y_test.npy (510,)
   [OK] Saved: features_info.json

[COMPLETE] Feature extraction pipeline finished!
```

### Output Files

```
data/features/
├── X_train.npy              # Training features (2,380 × 8,336)
├── y_train.npy              # Training labels (2,380)
├── X_val.npy                # Validation features (510 × 8,336)
├── y_val.npy                # Validation labels (510)
├── X_test.npy               # Test features (510 × 8,336)
├── y_test.npy               # Test labels (510)
├── scaler.pkl               # StandardScaler for normalization
├── image_paths.pkl          # Mapping to original images
└── features_info.json       # Feature metadata
```

---

## Understanding Feature Dimensions

### Why 8,336 Dimensions?

This might seem like a lot, but it's **much smaller than raw pixels**:

```
Raw pixels:     128 × 128 × 3 = 49,152 dimensions ❌ Too large
Features:       8,336 dimensions ✓ Optimal

Compression ratio: 49,152 / 8,336 = 5.9x smaller
```

### Typical Feature Distribution

For a glass image:
```
HOG features:       [0.12, 0.34, 0.5, ...]  (8,181 values)
                    ↑ High values for smooth edges

Color features:     [0.1, 0.15, 0.5, ...]   (96 values)
                    ↑ Higher blue/green (reflections)

LBP features:       [0.01, 0.02, 0.05, ...] (59 values)
                    ↑ Low values (smooth surface)
```

For a cardboard image:
```
HOG features:       [0.08, 0.25, 0.3, ...]  (8,181 values)
                    ↑ Medium values (regular edges)

Color features:     [0.3, 0.25, 0.1, ...]   (96 values)
                    ↑ Higher reds/browns (cardboard color)

LBP features:       [0.08, 0.12, 0.2, ...]  (59 values)
                    ↑ Higher values (textured surface)
```

---

## Feature Normalization

The script uses **StandardScaler** normalization:

```
Normalized Feature = (Original - Mean) / Standard Deviation
```

**Why normalize?**
- Different features have different ranges
- HOG: typically 0-1, Color: typically 0-1, but still different distributions
- Models converge faster with normalized features
- Prevents high-magnitude features from dominating

**Example:**
```python
# Before normalization
feature_vec = [0.5, 0.3, 1000.5, 0.2]  # Mixed scales!

# After normalization (zero mean, unit variance)
normalized = [-0.1, -0.3, 2.1, -0.4]   # All on same scale
```

---

## Visualizing Features

To understand what features capture, run:

```bash
python src/pipeline/feature_analysis.py
```

This creates detailed visualizations showing:
- **Original image**
- **HOG visualization** (edge map)
- **Color histograms** (RGB distribution)
- **LBP patterns** (texture map)

Example visualization for a single image:

```
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│  Original   │  │ HOG Edges    │  │ Red Channel  │
│   Image     │  │ Visualization│  │  Histogram   │
└─────────────┘  └──────────────┘  └──────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Green Channel │  │ Blue Channel │  │ LBP Texture  │
│  Histogram   │  │  Histogram   │  │     Map      │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Data Splitting Strategy

Features are split into three sets:

### Training Set (70% = 2,380 samples)
- **Purpose**: Used to learn model parameters
- **Usage**: Train SVM and KNN models

### Validation Set (15% = 510 samples)
- **Purpose**: Used to tune hyperparameters and prevent overfitting
- **Usage**: GridSearchCV uses this for cross-validation

### Test Set (15% = 510 samples)
- **Purpose**: Final unbiased performance evaluation
- **Usage**: Report accuracy, precision, recall, etc.

**Key Principle**: Test set is never seen during training!

```
Original Data (3,400 samples)
         │
         ├──→ Training (70%) ──→ Model Learning
         ├──→ Validation (15%) ──→ Hyperparameter Tuning
         └──→ Test (15%) ──→ Performance Evaluation
```

---

## Customizing Feature Extraction

### For Faster Processing

```python
# Use fewer HOG orientations (fewer features)
HOG_ORIENTATIONS = 6  # Instead of 9
HOG_PIXELS_PER_CELL = (16, 16)  # Larger cells (coarser)

# Fewer color bins
COLOR_BINS = 16  # Instead of 32

# Result: ~3,000-4,000 dimensions instead of 8,336
```

### For More Detailed Features

```python
# More HOG orientations
HOG_ORIENTATIONS = 12  # Instead of 9
HOG_PIXELS_PER_CELL = (4, 4)  # Smaller cells (finer)

# More color bins
COLOR_BINS = 64  # Instead of 32

# Result: ~15,000-20,000 dimensions
# Trade-off: More computation time, but richer representation
```

### Adding More Feature Types

You could extend the system with:
- **SIFT/SURF**: Complex corner/interest point features
- **Edge maps**: Canny edge detection features
- **Texture descriptors**: Gabor filters
- **Deep features**: Pre-trained CNN activations

---

## Troubleshooting

### Problem: "No images found"

**Cause**: `AUGMENTED_DATA_DIR` doesn't exist or is wrong

**Solution**:
```bash
# Verify Phase 1 completed
ls data/augmented/
# Should show: glass/, paper/, cardboard/, plastic/, metal/, trash/, unknown/
```

### Problem: Memory error during feature extraction

**Cause**: Too many images or computing on CPU

**Solutions**:
1. Process in batches (modify script to process fewer images per batch)
2. Reduce image size from 128×128 to 96×96
3. Use fewer HOG orientations
4. Close other applications

### Problem: Feature files not found error in Phase 3

**Cause**: Phase 2 didn't complete successfully

**Solution**:
```bash
# Re-run feature extraction
python src/preprocessing/feature_extractor.py

# Verify output files exist
ls data/features/
# Should show: X_train.npy, y_train.npy, X_val.npy, ... etc
```

---

## Feature Space Visualization

Although we can't visualize 8,336 dimensions directly, we can reduce to 2D:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load features
X_train = np.load('data/features/X_train.npy')
y_train = np.load('data/features/y_train.npy')

# Reduce to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_train)

# Plot
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train)
plt.colorbar(label='Class')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.show()
```

This helps visualize **class separability** - how well classes are separated in feature space.

---

## Key Concepts

### **Feature**
A computed value from image that captures meaningful information. Examples:
- HOG: Edge orientation in a cell
- Color: Pixel count in a color bin
- LBP: Texture pattern frequency

### **Feature Vector**
Concatenation of all features into a single vector: `[f1, f2, f3, ..., f8336]`

### **Dimensionality**
Number of features in vector. Higher = more information but more computation.

### **Normalization**
Rescaling features so they're on comparable scales. Essential for distance-based algorithms (KNN, SVM).

### **Class Separability**
How well-separated classes are in feature space. Better separation = easier to classify.

---

## Performance Impact

Feature quality directly affects model performance:

| Aspect | Impact |
|--------|--------|
| **Good features** | Models learn patterns quickly ✓ |
| **Redundant features** | Slower training, similar accuracy |
| **Missing features** | Lower accuracy, missed patterns |
| **Normalized features** | Faster convergence, better results |

---

## Next Steps

After Phase 2 completes:

1. **Verify features**: Check that feature files exist in `data/features/`
2. **Inspect feature statistics**: Use `features_info.json`
3. **Visualize features**: Run `src/pipeline/feature_analysis.py`
4. **Move to Phase 3**: Train models
   ```bash
   python main_train.py
   ```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Input** | Augmented images from `data/augmented/` |
| **Output** | Feature vectors in `data/features/` |
| **Feature Types** | HOG (8,181) + Color (96) + LBP (59) |
| **Total Dimensions** | 8,336 |
| **Compression** | ~6× smaller than raw pixels |
| **Data Split** | 70% train / 15% val / 15% test |
| **Duration** | 2-5 minutes |

**Key Takeaway**: Phase 2 converts raw pixels into meaningful numerical representations that ML models can efficiently learn from. The combination of HOG, color, and LBP captures complementary information about material properties.

---

**Previous**: [Phase 1: Data Augmentation](02_PHASE_1_AUGMENTATION.md)  
**Next**: [Phase 3: Model Training](04_PHASE_3_TRAINING.md)
