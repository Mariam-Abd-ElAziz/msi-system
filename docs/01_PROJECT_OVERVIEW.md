# MSI System - Material Stream Identification

## Project Overview

The **Material Stream Identification (MSI) System** is a comprehensive machine learning pipeline designed to automatically classify and identify different material types in waste streams. This system uses computer vision techniques and advanced machine learning algorithms to distinguish between **seven different material categories**:

1. **Glass** - Transparent/translucent glass items
2. **Paper** - Paper products and cardboard-like materials
3. **Cardboard** - Corrugated cardboard boxes and sheets
4. **Plastic** - Various plastic materials and products
5. **Metal** - Metallic items (aluminum, steel, etc.)
6. **Trash** - Mixed/ambiguous waste materials
7. **Unknown** - Items that don't clearly fit into categories 1-6

---

## System Architecture

The MSI System is built on a **three-phase machine learning pipeline**:

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT: RAW IMAGES                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  PHASE 1: DATA AUGMENTATION  │
        │  (Preprocessing)              │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  PHASE 2: FEATURE EXTRACTION │
        │  (HOG + Color + LBP)          │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  PHASE 3: MODEL TRAINING     │
        │  (SVM & KNN)                  │
        └──────────────┬───────────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────────┐
   │    UNIFIED PREDICTOR (Inference)            │
   │    - Single model predictions                │
   │    - Ensemble voting                         │
   │    - Confidence scoring                      │
   └─────────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  OUTPUT: CLASS PREDICTIONS   │
        │  + Confidence Scores         │
        └──────────────────────────────┘
```

---

## Key Components

### 1. **Data Preparation & Augmentation** (`src/preprocessing/augmentation.py`)
- Analyzes original dataset distribution
- Identifies class imbalances
- Augments images using:
  - Rotation (±30°)
  - Brightness adjustment (70%-130%)
  - Zoom (80%-120%)
  - Flipping (horizontal)
  - Translation (±10% pixels)
- Generates "Unknown" class from heavily augmented images
- **Output**: Balanced augmented dataset (~500 images per class)

### 2. **Feature Extraction** (`src/preprocessing/feature_extractor.py`)
- Extracts three complementary feature types:
  - **HOG (Histogram of Oriented Gradients)**: Edge and shape information (8,181 dimensions)
  - **Color Histogram**: Color distribution information (96 dimensions)
  - **LBP (Local Binary Patterns)**: Texture information (59 dimensions)
- Combines features into unified vectors (8,336 dimensions total)
- Normalizes features for model training
- **Output**: Feature vectors with corresponding labels for training/validation/test

### 3. **Model Training** (`src/models/`)
- **SVM (Support Vector Machine)**:
  - Kernel: RBF (Radial Basis Function)
  - Hyperparameter tuning via GridSearchCV
  - Optimal for complex decision boundaries
  - Probability estimates for confidence scoring
  
- **KNN (K-Nearest Neighbors)**:
  - Simple but effective distance-based classifier
  - Hyperparameter tuning (k-value, distance metric)
  - Fast inference time
  - Natural probability estimates

### 4. **Unified Predictor** (`src/models/unified_predictor.py`)
- Loads both trained models
- Provides single-model predictions
- Implements ensemble voting
- Supports batch predictions
- Returns confidence scores

---

## Technology Stack

### Dependencies
- **Computer Vision**: OpenCV (4.12.0+), scikit-image (0.25.2+)
- **Data Processing**: NumPy (<2.3.0), Pillow (12.0.0+)
- **Machine Learning**: scikit-learn (1.8.0+)
- **Visualization**: Matplotlib (3.10.8+), Seaborn (0.13.2+)
- **Utilities**: TQDM (4.67.1+) for progress tracking

### Python Requirements
- Python >= 3.12

---

## Directory Structure

```
msi-system/
├── src/
│   ├── preprocessing/
│   │   ├── augmentation.py          # Phase 1: Data augmentation
│   │   └── feature_extractor.py     # Phase 2: Feature extraction
│   ├── models/
│   │   ├── svm_training.py          # SVM model training
│   │   ├── knn_training.py          # KNN model training
│   │   ├── svm_analysis.py          # SVM analysis tools
│   │   ├── knn_analysis_helper.py   # KNN analysis tools
│   │   └── unified_predictor.py     # Unified inference interface
│   └── pipeline/
│       └── feature_analysis.py      # Feature visualization
├── docs/                            # Documentation
├── saved_models/                    # Trained models and configs
├── data/
│   ├── augmented/                   # Augmented dataset
│   └── features/                    # Extracted features
├── results/                         # Analysis plots and reports
├── main_train.py                    # Main training orchestrator
├── pyproject.toml                   # Project configuration
└── README.md                        # Quick start guide
```

---

## Workflow Overview

### Phase 1: Data Preparation (Optional, Manual)
```bash
python src/preprocessing/augmentation.py
```
- Loads original dataset from `dataset/` folder
- Explores class distribution
- Applies augmentation to balance classes
- Generates Unknown class examples
- Saves augmented images to `data/augmented/`

### Phase 2: Feature Extraction (Optional, Manual)
```bash
python src/preprocessing/feature_extractor.py
```
- Loads augmented images
- Extracts HOG, Color Histogram, and LBP features
- Normalizes and combines features
- Splits into train/validation/test sets
- Saves feature files to `data/features/`

### Phase 3: Model Training (Main Pipeline)
```bash
python main_train.py
```
- Trains SVM with hyperparameter tuning
- Trains KNN with hyperparameter tuning
- Generates analysis plots
- Saves models to `saved_models/`
- Creates comparison visualizations

---

## Model Performance

The system is evaluated using multiple metrics:

- **Accuracy**: Overall classification correctness
- **Precision**: Correct positive predictions per class
- **Recall**: Sensitivity for detecting each class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class error patterns
- **ROC Curves**: Model discrimination ability

Models are evaluated on separate train/validation/test sets to ensure generalization performance.

---

## Key Features

✓ **Automatic Data Augmentation**: Handles imbalanced datasets without manual intervention

✓ **Multi-Modal Features**: Combines shape (HOG), color, and texture (LBP) information

✓ **Hyperparameter Optimization**: GridSearchCV automatically finds optimal parameters

✓ **Multiple Models**: Supports both SVM and KNN with easy switching

✓ **Ensemble Voting**: Combines predictions from multiple models for improved accuracy

✓ **Confidence Scoring**: Provides prediction confidence for uncertainty quantification

✓ **Comprehensive Analysis**: Detailed performance reports and visualizations

✓ **Easy Inference**: Unified predictor interface for production deployment

---

## Use Cases

1. **Waste Sorting Automation**: Identify material types in recycling streams
2. **Quality Control**: Verify material composition in manufacturing
3. **Inventory Management**: Classify items in storage facilities
4. **Research**: Benchmark different classification approaches
5. **Production Optimization**: Improve sorting efficiency through data insights

---

## Performance Characteristics

| Aspect | Details |
|--------|---------|
| **Input Image Size** | 128×128 pixels (auto-resized) |
| **Feature Vector Dimension** | 8,336 |
| **Training Time** | 15-30 minutes (full GridSearchCV) |
| **Inference Time** | <50ms per image |
| **Memory Requirements** | ~2GB RAM (model + data) |
| **Number of Classes** | 7 (glass, paper, cardboard, plastic, metal, trash, unknown) |

---

## System Strengths

1. **Robust to Variations**: Data augmentation ensures generalization
2. **Multiple Perspectives**: HOG + Color + LBP capture complementary information
3. **Automatic Optimization**: Hyperparameter tuning removes manual configuration
4. **Ensemble Capability**: Voting improves accuracy over single models
5. **Uncertainty Quantification**: Confidence scores enable quality filtering

---

## Next Steps

To get started with the MSI System:

1. **Read Phase 1 Documentation** ([Phase 1: Data Augmentation](02_PHASE_1_AUGMENTATION.md)) for data preparation
2. **Read Phase 2 Documentation** ([Phase 2: Feature Extraction](03_PHASE_2_FEATURES.md)) for feature engineering
3. **Read Phase 3 Documentation** ([Phase 3: Model Training](04_PHASE_3_TRAINING.md)) for model training
4. **Read the User Manual** ([User Manual](05_MODELS_MANUAL.md)) for making predictions
5. **Read API Reference** ([API Reference](06_API_REFERENCE.md)) for detailed technical information

---

## Questions & Support

For detailed information on specific phases, see the corresponding documentation files in the `docs/` folder.

**Last Updated**: December 2024  
**Version**: 0.1.0
