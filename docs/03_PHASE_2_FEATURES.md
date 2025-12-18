# Phase 2 — CNN Feature Extractor (ResNet50)
This document describes the ResNet50-based feature extractor implemented in `src/preprocessing/feature_extractor.py`.
It explains configuration, usage, outputs, and how the extracted features are used by downstream classical models (KNN / SVM).

## Overview
The extractor performs two roles:

- Train a ResNet50 classifier head (optional) and save weights (`saved_models/cnn_feature_extractor.pth`).
- Run a ResNet backbone (final pooling features) over your dataset and cache pooled deep features as scaled NumPy arrays for Phase 3.

Design goals:

- Reuse saved CNN weights so you don't retrain every run.
- Produce compact pooled features (global average pooling) suitable for distance-based classifiers.
- Save a fitted `StandardScaler` so downstream pipelines get identical normalization.

## Key files

- Script: `src/preprocessing/feature_extractor.py`
- Outputs directory (default): `data/features/`
- Saved scaler and model: `saved_models/feature_scaler.pkl`, `saved_models/cnn_feature_extractor.pth`

## Configuration (constants inside the script)

- `DATASET_PATH`: default `data/augmented` — input ImageFolder layout
- `MODEL_DIR`: default `saved_models`
- `MODEL_FILENAME`: default `cnn_feature_extractor.pth`
- `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LR`, `TRAIN_RATIO`
- `FEATURES_DIR`: default `data/features`

The script uses torchvision transforms to resize images, convert to tensors, and normalize using ImageNet stats.

## What it does (high level)

1. Loads the dataset with `torchvision.datasets.ImageFolder` (class folders expected under `DATASET_PATH`).
2. Optionally trains a ResNet50 head (if you run the script normally). Best model weights are saved to `saved_models/`.
3. Builds a feature extractor by removing the final fully-connected layer from ResNet50 and global-average-pooling the output.
4. Runs images through the backbone in batches and collects pooled feature vectors (one vector per image).
5. Splits features into train/validation/test (by default it uses `TRAIN_RATIO` for train and the remainder as val; test copies val for completeness). Adjust the script if you need a separate test split.
6. Fits `sklearn.preprocessing.StandardScaler` on training features and transforms all splits.
7. Saves scaled arrays and labels as `.npy` files inside `data/features/` and the scaler as `saved_models/feature_scaler.pkl`.

## Command-line usage

From the repository root (examples):

1) Train (optional) and extract features (default behavior when run directly):

```bash
python src/preprocessing/feature_extractor.py
```

2) Extract features using previously saved weights (`saved_models/cnn_feature_extractor.pth`):

```bash
python src/preprocessing/feature_extractor.py --data-dir data/augmented --weights saved_models/cnn_feature_extractor.pth
```

3) Force CPU (useful on machines without CUDA or for quick local tests):

On Windows PowerShell:

```powershell
$env:FORCE_CPU = "1"; python src/preprocessing/feature_extractor.py --data-dir data/augmented --weights saved_models/cnn_feature_extractor.pth
```

With Poetry:

```bash
poetry run python src/preprocessing/feature_extractor.py --data-dir data/augmented --weights saved_models/cnn_feature_extractor.pth
```

Note: the script accepts parameters via its constants; you can edit those at the top of the file or extend the script with an argument parser for extra flexibility.

## Produced files

After a successful extraction the following files (by default) will be present:

- `data/features/X_train.npy` — scaled training features (NumPy array)
- `data/features/y_train.npy` — training labels
- `data/features/X_val.npy`, `data/features/y_val.npy` — validation split
- `data/features/X_test.npy`, `data/features/y_test.npy` — test split (script currently mirrors val)
- `saved_models/feature_scaler.pkl` — fitted `StandardScaler` object (pickle)
- `saved_models/cnn_feature_extractor.pth` — saved model weights (if training was run)

These files are the canonical inputs for Phase 3 training scripts under `src/models/`.

## Integration with Phase 3 (KNN / SVM)

- `src/models/knn_training.py` and `src/models/svm_training.py` expect scaled `.npy` feature/label files in `data/features/`.
- Workflow:
	1. Run the extractor once to produce `.npy` files and the scaler.
	2. Run `python main_train.py` (or the model-specific training scripts) to consume features.

If you prefer automation, modify the model training scripts to call `extract_and_cache_features()` when the expected files are missing.

## Device selection and environment notes

- The script auto-selects `cuda` when available. To force CPU, set environment variable `FORCE_CPU=1` (or set in PowerShell as shown above).
- If you use Poetry or a virtualenv, install a matching `torch`/`torchvision` wheel for your OS and CUDA version, or install the CPU-only build for simplicity.

Example (Poetry, CPU build):

```bash
poetry add --group pytorch --dev "torch" "torchvision"
```

If you need GPU support, follow PyTorch's official install instructions matching your CUDA toolkit.

## Quick test (tiny local run)

To validate the extractor without training at scale:

1. Create `data/augmented/` with one or two images per class (ImageFolder layout).
2. Run the extractor forcing CPU:

```powershell
$env:FORCE_CPU = "1"; python src/preprocessing/feature_extractor.py --data-dir data/augmented
```

3. Confirm `data/features/` contains `.npy` files and `saved_models/feature_scaler.pkl` exists.

This quick run verifies end-to-end I/O, feature extraction, scaling, and file outputs.

## Troubleshooting

- "No images found": Verify `DATASET_PATH` / `--data-dir` points to `data/augmented/` with class subfolders.
- Memory errors on CPU: reduce `IMAGE_SIZE` or `BATCH_SIZE`, or run on GPU when available.
- Torch import/install issues: ensure `torch` and `torchvision` are installed in the same Python environment used to run the script (Poetry users should install into their Poetry environment).

## Extending the extractor

- Add an argument parser to control image size, batch size, weights path, and overwrite behavior.
- Change the test split logic to produce a separate hold-out test set.
- Export features in other formats (Parquet, HDF5) if required by downstream tools.

---

**Previous**: [Phase 1: Data Augmentation](02_PHASE_1_AUGMENTATION.md)  
**Next**: [Phase 3: Model Training](04_PHASE_3_TRAINING.md)

