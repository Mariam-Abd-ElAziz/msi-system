"""
Material Stream Identification (MSI) System
A machine learning pipeline for material classification using computer vision.

Directory Structure:
- preprocessing/: Data augmentation and feature extraction
  - augmentation.py: Data augmentation (rotation, brightness, zoom, flip, translation)
  - feature_extractor.py: HOG, Color Histograms, and LBP feature extraction
  
- models/: Model training and inference
  - svm_training.py: SVM training with hyperparameter tuning
  - svm_analysis.py: SVM performance analysis and diagnostics
  - knn_training.py: KNN training with GridSearchCV tuning
  - knn_analysis_helper.py: KNN performance analysis and model comparison
  - unified_predictor.py: Multi-model inference with ensemble support
  
- pipeline/: Feature visualization and analysis utilities
  - feature_analysis.py: Feature visualization and dimension breakdown

Classification Task:
- 8 classes: glass, paper, cardboard, plastic, metal, trash, unknown
- Feature dimension: 8,255 (HOG + Color Histograms + LBP combined)
"""

# Import main classes for convenience
from src.preprocessing.feature_extractor import main as run_feature_extraction
from src.preprocessing.augmentation import main as run_augmentation
from src.models.knn_training import KNNTrainer
from src.models.knn_analysis_helper import KNNAnalyzer

__all__ = [
    'run_feature_extraction',
    'run_augmentation',
    'KNNTrainer',
    'KNNAnalyzer'
]
