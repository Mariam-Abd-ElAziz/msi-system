"""
Main training orchestrator for the MSI System
Runs all phases of the ML pipeline: augmentation, feature extraction, and model training
"""

import sys
from pathlib import Path
from src import KNNTrainer, KNNAnalyzer, run_augmentation, run_feature_extraction


def check_augmented_data_exists():
    """
    Check if augmented data already exists in data/augmented directory
    Returns True if augmented data is found, False otherwise
    """
    augmented_dir = Path("data/augmented")
    
    if not augmented_dir.exists():
        return False
    
    # Check if all required class directories have images
    required_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'unknown']
    
    for class_name in required_classes:
        class_dir = augmented_dir / class_name
        if not class_dir.exists():
            return False
        
        # Check if directory has at least one image
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        if len(image_files) == 0:
            return False
    
    return True


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    """Main training pipeline"""
    
    print_section("MATERIAL STREAM IDENTIFICATION (MSI) - TRAINING PIPELINE")
    
    # Phase 1: Data Augmentation
    print_section("PHASE 1: DATA AUGMENTATION")
    try:
        if check_augmented_data_exists():
            print("[INFO] Augmented data already exists!")
            print("[INFO] Skipping augmentation phase...")
        else:
            print("[INFO] Augmented data not found. Running augmentation...")
            run_augmentation()
    except Exception as e:
        print(f"[ERROR] Augmentation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Feature Extraction
    print_section("PHASE 2: FEATURE EXTRACTION")
    try:
        run_feature_extraction()
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Model Training
    print_section("PHASE 3: MODEL TRAINING - KNN ONLY")
    
    try:
        # Train KNN
        print("Training KNN Model...")
        knn_trainer = KNNTrainer()
        knn_trainer.train()
        
        # Analysis
        print_section("Model Analysis")
        knn_analyzer = KNNAnalyzer()
        knn_analyzer.get_model_info()
        
        print_section("TRAINING COMPLETE")
        print("Next steps:")
        print("  1. Review saved models in 'saved_models/' directory")
        print("  2. Use KNN model for inference on new data")
        
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
