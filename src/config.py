# Configuration file for dataset paths and parameters
ORIGINAL_DATA_DIR = 'data/raw'  
AUGMENTED_DATA_DIR = 'data/augmented' 
FEATURES_DIR = 'data/features'
MODELS_DIR = 'saved_models'
TRAIN_DIR='data/split/train'
VAL_DIR='data/split/val'
RESULTS_DIR = 'results'
MODEL_FILENAME = 'cnn_feature_extractor.pth'

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash','unknown']
TARGET_IMAGES_PER_CLASS = 500  
