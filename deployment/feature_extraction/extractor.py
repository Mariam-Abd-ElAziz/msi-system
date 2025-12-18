# wrapper/adapter code for the feature extraction module
import sys
import os
import pickle
import cv2
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import feature extraction functions
from src.preprocessing.feature_extractor import extract_combined_features, IMAGE_SIZE

class FeatureExtractor:
    def __init__(self):
        # Calculate the correct path to saved_models directory (works from any directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..', '..')
        # Load the feature scaler for normalization
        scaler_path = os.path.join(project_root, 'saved_models', 'feature_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def extract(self, frame):
        # Validate frame data
        if frame is None:
            print("Warning: Frame is None")
            return np.zeros(8255)
        
        if frame.size == 0:
            print("Warning: Frame is empty")
            return np.zeros(8255)
        
        # Check frame properties
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"Warning: Invalid frame shape: {frame.shape}")
            return np.zeros(8255)
        
        # Ensure frame is valid image data
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Basic data validation
        if np.any(frame < 0) or np.any(frame > 255):
            print("Warning: Frame has invalid pixel values")
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Resize frame to the expected input size
        resized_frame = cv2.resize(frame, IMAGE_SIZE)
        
        try:
            # Extract features
            features = extract_combined_features(resized_frame)
            
            # Normalize features using the scaler
            normalized_features = self.scaler.transform([features])[0]
            
            return normalized_features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(8255)
