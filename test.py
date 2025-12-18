"""
Test Prediction Pipeline
Material Stream Identification System

This script provides a predict function that:
1. Loads images from a given folder path
2. Loads the trained model
3. Extracts features using the CNN backbone
4. Makes predictions and returns results
"""

import os
import sys
import pickle
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path


# Image preprocessing configuration (must match training)
IMAGE_SIZE = 128

# Class mapping
CLASS_MAPPING = {
    0: 'glass', 1: 'paper', 2: 'cardboard',
    3: 'plastic', 4: 'metal', 5: 'trash', 6: 'unknown'
}


def load_model_and_scaler(bestModelPath):
    """
    Load the trained SVM/KNN model, feature scaler, and CNN backbone
    
    Args:
        bestModelPath: Path to the saved_models directory
    
    Returns:
        model: Trained classifier (SVM or KNN)
        scaler: Feature scaler
        cnn_model: CNN feature extractor
        device: torch device
        transform: Image transformation pipeline
    """
    model_dir = Path(bestModelPath)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load CNN backbone for feature extraction
    print("[INFO] Loading CNN feature extractor...")
    cnn_path = model_dir / 'cnn_feature_extractor.pth'
    
    if not cnn_path.exists():
        raise FileNotFoundError(f"CNN model not found at {cnn_path}")
    
    cnn_model = models.resnet50(pretrained=False)
    cnn_model.fc = nn.Identity()  # Remove classifier head
    
    state = torch.load(cnn_path, map_location=device)
    cnn_model.load_state_dict(state, strict=False)
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    
    # Load feature scaler
    print("[INFO] Loading feature scaler...")
    scaler_path = model_dir / 'feature_scaler.pkl'
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Feature scaler not found at {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load classifier model (try SVM first, then KNN)
    model = None
    model_type = None
    
    svm_path = model_dir / 'svm_model.pkl'
    knn_path = model_dir / 'knn_model.pkl'
    
    if svm_path.exists():
        print("[INFO] Loading SVM model...")
        with open(svm_path, 'rb') as f:
            model = pickle.load(f)
        model_type = 'svm'
        
        # Load SVM config for threshold
        config_path = model_dir / 'svm_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            threshold = config.get('optimal_threshold', 0.6)
        else:
            threshold = 0.6
    elif knn_path.exists():
        print("[INFO] Loading KNN model...")
        with open(knn_path, 'rb') as f:
            model = pickle.load(f)
        model_type = 'knn'
        threshold = None  # KNN doesn't use threshold
    else:
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    if model is None:
        raise RuntimeError("Failed to load any model")
    
    print(f"[OK] Loaded {model_type.upper()} model successfully")
    
    # Define image transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return model, scaler, cnn_model, device, transform, model_type, threshold


def load_images_from_folder(dataFilePath):
    """
    Load all images from the given folder path
    
    Args:
        dataFilePath: Path to folder containing images
    
    Returns:
        images: List of (image_array, filename) tuples
    """
    folder_path = Path(dataFilePath)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {dataFilePath}")
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Load all images
    images = []
    image_files = []
    
    for file_path in sorted(folder_path.iterdir()):
        if file_path.suffix.lower() in valid_extensions:
            try:
                # Read image using OpenCV
                img = cv2.imread(str(file_path))
                
                if img is None:
                    print(f"[WARNING] Failed to load: {file_path.name}")
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                image_files.append(file_path.name)
                
            except Exception as e:
                print(f"[WARNING] Error loading {file_path.name}: {e}")
                continue
    
    print(f"[INFO] Loaded {len(images)} images from {dataFilePath}")
    
    return images, image_files


def extract_features(images, cnn_model, scaler, transform, device):
    """
    Extract CNN features from images
    
    Args:
        images: List of image arrays
        cnn_model: CNN feature extractor
        scaler: Feature scaler
        transform: Image transformation pipeline
        device: torch device
    
    Returns:
        features: Numpy array of extracted and scaled features
    """
    features_list = []
    
    print("[INFO] Extracting features...")
    
    with torch.no_grad():
        for i, img in enumerate(images):
            try:
                # Validate image
                if img is None or img.size == 0:
                    features_list.append(np.zeros(2048))
                    continue
                
                if img.ndim != 3 or img.shape[2] != 3:
                    features_list.append(np.zeros(2048))
                    continue
                
                # Ensure uint8
                img = np.clip(img, 0, 255).astype(np.uint8)
                
                # Transform image
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Extract features
                feature_vector = cnn_model(img_tensor)
                feature_vector = feature_vector.cpu().numpy().reshape(1, -1)
                
                # Normalize with scaler
                feature_vector = scaler.transform(feature_vector)
                
                features_list.append(feature_vector.flatten())
                
            except Exception as e:
                print(f"[WARNING] Error extracting features for image {i}: {e}")
                features_list.append(np.zeros(2048))
    
    features = np.array(features_list)
    print(f"[OK] Extracted features with shape: {features.shape}")
    
    return features


def make_predictions(model, features, model_type, threshold=None):
    """
    Make predictions using the loaded model
    
    Args:
        model: Trained classifier
        features: Feature array
        model_type: 'svm' or 'knn'
        threshold: Confidence threshold for SVM (optional)
    
    Returns:
        predictions: List of predicted class names
        confidences: List of confidence scores
    """
    predictions = []
    confidences = []
    
    print("[INFO] Making predictions...")
    
    if model_type == 'svm' and threshold is not None:
        # SVM with confidence thresholding
        probabilities = model.predict_proba(features)
        
        for prob in probabilities:
            max_prob = prob.max()
            class_id = prob.argmax()
            
            # Apply rejection threshold
            if max_prob < threshold:
                class_id = 6  # Unknown class
            
            class_name = CLASS_MAPPING.get(class_id, 'unknown')
            predictions.append(class_name)
            confidences.append(max_prob)
    
    elif model_type == 'knn':
        # KNN prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            
            for prob in probabilities:
                max_prob = prob.max()
                class_id = prob.argmax()
                class_name = CLASS_MAPPING.get(class_id, 'unknown')
                predictions.append(class_name)
                confidences.append(max_prob)
        else:
            # KNN without probabilities
            class_ids = model.predict(features)
            predictions = [CLASS_MAPPING.get(cid, 'unknown') for cid in class_ids]
            confidences = [1.0] * len(predictions)  # Placeholder
    
    else:
        # Simple prediction without threshold
        class_ids = model.predict(features)
        predictions = [CLASS_MAPPING.get(cid, 'unknown') for cid in class_ids]
        
        # Try to get confidence scores
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            confidences = [prob.max() for prob in probabilities]
        else:
            confidences = [1.0] * len(predictions)
    
    print(f"[OK] Generated {len(predictions)} predictions")
    
    return predictions, confidences


def predict(dataFilePath, bestModelPath):
    """
    Main prediction function
    
    Args:
        dataFilePath: Path to folder containing test images
        bestModelPath: Path to saved_models directory
    
    Returns:
        predictions: List of predicted class names for each image
    """
    try:
        print("\n" + "=" * 70)
        print("PREDICTION PIPELINE")
        print("=" * 70)
        print(f"Data folder: {dataFilePath}")
        print(f"Model path: {bestModelPath}")
        print()
        
        # Step 1: Load model and preprocessing components
        model, scaler, cnn_model, device, transform, model_type, threshold = load_model_and_scaler(bestModelPath)
        
        # Step 2: Load images from folder
        images, image_files = load_images_from_folder(dataFilePath)
        
        if len(images) == 0:
            print("[WARNING] No valid images found in folder")
            return []
        
        # Step 3: Extract features
        features = extract_features(images, cnn_model, scaler, transform, device)
        
        # Step 4: Make predictions
        predictions, confidences = make_predictions(model, features, model_type, threshold)
        
        # Step 5: Display results
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"{'Image':<30} {'Prediction':<15} {'Confidence':<10}")
        print("-" * 70)
        
        for img_name, pred, conf in zip(image_files, predictions, confidences):
            print(f"{img_name:<30} {pred:<15} {conf:.4f}")
        
        print("=" * 70 + "\n")
        
        return predictions
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# Example usage
if __name__ == "__main__":
    # Example paths - adjust as needed
    dataFilePath = "data/test_images"
    bestModelPath = "saved_models"
    
    # Run prediction
    results = predict(dataFilePath, bestModelPath)
    
    print(f"\n[SUMMARY] Predicted {len(results)} images")
    print(f"[RESULTS] {results}")