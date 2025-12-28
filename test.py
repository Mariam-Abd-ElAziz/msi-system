import pickle
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import csv

# Image preprocessing configuration (must match training)
IMAGE_SIZE = 128
from config import IMAGE_SIZE, MODELS_DIR, TEST_DIR
# Class mapping
CLASS_MAPPING = {
    0: 'cardboard', 1: 'glass', 2: 'metal',
    3: 'paper', 4: 'plastic', 5: 'trash', 6: 'unknown'
}


def load_model_and_scaler(bestModelPath, model_type):
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
    model_type = model_type

    if model_type == 'svm':
        svm_path = model_dir / 'svm_model.pkl'
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
    elif model_type == 'knn':
        knn_path = model_dir / 'knn_model.pkl'
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

    if model_type == 'svm':
        # SVM with confidence thresholding
        probabilities = model.predict_proba(features)

        for prob in probabilities:
            max_prob = prob.max()
            class_id = prob.argmax()

            # Apply rejection threshold
            # REJECTION MECHANISM
            # If the model is not confident enough, force 'unknown'
            if threshold is not None and max_prob < threshold:
                class_id = 6  # Unknown class
                print(f"   -> Rejected sample (Conf: {max_prob:.2f} < {threshold})")

            class_name = CLASS_MAPPING.get(class_id, 'unknown')
            predictions.append(class_name)
            confidences.append(max_prob)
    # --- KNN STRATEGY: DISTANCE THRESHOLD ---
    elif model_type == 'knn':
        # 1. Get neighbors and distances
        # kneighbors returns (distances, indices) for the k nearest neighbors
        # This tells us HOW FAR the test image is from the training images
        distances, indices = model.kneighbors(features)

        # 2. Calculate the average distance to the k neighbors
        mean_distances = distances.mean(axis=1)

        # 3. Predict based on voting (standard KNN)
        raw_predictions = model.predict(features)

        # 4. REJECTION MECHANISM
        # If threshold is None, default to a safe value (e.g. 15.0 - requires tuning)
        # Higher distance = Image is very different from training data
        dist_threshold = threshold if threshold is not None else 15.0
        # dist_threshold = the ouput number of the previous cell

        for i, (pred_id, dist) in enumerate(zip(raw_predictions, mean_distances)):
            final_id = pred_id

            # If the image is too far away from known samples -> Unknown
            if dist > dist_threshold:
                final_id = 6  # Force Unknown ID
                print(f"   -> Rejected sample (Dist: {dist:.2f} > {dist_threshold})")

            # Create a confidence score (inverse of distance) for display
            # Closer distance = Higher confidence
            confidence = 1.0 / (1.0 + dist)

            class_name = CLASS_MAPPING.get(final_id, 'unknown')
            predictions.append(class_name)
            confidences.append(confidence)

  # elif model_type == 'knn':
  #     # KNN prediction
  #     if hasattr(model, 'predict_proba'):
  #         probabilities = model.predict_proba(features)

  #         for prob in probabilities:
  #             max_prob = prob.max()
  #             class_id = prob.argmax()
  #             class_name = CLASS_MAPPING.get(class_id, 'unknown')
  #             predictions.append(class_name)
  #             confidences.append(max_prob)
  #     else:
  #         # KNN without probabilities
  #         class_ids = model.predict(features)
  #         predictions = [CLASS_MAPPING.get(cid, 'unknown') for cid in class_ids]
  #         confidences = [1.0] * len(predictions)  # Placeholder

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


def predict(dataFilePath, bestModelPath, model_type, manual_threshold=None):
    """
    Main prediction function with CSV export
    """
    try:
        print("\n" + "=" * 70)
        print("PREDICTION PIPELINE")
        print("=" * 70)
        print(f"Data folder: {dataFilePath}")
        print(f"Model path: {bestModelPath}")
        print()

        # Step 1: Load model and preprocessing components
        model, scaler, cnn_model, device, transform, model_type, loaded_threshold = load_model_and_scaler(bestModelPath, model_type)

        # PRIORITY LOGIC:
        # 1. Use manual_threshold if provided (for testing)
        # 2. Else use loaded_threshold from config file
        # 3. Else default to 0.6
        if manual_threshold is not None:
            final_threshold = manual_threshold
        elif loaded_threshold is not None:
            final_threshold = loaded_threshold
        else:
            final_threshold = 0.6

        print(f"[CONFIG] Using Rejection Threshold: {final_threshold} for {model_type}")

        # Step 2: Load images from folder
        images, image_files = load_images_from_folder(dataFilePath)

        if len(images) == 0:
            print("[WARNING] No valid images found in folder")
            return []

        # Step 3: Extract features
        features = extract_features(images, cnn_model, scaler, transform, device)

        # Step 4: Make predictions
        predictions, confidences = make_predictions(model, features, model_type, final_threshold)

        # Step 5: Display results
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"{'Image':<30} {'Prediction':<15} {'Confidence':<10}")
        print("-" * 70)

        for img_name, pred, conf in zip(image_files, predictions, confidences):
            print(f"{img_name:<30} {pred:<15} {conf:.4f}")

        print("=" * 70 + "\n")

        # =========================================================
        # Step 6: Save Results to CSV [NEW ADDITION]
        # =========================================================
        # Save in a 'results' folder next to your 'saved_models' folder
        results_dir = Path('/content/results')
        results_dir.mkdir(parents=True, exist_ok=True)

        csv_filename = f"prediction_results_{model_type}.csv"
        csv_path = results_dir / csv_filename

        print(f"[SAVING] Saving results to CSV: {csv_path}")

        try:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Write Header
                writer.writerow(['Image Name', 'Prediction', 'Confidence', 'Threshold Used'])

                # Write Rows
                for img_name, pred, conf in zip(image_files, predictions, confidences):
                    writer.writerow([img_name, pred, f"{conf:.4f}", final_threshold])

            print(f"[OK] CSV saved successfully!")
        except Exception as e:
            print(f"[ERROR] Could not save CSV: {e}")

        return predictions

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    dataFilePath = TEST_DIR
    bestModelPath = MODELS_DIR

    # CHANGE THIS VALUE to the one you saw in Cell 7 output
    MY_KNN_THRESHOLD = 70.94

    # Run prediction
    results = predict(dataFilePath, bestModelPath, model_type='svm')
    #results = predict(dataFilePath, bestModelPath, model_type='knn', manual_threshold=MY_KNN_THRESHOLD)

    print(f"\n[SUMMARY] Predicted {len(results)} images")
    print(f"[RESULTS] {results}")