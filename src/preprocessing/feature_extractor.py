"""
Phase 2: Feature Extraction Pipeline
Material Stream Identification System

This script handles:
1. HOG (Histogram of Oriented Gradients) feature extraction
2. Color Histogram feature extraction
3. LBP (Local Binary Patterns) feature extraction
4. Feature combination and normalization
5. Data preparation for model training
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
AUGMENTED_DATA_DIR = 'data/augmented'
OUTPUT_DIR = 'data/features'
MODELS_DIR = 'saved_models'

# Image preprocessing
IMAGE_SIZE = (128, 128)  # Resize all images to this size

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

# Class mapping
CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_hog_features(image):
    """
    Extract HOG (Histogram of Oriented Gradients) features
    
    Args:
        image: Input image (BGR format from cv2)
    
    Returns:
        1D numpy array of HOG features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    return features


def extract_color_histogram_features(image):
    """
    Extract Color Histogram features from RGB channels
    
    Args:
        image: Input image (BGR format from cv2)
    
    Returns:
        1D numpy array of color histogram features (concatenated RGB histograms)
    """
    # Split into RGB channels (cv2 uses BGR, so reverse)
    b, g, r = cv2.split(image)
    
    # Compute histogram for each channel
    hist_r = cv2.calcHist([r], [0], None, [COLOR_BINS], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [COLOR_BINS], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [COLOR_BINS], [0, 256])
    
    # Normalize histograms
    hist_r = hist_r / (hist_r.sum() + 1e-7)
    hist_g = hist_g / (hist_g.sum() + 1e-7)
    hist_b = hist_b / (hist_b.sum() + 1e-7)
    
    # Concatenate all histograms
    color_features = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    
    return color_features


def extract_lbp_features(image):
    """
    Extract LBP (Local Binary Patterns) features for texture analysis
    
    Args:
        image: Input image (BGR format from cv2)
    
    Returns:
        1D numpy array of LBP histogram features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute LBP
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    
    # Compute histogram of LBP
    # For uniform patterns with P=8, we get 59 bins (58 uniform + 1 non-uniform)
    n_bins = LBP_POINTS * (LBP_POINTS - 1) + 3
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
    
    return lbp_hist


def extract_combined_features(image):
    """
    Extract and combine all features: HOG + Color Histogram + LBP
    
    Args:
        image: Input image (BGR format from cv2)
    
    Returns:
        1D numpy array of combined features
    """
    # Extract individual features
    hog_features = extract_hog_features(image)
    color_features = extract_color_histogram_features(image)
    lbp_features = extract_lbp_features(image)
    
    # Concatenate all features
    combined_features = np.concatenate([hog_features, color_features, lbp_features])
    
    return combined_features


def preprocess_image(image_path):
    """
    Load and preprocess an image
    
    Args:
        image_path: Path to image file
    
    Returns:
        Preprocessed image or None if failed
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        # Resize to standard size
        img = cv2.resize(img, IMAGE_SIZE)
        
        return img
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# ============================================================================
# FEATURE EXTRACTION PIPELINE
# ============================================================================

def extract_features_from_dataset(data_dir):
    """
    Extract features from all images in the dataset
    
    Args:
        data_dir: Directory containing class folders
    
    Returns:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)
        image_paths: list of image paths
    """
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION PIPELINE")
    print("=" * 70)
    
    features_list = []
    labels_list = []
    image_paths_list = []
    
    # Get first image to determine feature dimension
    print("\nAnalyzing feature dimensions...")
    sample_found = False
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if images:
                sample_img_path = os.path.join(class_dir, images[0])
                sample_img = preprocess_image(sample_img_path)
                if sample_img is not None:
                    sample_features = extract_combined_features(sample_img)

                    # Calculate individual feature dimensions
                    sample_hog = extract_hog_features(sample_img)
                    sample_color = extract_color_histogram_features(sample_img)
                    sample_lbp = extract_lbp_features(sample_img)

                    print(f"\nFeature Dimensions:")
                    print(f"   HOG features:           {len(sample_hog):>6} dimensions")
                    print(f"   Color Histogram:        {len(sample_color):>6} dimensions")
                    print(f"   LBP features:           {len(sample_lbp):>6} dimensions")
                    print(f"   {'-' * 40}")
                    print(f"   TOTAL combined:         {len(sample_features):>6} dimensions")
                    sample_found = True
                    break
        if sample_found:
            break

    if not sample_found:
        print("Error: No valid images found!")
        return None, None, None

    # Process all images
    print(f"\nExtracting features from all images...")

    total_images = 0
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            total_images += len(images)

    with tqdm(total=total_images, desc="   Processing images") as pbar:
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(data_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"   Warning: Skipping {class_name} - folder not found")
                continue

            class_id = CLASS_TO_ID[class_name]

            # Get all images in class
            images = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            # Process each image
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)

                # Preprocess image
                img = preprocess_image(img_path)

                if img is None:
                    pbar.update(1)
                    continue

                # Extract features
                try:
                    features = extract_combined_features(img)

                    features_list.append(features)
                    labels_list.append(class_id)
                    image_paths_list.append(img_path)

                except Exception as e:
                    print(f"\n   Warning: Failed to extract features from {img_name}: {e}")

                pbar.update(1)
    
    # Convert to numpy arrays
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"\n[OK] Feature extraction complete!")
    print(f"   Total samples: {len(features)}")
    print(f"   Feature dimension: {features.shape[1]}")
    print(f"   Labels shape: {labels.shape}")
    
    return features, labels, image_paths_list


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================

def normalize_features(features, scaler=None, save_scaler=True):
    """
    Normalize features using StandardScaler (z-score normalization)
    
    Args:
        features: numpy array of features
        scaler: existing scaler (for test data), or None to fit new scaler
        save_scaler: whether to save the scaler
    
    Returns:
        normalized_features: normalized numpy array
        scaler: the StandardScaler object used
    """
    print("\n" + "=" * 70)
    print("FEATURE NORMALIZATION")
    print("=" * 70)
    
    if scaler is None:
        print("\n[CONFIG] Fitting new StandardScaler...")
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        print(f"   Mean: {scaler.mean_[:5]}... (showing first 5)")
        print(f"   Std:  {scaler.scale_[:5]}... (showing first 5)")
        
        if save_scaler:
            os.makedirs(MODELS_DIR, exist_ok=True)
            scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"\n[SAVEDD] Scaler saved to: {scaler_path}")
    else:
        print("\n[CONFIG] Using existing StandardScaler...")
        normalized_features = scaler.transform(features)
    
    # Verify normalization
    print(f"\n[STATS] Normalization Statistics:")
    print(f"   Original - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    print(f"   Normalized - Mean: {normalized_features.mean():.4f}, Std: {normalized_features.std():.4f}")
    
    print("\n[OK] Feature normalization complete!")
    print("=" * 70)
    
    return normalized_features, scaler


# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_dataset(features, labels, image_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets with stratification
    
    Args:
        features: feature array
        labels: label array
        image_paths: list of image paths
        train_ratio: proportion for training
        val_ratio: proportion for validation
        test_ratio: proportion for testing
    
    Returns:
        Dictionary with train, val, test splits
    """
    from sklearn.model_selection import train_test_split
    
    print("\n" + "=" * 70)
    print("DATASET SPLITTING")
    print("=" * 70)
    
    # First split: train + temp (val + test)
    X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
        features, labels, image_paths,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=labels
    )
    
    # Second split: val and test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
        X_temp, y_temp, paths_temp,
        test_size=val_test_ratio,
        random_state=42,
        stratify=y_temp
    )
    
    # Print split statistics
    print(f"\n[SPLIT] Dataset Split:")
    print(f"   Training:   {len(X_train):>5} samples ({len(X_train)/len(features)*100:.1f}%)")
    print(f"   Validation: {len(X_val):>5} samples ({len(X_val)/len(features)*100:.1f}%)")
    print(f"   Test:       {len(X_test):>5} samples ({len(X_test)/len(features)*100:.1f}%)")
    print(f"   {'-' * 50}")
    print(f"   Total:      {len(features):>5} samples")
    
    # Print class distribution for each split
    print(f"\n[DISTRIBUTION] Class Distribution:")
    print(f"   {'Class':<12} {'Train':<8} {'Val':<8} {'Test':<8}")
    print(f"   {'-' * 50}")
    
    for class_name, class_id in CLASS_TO_ID.items():
        train_count = np.sum(y_train == class_id)
        val_count = np.sum(y_val == class_id)
        test_count = np.sum(y_test == class_id)
        print(f"   {class_name:<12} {train_count:<8} {val_count:<8} {test_count:<8}")
    
    print("=" * 70)
    
    return {
        'X_train': X_train, 'y_train': y_train, 'paths_train': paths_train,
        'X_val': X_val, 'y_val': y_val, 'paths_val': paths_val,
        'X_test': X_test, 'y_test': y_test, 'paths_test': paths_test
    }


# ============================================================================
# SAVE AND LOAD FUNCTIONS
# ============================================================================

def save_processed_data(data_dict, features_info):
    """
    Save processed features and labels
    
    Args:
        data_dict: dictionary with train/val/test splits
        features_info: dictionary with feature extraction parameters
    """
    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save features and labels
    print("\n[SAVING] Saving data files...")
    
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), data_dict['X_train'])
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), data_dict['y_train'])
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), data_dict['X_val'])
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), data_dict['y_val'])
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), data_dict['X_test'])
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), data_dict['y_test'])
    
    # Save image paths
    with open(os.path.join(OUTPUT_DIR, 'image_paths.pkl'), 'wb') as f:
        pickle.dump({
            'train': data_dict['paths_train'],
            'val': data_dict['paths_val'],
            'test': data_dict['paths_test']
        }, f)
    
    # Save feature extraction info
    with open(os.path.join(OUTPUT_DIR, 'features_info.json'), 'w') as f:
        json.dump(features_info, f, indent=2)
    
    print(f"   [OK] X_train.npy")
    print(f"   [OK] y_train.npy")
    print(f"   [OK] X_val.npy")
    print(f"   [OK] y_val.npy")
    print(f"   [OK] X_test.npy")
    print(f"   [OK] y_test.npy")
    print(f"   [OK] image_paths.pkl")
    print(f"   [OK] features_info.json")
    
    print(f"\n[DIR] All files saved to: {OUTPUT_DIR}/")
    print("=" * 70)


def visualize_feature_samples(features, labels, n_samples=5):
    """
    Visualize feature distributions for sample images from each class
    """
    print("\n[VIZ] Creating feature visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (class_name, class_id) in enumerate(CLASS_TO_ID.items()):
        # Get samples from this class
        class_indices = np.where(labels == class_id)[0]
        
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            sample_features = features[sample_idx]
            
            # Plot feature values
            axes[idx].plot(sample_features, linewidth=0.5, alpha=0.7)
            axes[idx].set_title(f'{class_name.capitalize()} - Feature Vector', 
                               fontweight='bold', fontsize=12)
            axes[idx].set_xlabel('Feature Index')
            axes[idx].set_ylabel('Feature Value')
            axes[idx].grid(True, alpha=0.3)
    
    # Hide last empty subplot if necessary
    if len(CLASS_TO_ID) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    output_path = os.path.join('results', 'feature_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   [OK] Saved to '{output_path}'")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for Phase 2
    """
    print("\n" + "=" * 25)
    print("PHASE 2: FEATURE EXTRACTION")
    print("=" * 25 + "\n")
    
    # Step 1: Extract features from all images
    features, labels, image_paths = extract_features_from_dataset(AUGMENTED_DATA_DIR)
    
    if features is None:
        print("[ERROR] Feature extraction failed!")
        return
    
    # Step 2: Split dataset
    data_splits = split_dataset(features, labels, image_paths)
    
    # Step 3: Normalize features (fit on training data only!)
    X_train_normalized, scaler = normalize_features(data_splits['X_train'], save_scaler=True)
    X_val_normalized, _ = normalize_features(data_splits['X_val'], scaler=scaler, save_scaler=False)
    X_test_normalized, _ = normalize_features(data_splits['X_test'], scaler=scaler, save_scaler=False)
    
    # Update data_splits with normalized features
    data_splits['X_train'] = X_train_normalized
    data_splits['X_val'] = X_val_normalized
    data_splits['X_test'] = X_test_normalized
    
    # Step 4: Save processed data
    features_info = {
        'image_size': IMAGE_SIZE,
        'hog_params': {
            'orientations': HOG_ORIENTATIONS,
            'pixels_per_cell': HOG_PIXELS_PER_CELL,
            'cells_per_block': HOG_CELLS_PER_BLOCK
        },
        'color_params': {
            'bins_per_channel': COLOR_BINS
        },
        'lbp_params': {
            'radius': LBP_RADIUS,
            'points': LBP_POINTS,
            'method': LBP_METHOD
        },
        'feature_dimension': features.shape[1],
        'class_names': CLASS_NAMES,
        'class_to_id': CLASS_TO_ID
    }
    
    save_processed_data(data_splits, features_info)
    
    # Step 5: Visualize features
    visualize_feature_samples(X_train_normalized, data_splits['y_train'])
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE!")
    print("=" * 70)
    print("\nSummary:")
    print(f"   Extracted {features.shape[1]}-dimensional features")
    print(f"   HOG + Color Histogram + LBP combined")
    print(f"   Features normalized with StandardScaler")
    print(f"   Dataset split: 70% train, 15% val, 15% test")
    print(f"   All data saved to '{OUTPUT_DIR}/'")
    print(f"   Scaler saved to '{MODELS_DIR}/'")

    print("\nNext Steps:")
    print("   -> Proceed to Phase 3: Model Training (SVM & k-NN)")
    print("   -> Use the saved .npy files for training")

    print("\n" + "=" * 25 + "\n")


if __name__ == "__main__":
    main()
