"""
Phase 3: SVM Training and Optimization
Material Stream Identification System

This script handles:
1. Loading preprocessed features
2. Hyperparameter tuning with GridSearchCV
3. SVM training with optimal parameters
4. Unknown class detection with confidence thresholding
5. Comprehensive evaluation and visualization
"""

import os
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROCESSED_DATA_DIR = 'data/features'
MODELS_DIR = 'saved_models'
RESULTS_DIR = 'results'

# Class names
CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_processed_data():
    """
    Load preprocessed features and labels
    """
    print("\n" + "=" * 70)
    print("LOADING PREPROCESSED DATA")
    print("=" * 70)
    
    try:
        # Load training data
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
        
        # Load validation data
        X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
        
        # Load test data
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
        
        # Load feature info
        with open(os.path.join(PROCESSED_DATA_DIR, 'features_info.json'), 'r') as f:
            features_info = json.load(f)
        
        print(f"\n[OK] Data loaded successfully!")
        print(f"\n[INFO] Dataset Information:")
        print(f"   Training set:   {X_train.shape[0]:>5} samples x {X_train.shape[1]:>5} features")
        print(f"   Validation set: {X_val.shape[0]:>5} samples x {X_val.shape[1]:>5} features")
        print(f"   Test set:       {X_test.shape[0]:>5} samples x {X_test.shape[1]:>5} features")
        
        # Class distribution
        print(f"\n[DISTRIBUTION] Training Set Class Distribution:")
        for class_id, class_name in enumerate(CLASS_NAMES):
            count = np.sum(y_train == class_id)
            percentage = (count / len(y_train)) * 100
            print(f"   {class_name:<12} {count:>5} samples ({percentage:>5.1f}%)")
        
        print("=" * 70)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, features_info
    
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: Could not find preprocessed data files!")
        print(f"   Missing file: {e.filename}")
        print(f"   Please run Phase 2 (feature extraction) first.")
        return None, None, None, None, None, None, None


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_svm_hyperparameters(X_train, y_train, quick_mode=False):
    """
    Find optimal SVM hyperparameters using GridSearchCV
    
    Args:
        X_train: Training features
        y_train: Training labels
        quick_mode: If True, use smaller grid for faster tuning
    
    Returns:
        best_params: Dictionary of best hyperparameters
        grid_search: Fitted GridSearchCV object
    """
    print("\n" + "=" * 70)
    print("SVM HYPERPARAMETER TUNING")
    print("=" * 70)
    
    if quick_mode:
        print("\n[FAST] Quick Mode: Testing limited parameter combinations")
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.001, 0.01],
            'kernel': ['rbf', 'poly']
        }
    else:
        print("\n[FULL] Full Mode: Comprehensive parameter search")
        param_grid = {
            'C': [0.1, 1, 10, 50, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'linear']
        }
    
    print(f"\n[PARAMS] Parameter Grid:")
    for param, values in param_grid.items():
        print(f"   {param:<10} {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n[COUNT] Total combinations to test: {total_combinations}")
    print(f"   With 5-fold CV: {total_combinations * 5} model fits")
    
    # Estimate time
    if quick_mode:
        est_time = "5-10 minutes"
    else:
        est_time = "15-30 minutes"
    print(f"   Estimated time: {est_time}")
    
    # Create SVM with probability estimates
    svm = SVC(probability=True, random_state=42, cache_size=1000)
    
    # GridSearchCV with cross-validation
    print(f"\n[START] Starting grid search...")
    start_time = time.time()
    
    # how to show every training in every parameter in every fold
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n[OK] Grid search complete!")
    print(f"   Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"\n[BEST] Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param:<10} {value}")
    
    print(f"\n[SCORE] Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    # Show top 5 parameter combinations
    print(f"\n[TOP] Top 5 Parameter Combinations:")
    results = grid_search.cv_results_
    indices = np.argsort(results['mean_test_score'])[::-1][:5]
    
    for i, idx in enumerate(indices, 1):
        params = results['params'][idx]
        score = results['mean_test_score'][idx]
        std = results['std_test_score'][idx]
        print(f"   {i}. Score: {score:.4f} (+/- {std:.4f})")
        print(f"      Params: C={params['C']}, gamma={params['gamma']}, kernel={params['kernel']}")
    
    print("=" * 70)
    
    return grid_search.best_params_, grid_search


def analyze_hyperparameter_impact(grid_search):
    """
    Visualize the impact of different hyperparameters
    """
    print("\n[ANALYSIS] Creating hyperparameter impact visualizations...")
    
    results = grid_search.cv_results_
    
    # Extract results for RBF kernel
    rbf_mask = np.array([p['kernel'] == 'rbf' for p in results['params']])
    if rbf_mask.any():
        rbf_results = {
            'C': [],
            'gamma': [],
            'score': []
        }
        
        for i, mask in enumerate(rbf_mask):
            if mask:
                rbf_results['C'].append(results['params'][i]['C'])
                gamma_val = results['params'][i]['gamma']
                # Convert 'scale' and 'auto' to numeric for plotting
                if gamma_val == 'scale':
                    gamma_val = 0.0001  # Placeholder
                elif gamma_val == 'auto':
                    gamma_val = 0.0005  # Placeholder
                rbf_results['gamma'].append(gamma_val)
                rbf_results['score'].append(results['mean_test_score'][i])
        
        # Create heatmap
        if rbf_results['C']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data for heatmap
            C_values = sorted(list(set(rbf_results['C'])))
            gamma_values = sorted(list(set(rbf_results['gamma'])))
            
            heatmap_data = np.zeros((len(gamma_values), len(C_values)))
            
            for c, g, s in zip(rbf_results['C'], rbf_results['gamma'], rbf_results['score']):
                i = gamma_values.index(g)
                j = C_values.index(c)
                heatmap_data[i, j] = s
            
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=C_values, yticklabels=[f'{g:.4f}' for g in gamma_values],
                       cbar_kws={'label': 'CV Accuracy'}, ax=ax)
            
            ax.set_xlabel('C (Regularization)', fontweight='bold')
            ax.set_ylabel('Gamma', fontweight='bold')
            ax.set_title('SVM Hyperparameter Impact (RBF Kernel)', fontweight='bold', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'hyperparameter_heatmap.png'), dpi=150)
            print("   [OK] Saved: hyperparameter_heatmap.png")
            plt.close()


# ============================================================================
# SVM TRAINING
# ============================================================================

def train_final_svm(X_train, y_train, best_params):
    """
    Train final SVM model with best parameters
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL SVM MODEL")
    print("=" * 70)
    
    print(f"\n[CONFIG] Configuration:")
    print(f"   Kernel:  {best_params['kernel']}")
    print(f"   C:       {best_params['C']}")
    print(f"   Gamma:   {best_params['gamma']}")
    print(f"   Probability: True (for confidence scores)")
    
    # Create and train SVM
    print(f"\n[TRAINING] Training SVM on full training set...")
    start_time = time.time()
    
    svm_model = SVC(
        kernel=best_params['kernel'],
        C=best_params['C'],
        gamma=best_params['gamma'],
        probability=True,  # Enable probability estimates
        random_state=42,
        cache_size=1000,
        verbose=False
    )
    
    svm_model.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    
    print(f"[OK] Training complete!")
    print(f"   Time: {elapsed_time:.2f} seconds")
    print(f"   Support vectors: {svm_model.n_support_.sum()}")
    print(f"   Support vectors per class: {svm_model.n_support_}")
    
    print("=" * 70)
    
    return svm_model


# ============================================================================
# UNKNOWN CLASS DETECTION
# ============================================================================

def predict_with_rejection(model, X, threshold=0.6):
    """
    Predict with confidence-based rejection for unknown class
    
    Args:
        model: Trained SVM model
        X: Feature array
        threshold: Confidence threshold (predictions below this -> unknown)
    
    Returns:
        predictions: Array of predicted classes (with unknown=6)
        confidences: Array of confidence scores
    """
    # Get probability predictions
    probabilities = model.predict_proba(X)
    
    # Get max probability and predicted class for each sample
    max_probs = probabilities.max(axis=1)
    predicted_classes = probabilities.argmax(axis=1)
    
    # Apply rejection: if confidence < threshold, classify as unknown (6)
    predictions = predicted_classes.copy()
    predictions[max_probs < threshold] = 6  # Unknown class
    
    return predictions, max_probs


def find_optimal_threshold(model, X_val, y_val):
    """
    Find optimal confidence threshold for unknown class detection
    
    Args:
        model: Trained SVM model
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        optimal_threshold: Best threshold value
        threshold_results: Dictionary with results for different thresholds
    """
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL REJECTION THRESHOLD")
    print("=" * 70)
    
    thresholds = np.arange(0.3, 0.9, 0.05)
    results = {
        'threshold': [],
        'known_accuracy': [],
        'unknown_recall': [],
        'overall_accuracy': [],
        'f1_weighted': []
    }
    
    print(f"\n[SEARCH] Testing {len(thresholds)} different thresholds...")
    
    for threshold in tqdm(thresholds, desc="   Testing thresholds"):
        predictions, confidences = predict_with_rejection(model, X_val, threshold)
        
        # Calculate metrics
        overall_acc = accuracy_score(y_val, predictions)
        
        # Known classes accuracy (excluding unknown)
        known_mask = y_val != 6
        if known_mask.any():
            known_acc = accuracy_score(y_val[known_mask], predictions[known_mask])
        else:
            known_acc = 0.0
        
        # Unknown class recall
        unknown_mask = y_val == 6
        if unknown_mask.any():
            unknown_recall = accuracy_score(y_val[unknown_mask], predictions[unknown_mask])
        else:
            unknown_recall = 0.0
        
        # Weighted F1
        _, _, f1, _ = precision_recall_fscore_support(y_val, predictions, average='weighted', zero_division=0)
        
        results['threshold'].append(threshold)
        results['known_accuracy'].append(known_acc)
        results['unknown_recall'].append(unknown_recall)
        results['overall_accuracy'].append(overall_acc)
        results['f1_weighted'].append(f1)
    
    # Find optimal threshold (maximize overall accuracy)
    optimal_idx = np.argmax(results['overall_accuracy'])
    optimal_threshold = results['threshold'][optimal_idx]
    
    print(f"\n[TARGET] Optimal Threshold: {optimal_threshold:.2f}")
    print(f"\n[PERF] Performance at Optimal Threshold:")
    print(f"   Overall Accuracy:   {results['overall_accuracy'][optimal_idx]:.4f}")
    print(f"   Known Accuracy:     {results['known_accuracy'][optimal_idx]:.4f}")
    print(f"   Unknown Recall:     {results['unknown_recall'][optimal_idx]:.4f}")
    print(f"   Weighted F1:        {results['f1_weighted'][optimal_idx]:.4f}")
    
    # Plot threshold analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(results['threshold'], results['overall_accuracy'], 'b-o', linewidth=2)
    axes[0, 0].axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
    axes[0, 0].set_xlabel('Threshold', fontweight='bold')
    axes[0, 0].set_ylabel('Overall Accuracy', fontweight='bold')
    axes[0, 0].set_title('Overall Accuracy vs Threshold', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(results['threshold'], results['known_accuracy'], 'g-o', linewidth=2)
    axes[0, 1].axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
    axes[0, 1].set_xlabel('Threshold', fontweight='bold')
    axes[0, 1].set_ylabel('Known Classes Accuracy', fontweight='bold')
    axes[0, 1].set_title('Known Classes Accuracy vs Threshold', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].plot(results['threshold'], results['unknown_recall'], 'orange', marker='o', linewidth=2)
    axes[1, 0].axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
    axes[1, 0].set_xlabel('Threshold', fontweight='bold')
    axes[1, 0].set_ylabel('Unknown Class Recall', fontweight='bold')
    axes[1, 0].set_title('Unknown Class Recall vs Threshold', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(results['threshold'], results['f1_weighted'], 'purple', marker='o', linewidth=2)
    axes[1, 1].axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
    axes[1, 1].set_xlabel('Threshold', fontweight='bold')
    axes[1, 1].set_ylabel('Weighted F1 Score', fontweight='bold')
    axes[1, 1].set_title('Weighted F1 Score vs Threshold', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'threshold_analysis.png'), dpi=150)
    print(f"\n[SAVED] Saved threshold analysis to: results/threshold_analysis.png")
    plt.close()
    
    print("=" * 70)
    
    return optimal_threshold, results


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X, y, threshold, dataset_name="Test"):
    """
    Comprehensive model evaluation
    """
    print(f"\n" + "=" * 70)
    print(f"{dataset_name.upper()} SET EVALUATION")
    print("=" * 70)
    
    # Make predictions
    predictions, confidences = predict_with_rejection(model, X, threshold)
    
    # Overall metrics
    accuracy = accuracy_score(y, predictions)
    
    print(f"\n[RESULTS] Overall Performance:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print(f"\n[REPORT] Detailed Classification Report:")
    print(classification_report(y, predictions, target_names=CLASS_NAMES, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontweight='bold', fontsize=12)
    plt.ylabel('True', fontweight='bold', fontsize=12)
    plt.title(f'Confusion Matrix - {dataset_name} Set\nAccuracy: {accuracy:.4f}', 
              fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{dataset_name.lower()}.png'), dpi=150)
    print(f"\n[SAVED] Saved confusion matrix to: results/confusion_matrix_{dataset_name.lower()}.png")
    plt.close()
    
    # Confidence distribution
    plt.figure(figsize=(12, 6))
    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = y == class_id
        if mask.any():
            plt.hist(confidences[mask], bins=30, alpha=0.5, label=class_name)
    
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    plt.xlabel('Confidence Score', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(f'Confidence Distribution - {dataset_name} Set', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confidence_distribution_{dataset_name.lower()}.png'), dpi=150)
    print(f"[SAVED] Saved confidence distribution to: results/confidence_distribution_{dataset_name.lower()}.png")
    plt.close()
    
    print("=" * 70)
    
    return accuracy, predictions, confidences


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_svm_model(model, threshold, best_params, results_summary):
    """
    Save trained SVM model and metadata
    """
    print("\n" + "=" * 70)
    print("SAVING SVM MODEL")
    print("=" * 70)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n[SAVED] Model saved to: {model_path}")
    
    # Save model configuration
    config = {
        'best_params': best_params,
        'optimal_threshold': threshold,
        'results': results_summary,
        'class_names': CLASS_NAMES
    }
    
    config_path = os.path.join(MODELS_DIR, 'svm_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[SAVED] Configuration saved to: {config_path}")
    
    print("=" * 70)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(quick_mode=False):
    """
    Main training pipeline
    
    Args:
        quick_mode: If True, use faster but less thorough hyperparameter tuning
    """
    print("\n" + "[ML] " * 25)
    print("PHASE 3: SVM TRAINING & OPTIMIZATION")
    print("[ML] " * 25 + "\n")
    
    # Step 1: Load data
    X_train, y_train, X_val, y_val, X_test, y_test, features_info = load_processed_data()
    
    if X_train is None:
        return
    
    # Step 2: Hyperparameter tuning
    # print("\n" + "[TARGET] " * 35)
    # best_params, grid_search = tune_svm_hyperparameters(X_train, y_train, quick_mode=quick_mode)
    # analyze_hyperparameter_impact(grid_search)
    best_params = {
        "C": 10,
        "gamma": "auto",
        "kernel": "rbf"
    }
    
    # Step 3: Train final model
    svm_model = train_final_svm(X_train, y_train, best_params)
    
    # Step 4: Find optimal threshold
    optimal_threshold, threshold_results = find_optimal_threshold(svm_model, X_val, y_val)
    
    # Step 5: Evaluate on validation set
    val_acc, val_preds, val_confs = evaluate_model(
        svm_model, X_val, y_val, optimal_threshold, "Validation"
    )
    
    # Step 6: Evaluate on test set
    test_acc, test_preds, test_confs = evaluate_model(
        svm_model, X_test, y_test, optimal_threshold, "Test"
    )
    
    # Step 7: Evaluate on training set (to check overfitting)
    train_acc, _, _ = evaluate_model(
        svm_model, X_train, y_train, optimal_threshold, "Training"
    )
    
    # Step 8: Save model
    results_summary = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'feature_dimension': X_train.shape[1],
        'training_samples': len(X_train)
    }
    
    save_svm_model(svm_model, optimal_threshold, best_params, results_summary)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE!")
    print("=" * 70)
    print(f"\n[RESULTS] Final Results:")
    print(f"   Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    if train_acc - test_acc > 0.1:
        print(f"\n[WARNING] Warning: Possible overfitting detected!")
        print(f"   Train-Test gap: {(train_acc - test_acc)*100:.2f}%")
    else:
        print(f"\n[OK] Good generalization!")
        print(f"   Train-Test gap: {(train_acc - test_acc)*100:.2f}%")
    
    if test_acc >= 0.85:
        print(f"\n[SUCCESS] SUCCESS! Test accuracy >= 85% target!")
    else:
        print(f"\n[WARNING] Test accuracy below 85% target.")
        print(f"   Consider: More data augmentation, feature tuning, or ensemble methods")
    
    print(f"\n[DIR] All results saved to: {RESULTS_DIR}/")
    print(f"[DIR] Model saved to: {MODELS_DIR}/")
    
    print(f"\n[NEXT] Next Steps:")
    print("   1. Review confusion matrix to identify problem classes")
    print("   2. Analyze misclassified examples")
    print("   3. Proceed to Phase 4: Real-time Deployment")
    
    print("\n" + "[ML] " * 25 + "\n")


if __name__ == "__main__":
    # Set quick_mode=True for faster testing (5-10 min)
    # Set quick_mode=False for best results (15-30 min)
    main(quick_mode=False)
