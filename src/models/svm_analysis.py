"""
SVM Performance Analysis & Improvement Helper

Use this script to:
1. Analyze misclassified examples
2. Understand confusion patterns
3. Get suggestions for improvement
"""

import os
import numpy as np
import pickle
import json
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DATA_DIR = 'data/features'
MODELS_DIR = 'saved_models'
RESULTS_DIR = 'results'
AUGMENTED_DATA_DIR = 'data/augmented'

CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

def load_model_and_data():
    """Load trained model, data, and configuration"""
    print("[LOAD] Loading model and data...")
    
    # Load model
    with open(os.path.join(MODELS_DIR, 'svm_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load configuration
    with open(os.path.join(MODELS_DIR, 'svm_config.json'), 'r') as f:
        config = json.load(f)
    
    # Load test data
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    
    # Load image paths
    with open(os.path.join(PROCESSED_DATA_DIR, 'image_paths.pkl'), 'rb') as f:
        paths_dict = pickle.load(f)
    
    print("[OK] Loaded successfully!\n")
    
    return model, config, X_test, y_test, paths_dict['test']


# ============================================================================
# ANALYZE MISCLASSIFICATIONS
# ============================================================================

def analyze_misclassifications(model, X_test, y_test, test_paths, threshold):
    """
    Analyze which samples are being misclassified
    """
    print("=" * 70)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 70)
    
    # Get predictions
    probabilities = model.predict_proba(X_test)
    max_probs = probabilities.max(axis=1)
    predicted_classes = probabilities.argmax(axis=1)
    
    # Apply rejection threshold
    predictions = predicted_classes.copy()
    predictions[max_probs < threshold] = 6
    
    # Find misclassified samples
    misclassified_mask = predictions != y_test
    misclassified_indices = np.where(misclassified_mask)[0]
    
    num_misclassified = len(misclassified_indices)
    total_samples = len(y_test)
    error_rate = (num_misclassified / total_samples) * 100
    
    print(f"\n[STATS] Overall Statistics:")
    print(f"   Total test samples: {total_samples}")
    print(f"   Misclassified: {num_misclassified}")
    print(f"   Error rate: {error_rate:.2f}%")
    print(f"   Accuracy: {100 - error_rate:.2f}%")
    
    # Analyze confusion patterns
    print(f"\n[ANALYSIS] Most Common Confusion Patterns:")
    confusion_pairs = {}
    
    for idx in misclassified_indices:
        true_class = y_test[idx]
        pred_class = predictions[idx]
        pair = (CLASS_NAMES[true_class], CLASS_NAMES[pred_class])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    for i, ((true_name, pred_name), count) in enumerate(sorted_pairs[:10], 1):
        percentage = (count / num_misclassified) * 100
        print(f"   {i:2d}. {true_name:>10} -> {pred_name:<10} : {count:>3} times ({percentage:>5.1f}%)")
    
    # Confidence analysis of misclassifications
    misclassified_confidences = max_probs[misclassified_mask]
    correctly_classified_confidences = max_probs[~misclassified_mask]
    
    print(f"\n[ANALYSIS] Confidence Analysis:")
    print(f"   Misclassified samples:")
    print(f"      Mean confidence: {misclassified_confidences.mean():.4f}")
    print(f"      Median confidence: {np.median(misclassified_confidences):.4f}")
    print(f"      Min confidence: {misclassified_confidences.min():.4f}")
    print(f"      Max confidence: {misclassified_confidences.max():.4f}")
    
    print(f"\n   Correctly classified samples:")
    print(f"      Mean confidence: {correctly_classified_confidences.mean():.4f}")
    print(f"      Median confidence: {np.median(correctly_classified_confidences):.4f}")
    
    return misclassified_indices, predictions, max_probs


def visualize_misclassified_examples(misclassified_indices, y_test, predictions, 
                                     max_probs, test_paths, num_examples=12):
    """
    Visualize misclassified examples
    """
    print(f"\n[VIZ] Creating visualization of {num_examples} misclassified examples...")
    
    # Select random misclassified samples
    if len(misclassified_indices) > num_examples:
        selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
    else:
        selected_indices = misclassified_indices[:num_examples]
    
    # Create figure
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_indices):
        if i >= len(axes):
            break
        
        # Load image
        img_path = test_paths[idx]
        img = cv2.imread(img_path)
        
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            true_class = y_test[idx]
            pred_class = predictions[idx]
            confidence = max_probs[idx]
            
            # Plot
            axes[i].imshow(img_rgb)
            axes[i].set_title(
                f"True: {CLASS_NAMES[true_class]}\n"
                f"Pred: {CLASS_NAMES[pred_class]}\n"
                f"Conf: {confidence:.2f}",
                fontsize=10,
                color='red',
                fontweight='bold'
            )
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(selected_indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'misclassified_examples.png'), dpi=150)
    print(f"   [OK] Saved to: results/misclassified_examples.png")
    plt.close()


# ============================================================================
# PER-CLASS ANALYSIS
# ============================================================================

def analyze_per_class_performance(model, X_test, y_test, threshold):
    """
    Analyze performance for each class individually
    """
    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Get predictions
    probabilities = model.predict_proba(X_test)
    max_probs = probabilities.max(axis=1)
    predicted_classes = probabilities.argmax(axis=1)
    predictions = predicted_classes.copy()
    predictions[max_probs < threshold] = 6
    
    print(f"\n{'Class':<12} {'Total':<8} {'Correct':<8} {'Wrong':<8} {'Accuracy':<10} {'Avg Conf':<10}")
    print("-" * 70)
    
    class_stats = []
    
    for class_id, class_name in enumerate(CLASS_NAMES):
        # Get samples from this class
        class_mask = y_test == class_id
        class_samples = class_mask.sum()
        
        if class_samples == 0:
            continue
        
        # Calculate metrics
        class_predictions = predictions[class_mask]
        class_confidences = max_probs[class_mask]
        
        correct = (class_predictions == class_id).sum()
        wrong = class_samples - correct
        accuracy = correct / class_samples
        avg_conf = class_confidences.mean()
        
        class_stats.append({
            'name': class_name,
            'accuracy': accuracy,
            'avg_conf': avg_conf,
            'samples': class_samples
        })
        
        print(f"{class_name:<12} {class_samples:<8} {correct:<8} {wrong:<8} "
              f"{accuracy*100:<9.2f}% {avg_conf:<10.4f}")
    
    print("-" * 70)
    
    # Identify problem classes
    print(f"\n[WARNING] Problem Classes (accuracy < 80%):")
    problem_classes = [c for c in class_stats if c['accuracy'] < 0.80]
    
    if problem_classes:
        for cls in problem_classes:
            print(f"   - {cls['name']:<12} {cls['accuracy']*100:.2f}% "
                  f"(avg confidence: {cls['avg_conf']:.4f})")
    else:
        print(f"   [OK] All classes performing well (>=80% accuracy)!")
    
    # Identify best classes
    print(f"\n[OK] Best Performing Classes (accuracy >= 90%):")
    best_classes = [c for c in class_stats if c['accuracy'] >= 0.90]
    
    for cls in best_classes:
        print(f"   - {cls['name']:<12} {cls['accuracy']*100:.2f}% "
              f"(avg confidence: {cls['avg_conf']:.4f})")


# ============================================================================
# IMPROVEMENT SUGGESTIONS
# ============================================================================

def suggest_improvements(model, X_test, y_test, predictions, threshold):
    """
    Provide suggestions for improving model performance
    """
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 70)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    suggestions = []
    
    # 1. Check overall accuracy
    accuracy = (y_test == predictions).mean()
    
    if accuracy < 0.85:
        suggestions.append({
            'priority': 'HIGH',
            'issue': f'Overall accuracy is {accuracy*100:.2f}% (target: 85%+)',
            'solutions': [
                'Collect more training data for underperforming classes',
                'Apply more aggressive data augmentation',
                'Try feature selection to remove noisy features',
                'Experiment with different kernel functions',
                'Consider ensemble methods (combining SVM with other classifiers)'
            ]
        })
    
    # 2. Check for class imbalance issues
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (predictions[class_mask] == i).mean()
            
            if class_acc < 0.70:
                suggestions.append({
                    'priority': 'HIGH',
                    'issue': f'Poor performance on {class_name} class ({class_acc*100:.2f}%)',
                    'solutions': [
                        f'Increase training samples for {class_name} (current augmentation may be insufficient)',
                        f'Review {class_name} images for quality issues or mislabeling',
                        f'Add more diverse augmentation specifically for {class_name}',
                        f'Collect real-world {class_name} samples with more variety'
                    ]
                })
    
    # 3. Check for specific confusion patterns
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 5:  # More than 5 confusions
                confusion_rate = cm[i, j] / (cm[i].sum() + 1e-7)
                if confusion_rate > 0.15:  # 15% confusion rate
                    suggestions.append({
                        'priority': 'MEDIUM',
                        'issue': f'{CLASS_NAMES[i]} frequently confused with {CLASS_NAMES[j]} '
                                f'({cm[i, j]} times, {confusion_rate*100:.1f}%)',
                        'solutions': [
                            f'Add more distinctive features that separate {CLASS_NAMES[i]} and {CLASS_NAMES[j]}',
                            f'Review training images of both classes for mislabeling',
                            f'Add augmentations that highlight differences between these classes',
                            f'Consider using class-specific feature weights'
                        ]
                    })
    
    # 4. Check threshold effectiveness
    probabilities = model.predict_proba(X_test)
    max_probs = probabilities.max(axis=1)
    low_conf_mask = max_probs < threshold
    low_conf_correct = ((predictions == y_test) & low_conf_mask).sum()
    low_conf_total = low_conf_mask.sum()
    
    if low_conf_total > 0:
        low_conf_acc = low_conf_correct / low_conf_total
        if low_conf_acc < 0.5:
            suggestions.append({
                'priority': 'MEDIUM',
                'issue': f'Low-confidence predictions are often wrong ({low_conf_acc*100:.2f}% accuracy)',
                'solutions': [
                    'Consider increasing the rejection threshold',
                    'Model may need more training data to be confident',
                    'Some features may be noisy - try feature selection'
                ]
            })
    
    # Print suggestions
    if not suggestions:
        print("\n[SUCCESS] Great! No major issues detected.")
        print("   Your model is performing well!")
    else:
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        suggestions.sort(key=lambda x: priority_order[x['priority']])
        
        for i, suggestion in enumerate(suggestions, 1):
            priority_color = '[HIGH]' if suggestion['priority'] == 'HIGH' else '[MED]'
            print(f"\n{priority_color} Suggestion #{i} [{suggestion['priority']} PRIORITY]")
            print(f"   Issue: {suggestion['issue']}")
            print(f"   Possible solutions:")
            for sol in suggestion['solutions']:
                print(f"      - {sol}")
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Run complete performance analysis
    """
    print("\n" + "[ANALYSIS] " * 25)
    print("SVM PERFORMANCE ANALYSIS")
    print("[ANALYSIS] " * 25 + "\n")
    
    try:
        # Load model and data
        model, config, X_test, y_test, test_paths = load_model_and_data()
        threshold = config['optimal_threshold']
        
        print(f"[CONFIG] Model Configuration:")
        print(f"   Kernel: {config['best_params']['kernel']}")
        print(f"   C: {config['best_params']['C']}")
        print(f"   Gamma: {config['best_params']['gamma']}")
        print(f"   Threshold: {threshold:.2f}")
        print(f"   Test Accuracy: {config['results']['test_accuracy']*100:.2f}%")
        print()
        
        # Analyze misclassifications
        misclassified_indices, predictions, max_probs = analyze_misclassifications(
            model, X_test, y_test, test_paths, threshold
        )
        
        # Visualize misclassified examples
        if len(misclassified_indices) > 0:
            visualize_misclassified_examples(
                misclassified_indices, y_test, predictions, 
                max_probs, test_paths, num_examples=12
            )
        
        # Per-class analysis
        analyze_per_class_performance(model, X_test, y_test, threshold)
        
        # Improvement suggestions
        suggest_improvements(model, X_test, y_test, predictions, threshold)
        
        print("\n[OK] Analysis complete!")
        print(f"[DIR] Check the '{RESULTS_DIR}' folder for visualizations.")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: Could not find required files!")
        print(f"   Make sure you've run Phase 3 (SVM training) first.")
        print(f"   Missing: {e}")
    
    print("\n" + "[ANALYSIS] " * 25 + "\n")


if __name__ == "__main__":
    main()
