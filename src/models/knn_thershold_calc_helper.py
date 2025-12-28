import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import FEATURES_DIR, MODELS_DIR

# CONFIGURATION
KNN_MODEL_PATH = os.path.join(MODELS_DIR, 'knn_model.pkl')

def calculate_threshold():
    print("=" * 60)
    print("KNN DISTANCE THRESHOLD CALCULATOR")
    print("=" * 60)

    # 1. Load Data and Model
    print(f"[INFO] Loading data from {FEATURES_DIR}...")
    try:
        X_train = np.load(os.path.join(FEATURES_DIR, 'X_train.npy'))
        X_val = np.load(os.path.join(FEATURES_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(FEATURES_DIR, 'y_val.npy'))

        print(f"[INFO] Loading KNN model from {KNN_MODEL_PATH}...")
        with open(KNN_MODEL_PATH, 'rb') as f:
            knn = pickle.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Please ensure you have run the training phases first.")
        return

    # 2. Calculate Distances for Validation Data
    # We use validation data because it wasn't used to build the tree,
    # so it simulates 'new' data better than training data.
    print("[INFO] Calculating distances for validation set...")

    # kneighbors returns (distances, indices)
    # distances shape: (num_samples, n_neighbors)
    distances, _ = knn.kneighbors(X_val)

    # We care about the average distance to the k neighbors
    mean_distances = distances.mean(axis=1)

    # 3. Analyze the Statistics
    min_dist = np.min(mean_distances)
    max_dist = np.max(mean_distances)
    avg_dist = np.mean(mean_distances)
    std_dist = np.std(mean_distances)

    # Calculate percentiles (safe zones)
    p95 = np.percentile(mean_distances, 95)
    p99 = np.percentile(mean_distances, 99)

    print("\n" + "-" * 40)
    print("DISTANCE STATISTICS (KNOWN CLASSES)")
    print("-" * 40)
    print(f"Min Distance:     {min_dist:.4f}")
    print(f"Average Distance: {avg_dist:.4f}")
    print(f"Max Distance:     {max_dist:.4f}")
    print(f"Std Deviation:    {std_dist:.4f}")
    print("-" * 40)
    print(f"95% of data is below: {p95:.4f}")
    print(f"99% of data is below: {p99:.4f}")
    print("-" * 40)

    # 4. Recommendation Logic
    # A safe threshold is usually Mean + 3*StdDev or the 99th percentile.
    # We want to accept 99% of valid images, and reject anything further out.
    recommended_threshold = p99 * 1.1  # Add 10% buffer

    print(f"\n[RECOMMENDATION] Suggested Threshold: {recommended_threshold:.2f}")
    print(f"(Use this value for 'dist_threshold' in your predict function)")

    # 5. Visual Histogram
    # This chart helps you verify if the threshold makes sense visually
    plt.figure(figsize=(10, 6))
    plt.hist(mean_distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95th Percentile ({p95:.2f})')
    plt.axvline(recommended_threshold, color='red', linestyle='--', linewidth=2, label=f'Recommended ({recommended_threshold:.2f})')

    plt.title('KNN Distance Distribution (Validation Set)')
    plt.xlabel('Average Distance to Neighbors')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(MODELS_DIR, 'knn_distance_distribution.png')
    plt.savefig(save_path)
    print(f"\n[GRAPH] Saved distribution plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    calculate_threshold()