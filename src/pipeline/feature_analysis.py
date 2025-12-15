"""
Feature Analysis Helper
Visualize and understand HOG, Color Histogram, and LBP features

Run this to see what each feature descriptor actually captures!
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

AUGMENTED_DATA_DIR = 'data/augmented'
IMAGE_SIZE = (128, 128)

# Feature parameters (same as main script)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_BINS = 32
LBP_RADIUS = 1
LBP_POINTS = 8

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_single_image_features(image_path):
    """
    Visualize HOG, Color Histogram, and LBP for a single image
    """
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return
    
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # ========================================================================
    # 1. ORIGINAL IMAGE
    # ========================================================================
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # ========================================================================
    # 2. HOG FEATURES
    # ========================================================================
    # Extract HOG with visualization
    hog_features, hog_image = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    
    # Rescale HOG image for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    # Plot HOG visualization
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(hog_image_rescaled, cmap='hot')
    ax2.set_title(f'HOG Visualization\n({len(hog_features)} features)', 
                  fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    # Plot HOG feature vector
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(hog_features[:200], linewidth=0.5)  # Show first 200 features
    ax3.set_title('HOG Feature Vector (first 200)', fontweight='bold', fontsize=10)
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    
    # Plot HOG histogram
    ax4 = plt.subplot(3, 4, 4)
    ax4.hist(hog_features, bins=50, color='red', alpha=0.7)
    ax4.set_title('HOG Feature Distribution', fontweight='bold', fontsize=10)
    ax4.set_xlabel('Feature Value')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # 3. COLOR HISTOGRAM FEATURES
    # ========================================================================
    # Split into RGB channels
    b, g, r = cv2.split(img)
    
    # Compute histograms
    hist_r = cv2.calcHist([r], [0], None, [COLOR_BINS], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [COLOR_BINS], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [COLOR_BINS], [0, 256])
    
    # Normalize
    hist_r = hist_r / (hist_r.sum() + 1e-7)
    hist_g = hist_g / (hist_g.sum() + 1e-7)
    hist_b = hist_b / (hist_b.sum() + 1e-7)
    
    # Plot individual channels
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(r, cmap='Reds')
    ax5.set_title('Red Channel', fontweight='bold', fontsize=12)
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(g, cmap='Greens')
    ax6.set_title('Green Channel', fontweight='bold', fontsize=12)
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.imshow(b, cmap='Blues')
    ax7.set_title('Blue Channel', fontweight='bold', fontsize=12)
    ax7.axis('off')
    
    # Plot combined histogram
    ax8 = plt.subplot(3, 4, 8)
    x_axis = np.arange(COLOR_BINS)
    ax8.plot(x_axis, hist_r, color='red', label='Red', linewidth=2)
    ax8.plot(x_axis, hist_g, color='green', label='Green', linewidth=2)
    ax8.plot(x_axis, hist_b, color='blue', label='Blue', linewidth=2)
    ax8.set_title(f'Color Histograms\n({COLOR_BINS*3} features)', 
                  fontweight='bold', fontsize=12)
    ax8.set_xlabel('Bin')
    ax8.set_ylabel('Normalized Frequency')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # ========================================================================
    # 4. LBP FEATURES
    # ========================================================================
    # Compute LBP
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    
    # Compute histogram
    n_bins = LBP_POINTS * (LBP_POINTS - 1) + 3
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
    
    # Plot grayscale image
    ax9 = plt.subplot(3, 4, 9)
    ax9.imshow(gray, cmap='gray')
    ax9.set_title('Grayscale Image', fontweight='bold', fontsize=12)
    ax9.axis('off')
    
    # Plot LBP image
    ax10 = plt.subplot(3, 4, 10)
    ax10.imshow(lbp, cmap='gray')
    ax10.set_title(f'LBP Pattern\n(P={LBP_POINTS}, R={LBP_RADIUS})', 
                   fontweight='bold', fontsize=12)
    ax10.axis('off')
    
    # Plot LBP histogram
    ax11 = plt.subplot(3, 4, 11)
    ax11.bar(range(len(lbp_hist)), lbp_hist, color='purple', alpha=0.7)
    ax11.set_title(f'LBP Histogram\n({len(lbp_hist)} features)', 
                   fontweight='bold', fontsize=12)
    ax11.set_xlabel('LBP Pattern')
    ax11.set_ylabel('Frequency')
    ax11.grid(True, alpha=0.3)
    
    # Plot texture intensity
    ax12 = plt.subplot(3, 4, 12)
    im = ax12.imshow(lbp, cmap='viridis')
    ax12.set_title('LBP Texture Map\n(Color-coded)', fontweight='bold', fontsize=12)
    ax12.axis('off')
    plt.colorbar(im, ax=ax12, fraction=0.046)
    
    # ========================================================================
    # FINAL FORMATTING
    # ========================================================================
    plt.suptitle(f'Feature Extraction Breakdown: {os.path.basename(image_path)}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    os.makedirs('results', exist_ok=True)
    output_name = f"feature_breakdown_{os.path.basename(image_path)}"
    output_path = os.path.join('results', output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved visualization to: {output_path}")
    
    plt.show()


def compare_features_across_classes():
    """
    Compare features across different material classes
    """
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    
    fig, axes = plt.subplots(len(class_names), 4, figsize=(16, 3*len(class_names)))
    
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(AUGMENTED_DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            continue
        
        # Get first image from class
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            continue
        
        img_path = os.path.join(class_dir, images[0])
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        img = cv2.resize(img, IMAGE_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features
        hog_features, hog_image = hog(gray, orientations=HOG_ORIENTATIONS,
                                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                                       cells_per_block=HOG_CELLS_PER_BLOCK,
                                       visualize=True)
        
        b, g, r = cv2.split(img)
        hist_r = cv2.calcHist([r], [0], None, [COLOR_BINS], [0, 256]).flatten()
        hist_g = cv2.calcHist([g], [0], None, [COLOR_BINS], [0, 256]).flatten()
        hist_b = cv2.calcHist([b], [0], None, [COLOR_BINS], [0, 256]).flatten()
        
        lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
        n_bins = LBP_POINTS * (LBP_POINTS - 1) + 3
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Plot
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f'{class_name.upper()}', fontweight='bold')
        axes[idx, 0].axis('off')
        
        hog_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        axes[idx, 1].imshow(hog_rescaled, cmap='hot')
        axes[idx, 1].set_title('HOG')
        axes[idx, 1].axis('off')
        
        x = np.arange(COLOR_BINS)
        axes[idx, 2].plot(x, hist_r/hist_r.sum(), 'r-', linewidth=2, alpha=0.7)
        axes[idx, 2].plot(x, hist_g/hist_g.sum(), 'g-', linewidth=2, alpha=0.7)
        axes[idx, 2].plot(x, hist_b/hist_b.sum(), 'b-', linewidth=2, alpha=0.7)
        axes[idx, 2].set_title('Color Histogram')
        axes[idx, 2].grid(True, alpha=0.3)
        
        axes[idx, 3].bar(range(len(lbp_hist)), lbp_hist/lbp_hist.sum(), 
                         color='purple', alpha=0.7)
        axes[idx, 3].set_title('LBP Histogram')
        axes[idx, 3].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Comparison Across Material Classes', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    output_path = os.path.join('results', 'feature_comparison_classes.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved comparison to: {output_path}")
    plt.show()


def analyze_feature_dimensions():
    """
    Print detailed breakdown of feature dimensions
    """
    print("\n" + "=" * 70)
    print("FEATURE DIMENSION ANALYSIS")
    print("=" * 70)
    
    # Create a sample image
    sample_img = np.random.randint(0, 255, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    
    # HOG
    hog_feat = hog(gray, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   feature_vector=True)
    
    # Color
    color_feat = COLOR_BINS * 3  # RGB
    
    # LBP
    lbp_feat = LBP_POINTS * (LBP_POINTS - 1) + 3
    
    total = len(hog_feat) + color_feat + lbp_feat
    
    print(f"\n[FEATURES] Feature Breakdown:")
    print(f"\n1. HOG Features:")
    print(f"   - Orientations: {HOG_ORIENTATIONS}")
    print(f"   - Pixels per cell: {HOG_PIXELS_PER_CELL}")
    print(f"   - Cells per block: {HOG_CELLS_PER_BLOCK}")
    print(f"   - Image size: {IMAGE_SIZE}")
    print(f"   - Total HOG features: {len(hog_feat)}")
    print(f"   - Purpose: Captures edge directions and shapes")
    
    print(f"\n2. Color Histogram Features:")
    print(f"   - Bins per channel: {COLOR_BINS}")
    print(f"   - Channels: 3 (R, G, B)")
    print(f"   - Total color features: {color_feat}")
    print(f"   - Purpose: Captures color distribution patterns")
    
    print(f"\n3. LBP Features:")
    print(f"   - Radius: {LBP_RADIUS}")
    print(f"   - Points: {LBP_POINTS}")
    print(f"   - Method: uniform")
    print(f"   - Total LBP features: {lbp_feat}")
    print(f"   - Purpose: Captures texture information")
    
    print(f"\n{'=' * 70}")
    print(f"TOTAL COMBINED FEATURES: {total} dimensions")
    print(f"{'=' * 70}")
    
    print(f"\n[TIP] Why this combination works:")
    print(f"   [OK] HOG captures SHAPE (edges, contours)")
    print(f"   [OK] Color Histogram captures COLOR (material appearance)")
    print(f"   [OK] LBP captures TEXTURE (surface patterns)")
    print(f"   => Together they describe materials comprehensively!")
    
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Interactive feature analysis
    """
    print("\n" + "[FEATURE] " * 25)
    print("FEATURE ANALYSIS HELPER")
    print("[FEATURE] " * 25 + "\n")
    
    # Analyze dimensions
    analyze_feature_dimensions()
    
    # Find a sample image
    print("[SEARCH] Finding sample images for visualization...\n")
    
    sample_images = []
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    
    for class_name in class_names:
        class_dir = os.path.join(AUGMENTED_DATA_DIR, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir)[:1]
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                img_path = os.path.join(class_dir, images[0])
                sample_images.append((class_name, img_path))
    
    if sample_images:
        print(f"Found {len(sample_images)} sample images.\n")
        print("Options:")
        print("  1. Visualize detailed features for ONE image")
        print("  2. Compare features across ALL classes")
        print("  3. Both (recommended)\n")
        
        choice = input("Enter choice (1/2/3) [default: 3]: ").strip() or "3"
        
        if choice in ["1", "3"]:
            print(f"\n[VIZ] Analyzing detailed features for: {sample_images[0][0]}")
            visualize_single_image_features(sample_images[0][1])
        
        if choice in ["2", "3"]:
            print(f"\n[ANALYSIS] Comparing features across classes...")
            compare_features_across_classes()
        
        print("\n[SUCCESS] Analysis complete! Check the generated images.")
    else:
        print("[ERROR] No sample images found. Make sure to run Phase 1 first!")
    
    print("\n" + "[ANALYSIS] " * 25 + "\n")


if __name__ == "__main__":
    main()
