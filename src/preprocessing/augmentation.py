"""
Phase 1: Data Preparation & Augmentation
Material Stream Identification System

This script handles:
1. Dataset exploration and analysis
2. Data augmentation (30% minimum increase)
3. Class balancing (~500 images per class)
4. Unknown class generation (class 6)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
ORIGINAL_DATA_DIR = 'dataset'  # Your original dataset folder
AUGMENTED_DATA_DIR = 'data/augmented'  # Output folder
TARGET_IMAGES_PER_CLASS = 500  # Target number of images per class

# Class names (0-5 are provided, 6 will be generated)
CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

# Augmentation parameters
ROTATION_RANGE = 30  # degrees
BRIGHTNESS_RANGE = (0.7, 1.3)  # 70% to 130%
ZOOM_RANGE = (0.8, 1.2)  # 80% to 120%
FLIP_PROBABILITY = 0.5

# ============================================================================
# STEP 1: DATASET EXPLORATION
# ============================================================================

def explore_dataset(data_dir):
    """
    Analyze the original dataset and print statistics
    """
    print("=" * 70)
    print("DATASET EXPLORATION")
    print("=" * 70)
    
    class_stats = {}
    total_images = 0
    
    for class_name in CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_path):
            print(f"WARNING: Folder '{class_name}' not found!")
            continue
        
        # Count images
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        count = len(images)
        total_images += count
        
        class_stats[class_name] = {
            'count': count,
            'images': images
        }
        
        # Sample image dimensions
        if count > 0:
            sample_img_path = os.path.join(class_path, images[0])
            sample_img = cv2.imread(sample_img_path)
            if sample_img is not None:
                height, width = sample_img.shape[:2]
                class_stats[class_name]['sample_size'] = (width, height)
    
    # Print statistics
    print(f"[STATS] Dataset Statistics:")
    print(f"{'Class':<15} {'Count':<10} {'Percentage':<12} {'Sample Size'}")
    print("-" * 70)
    
    for class_name in CLASS_NAMES:
        if class_name in class_stats:
            count = class_stats[class_name]['count']
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            sample_size = class_stats[class_name].get('sample_size', 'N/A')
            print(f"{class_name:<15} {count:<10} {percentage:>6.2f}%      {sample_size}")
    
    print("-" * 70)
    print(f"{'TOTAL':<15} {total_images:<10} 100.00%")
    print()
    
    # Identify imbalances
    if class_stats:
        counts = [stats['count'] for stats in class_stats.values()]
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"[ANALYSIS] Class Imbalance Analysis:")
        print(f"   Max class size: {max_count}")
        print(f"   Min class size: {min_count}")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 2:
            print(f"   Warning: Significant imbalance detected! Augmentation needed.")
        else:
            print(f"   Classes are relatively balanced.")
    
    print("\n" + "=" * 70 + "\n")
    
    return class_stats, total_images


# ============================================================================
# STEP 2: AUGMENTATION FUNCTIONS
# ============================================================================

def rotate_image(image, angle):
    """Rotate image by given angle"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height), 
                             borderMode=cv2.BORDER_REFLECT)
    return rotated


def adjust_brightness(image, factor):
    """Adjust image brightness"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def zoom_image(image, zoom_factor):
    """Zoom in/out on image"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    # Resize
    resized = cv2.resize(image, (new_width, new_height))
    
    # Crop or pad to original size
    if zoom_factor > 1:  # Zoom in - crop center
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        result = resized[start_y:start_y+height, start_x:start_x+width]
    else:  # Zoom out - pad
        result = np.zeros((height, width, 3), dtype=np.uint8)
        start_y = (height - new_height) // 2
        start_x = (width - new_width) // 2
        result[start_y:start_y+new_height, start_x:start_x+new_width] = resized
    
    return result


def flip_image(image, flip_code):
    """Flip image horizontally (1) or vertically (0)"""
    return cv2.flip(image, flip_code)


def translate_image(image, tx, ty):
    """Translate image by tx, ty pixels"""
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, matrix, (width, height),
                                borderMode=cv2.BORDER_REFLECT)
    return translated


def augment_image(image):
    """
    Apply random augmentation to a single image
    Returns the augmented image
    """
    img = image.copy()
    
    # Random rotation
    if random.random() > 0.3:
        angle = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        img = rotate_image(img, angle)
    
    # Random brightness
    if random.random() > 0.3:
        factor = random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        img = adjust_brightness(img, factor)
    
    # Random zoom
    if random.random() > 0.3:
        zoom = random.uniform(ZOOM_RANGE[0], ZOOM_RANGE[1])
        img = zoom_image(img, zoom)
    
    # Random horizontal flip
    if random.random() > 0.5:
        img = flip_image(img, 1)
    
    # Random translation
    if random.random() > 0.4:
        height, width = img.shape[:2]
        tx = random.randint(-int(width*0.1), int(width*0.1))
        ty = random.randint(-int(height*0.1), int(height*0.1))
        img = translate_image(img, tx, ty)
    
    return img


# ============================================================================
# STEP 3: AUGMENTATION PIPELINE
# ============================================================================

def augment_class(class_name, original_images, original_dir, output_dir, target_count):
    """
    Augment a single class to reach target_count images
    """
    current_count = len(original_images)
    needed = target_count - current_count
    
    print(f"[PROCESSING] Processing class: {class_name}")
    print(f"   Original: {current_count} images")
    print(f"   Target: {target_count} images")
    print(f"   Need to generate: {needed} images")
    
    # Create output directory
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Copy original images
    print(f"   - Copying original images...")
    for img_name in original_images:
        src = os.path.join(original_dir, class_name, img_name)
        dst = os.path.join(class_output_dir, img_name)
        shutil.copy2(src, dst)
    
    # Generate augmented images
    if needed > 0:
        print(f"   - Generating {needed} augmented images...")
        
        aug_count = 0
        with tqdm(total=needed, desc=f"   Augmenting {class_name}") as pbar:
            while aug_count < needed:
                # Randomly select an original image
                original_img_name = random.choice(original_images)
                original_img_path = os.path.join(original_dir, class_name, original_img_name)
                
                # Read image
                img = cv2.imread(original_img_path)
                if img is None:
                    continue
                
                # Augment
                aug_img = augment_image(img)
                
                # Save with unique name
                base_name = os.path.splitext(original_img_name)[0]
                ext = os.path.splitext(original_img_name)[1]
                aug_img_name = f"{base_name}_aug_{aug_count}{ext}"
                aug_img_path = os.path.join(class_output_dir, aug_img_name)
                
                cv2.imwrite(aug_img_path, aug_img)
                
                aug_count += 1
                pbar.update(1)
    
    final_count = len(os.listdir(class_output_dir))
    print(f"   [OK] Final count: {final_count} images")
    
    return final_count


# ============================================================================
# STEP 4: UNKNOWN CLASS GENERATION
# ============================================================================

def generate_unknown_class(augmented_dir, target_count=400):
    """
    Generate Unknown class (ID: 6) from:
    1. Heavily blurred existing images
    2. Mixed/ambiguous samples
    """
    print("\n" + "=" * 70)
    print("GENERATING UNKNOWN CLASS (Class 6)")
    print("=" * 70)
    
    unknown_dir = os.path.join(augmented_dir, 'unknown')
    os.makedirs(unknown_dir, exist_ok=True)
    
    all_images = []
    
    # Collect all existing images
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(augmented_dir, class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            all_images.extend(images)
    
    print(f"\n[STATS] Generating {target_count} unknown samples...")
    print(f"   Strategy: Heavy blur + random transformations")
    
    with tqdm(total=target_count, desc="   Creating unknown samples") as pbar:
        for i in range(target_count):
            # Randomly select an image
            img_path = random.choice(all_images)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Apply heavy blur (this makes it "unknown")
            blur_amount = random.randint(15, 35)
            if blur_amount % 2 == 0:
                blur_amount += 1
            img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
            
            # Optionally add more distortions
            if random.random() > 0.5:
                # Add noise
                noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
            
            if random.random() > 0.5:
                # Extreme brightness change
                factor = random.choice([0.3, 0.4, 1.7, 1.8])
                img = adjust_brightness(img, factor)
            
            # Save
            unknown_img_name = f"unknown_{i:04d}.jpg"
            unknown_img_path = os.path.join(unknown_dir, unknown_img_name)
            cv2.imwrite(unknown_img_path, img)
            
            pbar.update(1)
    
    final_count = len(os.listdir(unknown_dir))
    print(f"\n   [OK] Generated {final_count} unknown samples")
    print("=" * 70 + "\n")
    
    return final_count


# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

def visualize_samples(augmented_dir):
    """
    Display sample images from each class
    """
    print("\n[INFO] Visualizing samples from each class...")
    
    all_classes = CLASS_NAMES + ['unknown']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(all_classes):
        class_dir = os.path.join(augmented_dir, class_name)
        
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if images:
                # Select a random image
                sample_img_name = random.choice(images)
                sample_img_path = os.path.join(class_dir, sample_img_name)
                img = cv2.imread(sample_img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[idx].imshow(img)
                axes[idx].set_title(f"{class_name.capitalize()} ({len(images)} images)")
                axes[idx].axis('off')
    
    # Hide last empty subplot
    if len(all_classes) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("   [OK] Saved visualization to 'dataset_samples.png'")
    plt.show()


def plot_class_distribution(augmented_dir):
    """
    Plot bar chart of class distribution
    """
    print("\n[CHART] Creating class distribution chart...")
    
    all_classes = CLASS_NAMES + ['unknown']
    counts = []
    
    for class_name in all_classes:
        class_dir = os.path.join(augmented_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            counts.append(count)
        else:
            counts.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(all_classes)), counts, color='steelblue', alpha=0.8)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title('Augmented Dataset Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(all_classes)), 
               [c.capitalize() for c in all_classes], 
               rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    print("   [OK] Saved chart to 'class_distribution.png'")
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\n" + "=" * 25)
    print("PHASE 1: DATA PREPARATION & AUGMENTATION")
    print("=" * 25 + "\n")
    
    # Step 1: Explore original dataset
    class_stats, total_original = explore_dataset(ORIGINAL_DATA_DIR)
    
    if not class_stats:
        print("[ERROR] Error: No valid dataset found!")
        return
    
    # Step 2: Create output directory
    os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)
    
    # Step 3: Augment each class
    print("\n" + "=" * 70)
    print("AUGMENTATION PIPELINE")
    print("=" * 70)
    
    augmented_stats = {}
    
    for class_name in CLASS_NAMES:
        if class_name in class_stats:
            original_images = class_stats[class_name]['images']
            final_count = augment_class(
                class_name, 
                original_images,
                ORIGINAL_DATA_DIR,
                AUGMENTED_DATA_DIR,
                TARGET_IMAGES_PER_CLASS
            )
            augmented_stats[class_name] = final_count
    
    # Step 4: Generate Unknown class
    unknown_count = generate_unknown_class(AUGMENTED_DATA_DIR, target_count=400)
    augmented_stats['unknown'] = unknown_count
    
    # Step 5: Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    total_augmented = sum(augmented_stats.values())
    increase_percentage = ((total_augmented - total_original) / total_original) * 100
    
    print(f"\n[STATS] Final Dataset Statistics:")
    print(f"{'Class':<15} {'Original':<12} {'Augmented':<12} {'Increase'}")
    print("-" * 70)
    
    for class_name in CLASS_NAMES:
        original = class_stats[class_name]['count']
        augmented = augmented_stats.get(class_name, 0)
        increase = augmented - original
        print(f"{class_name:<15} {original:<12} {augmented:<12} +{increase}")
    
    print(f"{'unknown':<15} {0:<12} {unknown_count:<12} +{unknown_count}")
    print("-" * 70)
    print(f"{'TOTAL':<15} {total_original:<12} {total_augmented:<12} "
          f"+{total_augmented - total_original} ({increase_percentage:.1f}%)")
    
    if increase_percentage >= 30:
        print(f"\n[SUCCESS] Dataset increased by {increase_percentage:.1f}% (target: 30%)")
    else:
        print(f"\n[WARNING] Dataset only increased by {increase_percentage:.1f}% "
              f"(target: 30%)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
