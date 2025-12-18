# Phase 1: Data Augmentation & Preparation

## Overview

**Phase 1** handles initial data preparation and augmentation. This critical phase ensures that the machine learning pipeline has a well-balanced, diverse dataset for training robust models.

## Why Data Augmentation?

Real-world datasets often have **class imbalances** - some material types may have more images than others. Without augmentation:
- Models can become biased toward frequently occurring classes
- Rare classes may be under-learned
- Overall system performance degrades

Data augmentation solves this by:
1. Creating synthetic variations of existing images
2. Balancing the number of samples per class
3. Improving model generalization to unseen variations

---

## File Location

```
src/preprocessing/augmentation.py
```

## Configuration

The augmentation script is controlled by parameters at the top of the file:

```python
# Paths
ORIGINAL_DATA_DIR = 'dataset'        # Input folder with original images
AUGMENTED_DATA_DIR = 'data/augmented' # Output folder for augmented data
TARGET_IMAGES_PER_CLASS = 500        # Target samples per class

# Class names (indices 0-5 are provided, 6 is generated)
CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

# Augmentation intensity parameters
ROTATION_RANGE = 30           # Degrees (±30°)
BRIGHTNESS_RANGE = (0.7, 1.3) # 70% to 130%
ZOOM_RANGE = (0.8, 1.2)       # 80% to 120%
FLIP_PROBABILITY = 0.5        # 50% chance of flip
```

### Key Settings Explained

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `TARGET_IMAGES_PER_CLASS` | 500 | 100-1000 | How many images to generate per class |
| `ROTATION_RANGE` | 30 | 0-45 | Maximum rotation angle in degrees |
| `BRIGHTNESS_RANGE` | (0.7, 1.3) | (0.5, 1.5) | Brightness variation factor |
| `ZOOM_RANGE` | (0.8, 1.2) | (0.5, 2.0) | Zoom in/out factor |
| `FLIP_PROBABILITY` | 0.5 | 0.0-1.0 | Probability of horizontal flip |

---

## How It Works

### Step 1: Dataset Exploration

When you run the script, it first analyzes the original dataset:

```
========================================================================
DATASET EXPLORATION
========================================================================

[STATS] Dataset Statistics:
Class            Count      Percentage    Sample Size
------------------------------------------------------------------------
glass               150       20.00%        (480, 640)
paper               180       24.00%        (480, 640)
cardboard           140       18.67%        (480, 640)
plastic             200       26.67%        (480, 640)
metal                80       10.67%        (480, 640)
trash                50        6.67%        (480, 640)
------------------------------------------------------------------------
TOTAL               800      100.00%

[ANALYSIS] Class Imbalance Analysis:
   Max class size: 200
   Min class size: 50
   Imbalance ratio: 4.00x
   Warning: Significant imbalance detected! Augmentation needed.
```

This output shows:
- **Count**: Number of original images per class
- **Percentage**: Distribution of classes
- **Sample Size**: Image dimensions
- **Imbalance Ratio**: Maximum/minimum class size (higher = more imbalanced)

### Step 2: Augmentation Techniques

The script applies the following augmentation operations **randomly** to each image:

#### A. **Rotation**
- Rotates image by random angle within `ROTATION_RANGE`
- **Purpose**: Makes model robust to different angles/orientations
- **Example**: A glass bottle can be photographed at different angles

```python
rotate_image(image, angle=-15)  # Rotate 15 degrees counterclockwise
rotate_image(image, angle=25)   # Rotate 25 degrees clockwise
```

#### B. **Brightness Adjustment**
- Multiplies image brightness by factor in `BRIGHTNESS_RANGE`
- **Purpose**: Handles different lighting conditions (indoor/outdoor, shadows)
- **Example**: 0.7 × brightness = darker image, 1.3 × brightness = lighter image

```python
adjust_brightness(image, factor=0.85)  # Darken image by 15%
adjust_brightness(image, factor=1.15)  # Brighten image by 15%
```

#### C. **Zoom**
- Zooms in (crops center) or out (pads borders) by random factor
- **Purpose**: Simulates different distances from camera
- **Example**: Close-up vs. far-away view of same object

```python
zoom_image(image, zoom_factor=1.1)  # Zoom in 10%
zoom_image(image, zoom_factor=0.9)  # Zoom out 10%
```

#### D. **Flipping**
- Randomly flips image horizontally
- **Purpose**: Creates mirror images (many objects look similar when flipped)
- **Probability**: Controlled by `FLIP_PROBABILITY`

```python
flip_image(image, flip_code=1)  # Horizontal flip
flip_image(image, flip_code=0)  # Vertical flip (rarely used)
```

#### E. **Translation**
- Shifts image by random pixels in X/Y directions
- **Purpose**: Handles different object positions in frame
- **Range**: ±10% of image dimensions

```python
translate_image(image, tx=20, ty=-15)  # Shift right 20px, up 15px
```

### Step 3: Per-Class Augmentation

For each class, the script:

1. **Copies** all original images to output folder
2. **Generates** additional synthetic images until reaching `TARGET_IMAGES_PER_CLASS`
3. **Applies** random augmentations to selected originals
4. **Saves** augmented images with `_aug_` prefix

**Example Process for 'glass' class:**
```
Original: 150 images
Target: 500 images
Needed: 350 augmented images

Progress: ████████████████████████░░░░░░░░░░░░░░░░░ 87% | 350/350 generated
```

### Step 4: Unknown Class Generation

The "Unknown" class (class 6) is synthetically generated from augmented images:

```python
def generate_unknown_class(augmented_dir, target_count=400):
    """
    Generate Unknown class from:
    1. Heavily blurred existing images
    2. Mixed/ambiguous samples
    """
```

This class includes:
- Heavily blurred/degraded images (hard to classify)
- Mixed materials (pieces of multiple classes)
- Low-quality captures

**Purpose**: Teaches the model to reject ambiguous inputs instead of forcing wrong classifications

---

## Output Structure

After running `augmentation.py`, the `data/augmented/` folder contains:

```
data/augmented/
├── glass/              # ~500 images
│   ├── glass_001.jpg
│   ├── glass_002.jpg
│   ├── glass_001_aug_0.jpg
│   ├── glass_001_aug_1.jpg
│   └── ...
├── paper/              # ~500 images
├── cardboard/          # ~500 images
├── plastic/            # ~500 images
├── metal/              # ~500 images
├── trash/              # ~500 images
└── unknown/            # ~400 images
```

**Total**: ~3,400 images (balanced across 7 classes)

---

## Running Phase 1

### Basic Usage

```bash
# From the project root directory
python src/preprocessing/augmentation.py
```

### Expected Output

```
============================================================================
DATASET EXPLORATION
============================================================================

[STATS] Dataset Statistics:
glass              150       20.00%      (480, 640)
paper              180       24.00%      (480, 640)
...

[ANALYSIS] Class Imbalance Analysis:
   Max class size: 200
   Min class size: 50
   Imbalance ratio: 4.00x
   Warning: Significant imbalance detected!

============================================================================
AUGMENTATION PIPELINE
============================================================================

[PROCESSING] Processing class: glass
   Original: 150 images
   Target: 500 images
   Need to generate: 350 images
   - Copying original images...
   - Generating 350 augmented images...
   Augmenting glass ████████████████████████████████████████ 100% | 350/350
   [OK] Final count: 500 images

[PROCESSING] Processing class: paper
...

============================================================================
GENERATING UNKNOWN CLASS (Class 6)
============================================================================

Generating Unknown class from mixed/blurred samples...
[OK] Generated 400 unknown class images

[SUMMARY] Augmentation Complete!
   Original images: 800
   Augmented images: 2,600
   Total images: 3,400
   Average per class: 486
```

### Execution Time

- **Full dataset**: 5-15 minutes
- Depends on:
  - Number of original images
  - Computer speed (CPU/disk I/O)
  - `TARGET_IMAGES_PER_CLASS` setting

---

## Customizing Augmentation

### For Small Datasets (< 100 images per class)

```python
ROTATION_RANGE = 45           # Increase variation
BRIGHTNESS_RANGE = (0.5, 1.5) # Larger variation
ZOOM_RANGE = (0.6, 1.4)       # More zoom variation
TARGET_IMAGES_PER_CLASS = 800  # Generate more samples
```

### For Already Balanced Datasets

```python
TARGET_IMAGES_PER_CLASS = 300  # Less generation needed
```

### For More Conservative Augmentation

```python
ROTATION_RANGE = 15           # Smaller angles
BRIGHTNESS_RANGE = (0.85, 1.15) # Subtle brightness
ZOOM_RANGE = (0.9, 1.1)       # Minimal zoom
```

---

## Understanding the Results

### What Good Augmentation Looks Like

After augmentation, your dataset should be:

1. **Balanced**: Each class has roughly same number of images
   ```
   glass:      500 images ✓
   paper:      500 images ✓
   cardboard:  500 images ✓
   plastic:    500 images ✓
   metal:      500 images ✓
   trash:      500 images ✓
   unknown:    400 images ✓
   ```

2. **Diverse**: Augmented images vary in appearance
   - Different angles (rotations)
   - Different lighting (brightness)
   - Different scales (zoom)
   - Different positions (translation)

3. **Realistic**: Augmentations look plausible, not distorted
   - Objects still recognizable
   - Variations match real-world conditions

### What to Avoid

❌ **Over-augmentation**: Images become unrecognizable
- Increase `ROTATION_RANGE` beyond 45°
- Decrease `BRIGHTNESS_RANGE` below (0.5, 1.5)
- Extreme zoom values

❌ **Under-augmentation**: Classes still imbalanced
- Too small `TARGET_IMAGES_PER_CLASS`
- Need to increase augmentation intensity

❌ **Unrealistic variations**: Break causal relationships
- Don't augment in ways that wouldn't occur in practice
- If all images are captured horizontally, don't flip vertically

---

## Troubleshooting

### Problem: "No images found in dataset folder"

**Cause**: Incorrect `ORIGINAL_DATA_DIR` path or folder structure

**Solution**:
```python
# Verify folder structure:
# dataset/
# ├── glass/
# ├── paper/
# ├── cardboard/
# ├── plastic/
# ├── metal/
# └── trash/

# Check ORIGINAL_DATA_DIR setting
ORIGINAL_DATA_DIR = 'dataset'  # Must exist in project root
```

### Problem: "ValueError: invalid literal for int()"

**Cause**: Non-image files in class folders (e.g., `.txt`, `.json`)

**Solution**:
```python
# The script filters by extension:
images = [f for f in os.listdir(class_path) 
          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# Remove non-image files manually or place images in separate folder
```

### Problem: Augmentation too slow

**Cause**: Large images or too high `TARGET_IMAGES_PER_CLASS`

**Solutions**:
1. Reduce `TARGET_IMAGES_PER_CLASS` to 300-400
2. Pre-resize images to 128×128 before running
3. Use SSD (faster than HDD) for image storage
4. Close other applications to free RAM

---

## Key Concepts

### **Class Imbalance**
When classes have unequal samples. Example: 200 glass images vs. 50 trash images.

**Effect**: Model becomes biased toward majority class (glass).

**Solution**: Augment minority classes (trash) until balanced.

### **Augmentation**
Creating synthetic variations of existing data without collecting new samples.

**Benefit**: More training data = better generalization.

**Trade-off**: Synthetic data ≠ real data, but helps with limited datasets.

### **Train/Validation/Test Split**
Data is later divided into three sets:
- **Train** (70%): Used to learn patterns
- **Validation** (15%): Used to tune hyperparameters
- **Test** (15%): Used to measure final performance

---

## Next Steps

After Phase 1 completes:

1. **Verify output**: Check that `data/augmented/` exists and contains balanced images
2. **Move to Phase 2**: Run feature extraction
   ```bash
   python src/preprocessing/feature_extractor.py
   ```
3. **Analyze features**: Visualize what features look like
   ```bash
   python src/pipeline/feature_analysis.py
   ```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Input** | Raw images in `dataset/` folder |
| **Output** | Augmented images in `data/augmented/` |
| **Duration** | 5-15 minutes |
| **Transformations** | Rotation, brightness, zoom, flip, translation |
| **Purpose** | Balance classes + improve generalization |
| **Classes** | 6 original + 1 generated unknown = 7 total |
| **Target Size** | ~500 images per class |

**Key Takeaway**: Phase 1 ensures your dataset is balanced and diverse, giving the ML models the best chance to learn robust patterns.

---

**Next Document**: [Phase 2: Feature Extraction](03_PHASE_2_FEATURES.md)
