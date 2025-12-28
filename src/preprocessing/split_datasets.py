import shutil
from sklearn.model_selection import train_test_split
import os
from config import ORIGINAL_DATA_DIR, TRAIN_DIR, VAL_DIR, CLASS_NAMES
def split_dataset(original_dir, train_dir, val_dir, split_ratio=0.8, seed=42):
    """
    Split dataset into train and validation sets BEFORE augmentation
    """
    print("\n" + "=" * 70)
    print("SPLITTING DATASET INTO TRAIN / VALIDATION")
    print("=" * 70)

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_path = os.path.join(original_dir, class_name)
        if not os.path.exists(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        train_imgs, val_imgs = train_test_split(
            images, train_size=split_ratio, random_state=seed, shuffle=True
        )

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(train_dir, class_name, img)
            )

        for img in val_imgs:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(val_dir, class_name, img)
            )

        print(f"[SPLIT] {class_name}: {len(train_imgs)} train | {len(val_imgs)} val")

    print("[OK] Dataset split completed\n")

if __name__ == "__main__":

    split_dataset(ORIGINAL_DATA_DIR, TRAIN_DIR, VAL_DIR, split_ratio=0.8, seed=42)