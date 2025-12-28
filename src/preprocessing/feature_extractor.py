"""
Phase 2 :CNN Feature Extractor

This file is a feature-extraction pipeline with a
compact PyTorch CNN training script that uses a pretrained ResNet50
to learn features (and save weights). The implementation is organized
into configuration, helpers, training/validation loops, and a main
pipeline.
"""

import os
import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# CONFIGURATION
from config import AUGMENTED_DATA_DIR, MODELS_DIR, VAL_DIR, MODEL_DIR, MODEL_FILENAME
# Training configuration
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-5
patience = 3
epochs_without_improvement = 0


FEATURES_DIR = os.environ.get('FEATURES_DIR', '/content/drive/MyDrive/CNN//features')

# TRANSFORMS & DATASET HELPERS

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def is_valid_image(path: str) -> bool:
    """Quickly check an image file for basic integrity using PIL.

    Returns True when the file opens and verifies; False otherwise.
    Prints a short message for skipped files.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        print("Skipping corrupted file:", path)
        return False


def get_device():
    """Return torch.device: CUDA if available and not forced off, else CPU.

    Honor env var `FORCE_CPU=1` to force CPU even when CUDA is available.
    """
    force_cpu = os.environ.get('FORCE_CPU', '') in ('1', 'true', 'True')
    if not force_cpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def prepare_datasets(dataset_path: str):
    """Load ImageFolder, filter invalid files, and split into train/val.

    Returns train_loader, val_loader and number of classes.
    """
    train_dataset = datasets.ImageFolder(root=AUGMENTED_DATA_DIR, transform=transform, is_valid_file=is_valid_image)
    val_dataset   = datasets.ImageFolder(root=VAL_DIR, transform=transform, is_valid_file=is_valid_image)
    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)
    print(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, num_classes


def extract_and_cache_features(dataset_path: str = AUGMENTED_DATA_DIR, weights_path: str = None, overwrite: bool = False):
    """Extract pooled CNN features using ResNet backbone and cache to .npy files.

    Saves: X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy
    under `FEATURES_DIR` (default 'data/features'). Splits dataset using TRAIN_RATIO.
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # If files exist and not overwriting, skip
    expected = [os.path.join(FEATURES_DIR, n) for n in (
        'X_train.npy','X_val.npy','X_test.npy','y_train.npy','y_val.npy','y_test.npy')]
    if not overwrite and all(os.path.exists(p) for p in expected):
        print(f"Feature files already exist in {FEATURES_DIR}, use overwrite=True to regenerate.")
        return

    # Build dataset
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform, is_valid_file=is_valid_image)
    if len(full_dataset) == 0:
        raise RuntimeError(f"No images found in {dataset_path}")

    device = get_device()
    print('Using device for feature extraction:', device)

    # Load model and weights if available
    try:
        model = models.resnet50(pretrained=False)
    except Exception:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Try to load provided weights or default saved model
    if weights_path is None:
        weights_path = os.path.join(MODELS_DIR, MODEL_FILENAME)

    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded CNN weights from {weights_path}")
        except Exception:
            # partial load: update matching shapes
            sd = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in sd and sd[k].shape == v.shape}
            sd.update(filtered)
            model.load_state_dict(sd)
            print("Loaded subset of weights (partial load)")
    else:
        try:
            model = models.resnet50(pretrained=True)
            print('No saved weights found; using ImageNet pretrained backbone')
        except Exception:
            print('Warning: using randomly initialized ResNet (no pretrained weights)')

    model = model.to(device)
    model.eval()

    # Feature extractor: everything except the final fc layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()


    def _compute(ds):

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        feats = []
        labels = []
        with torch.no_grad():
            for imgs, labs in tqdm(loader, desc='Extracting features'):
                imgs = imgs.to(device)
                out = feature_extractor(imgs)
                out = out.reshape(out.size(0), -1).cpu().numpy()
                feats.append(out)
                labels.append(labs.numpy())
        if feats:
            return np.vstack(feats), np.concatenate(labels)
        return np.zeros((0,0)), np.array([])

    train_ds = datasets.ImageFolder(root=AUGMENTED_DATA_DIR, transform=transform, is_valid_file=is_valid_image)
    val_ds = datasets.ImageFolder(root=VAL_DIR, transform=transform, is_valid_file=is_valid_image)
    X_train, y_train = _compute(train_ds)
    X_val, y_val = _compute(val_ds)

    # Normalize features (fit on training set) and save scaler
    if X_train.size == 0:
        print("No features extracted; skipping save.")
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val.size else X_val

    # Ensure model dir exists and save scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save scaled features and labels
    np.save(os.path.join(FEATURES_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(FEATURES_DIR, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(FEATURES_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(FEATURES_DIR, 'y_val.npy'), y_val)

    print(f"Saved scaled features to {FEATURES_DIR}: X_train={X_train_scaled.shape}, X_val={X_val_scaled.shape}")
    print(f"Scaler saved to: {scaler_path}")


# MODEL, LOSS, OPTIMIZER

def build_model(num_classes: int, device: torch.device):
    """Create a pretrained ResNet50 and replace the final head.

    The function returns the model moved to the given device.
    """
    try:
        model = models.resnet50(pretrained=True)
    except Exception:
        # Fallback for torchvision versions where 'pretrained' is deprecated
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    #FREEZE ENTIRE BACKBONE
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head with the number of classes we have
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
    )

    #  Unfreeze classifier head
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)


# TRAIN / VALIDATION LOOPS


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num, print_every=10):
    """Train for one epoch and return avg loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch_num+1}")):
        try:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % print_every == 0 or (i + 1) == len(loader):
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                progress = (i + 1) / len(loader) * 100
                avg_loss = running_loss / (i + 1)
                avg_acc = correct / total if total > 0 else 0
                batch_acc = (preds == labels).sum().item() / len(labels)
                gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
                print(f"[{timestamp}] Epoch {epoch_num+1} | Batch {i+1}/{len(loader)} ({progress:.1f}%) | "
                      f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | Batch Acc: {batch_acc:.4f} | "
                      f"Avg Acc: {avg_acc:.4f} | GPU Mem: {gpu_mem:.1f}MB", flush=True)

        except Exception as e:
            print(f"Skipping batch {i+1} due to error: {e}", flush=True)
            continue

    return running_loss / len(loader), correct / total if total > 0 else 0


def validate(model, loader, device):
    """Evaluate model on validation loader and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 20 == 0:
                print(f"[Validation] Batch {i+1}/{len(loader)} | Batch Acc: {(preds==labels).sum().item()/len(labels):.4f} | Total Acc So Far: {correct/total:.4f}", flush=True)

    return correct / total if total > 0 else 0


# SAVE / MAIN PIPELINE


def save_model_state(model, model_dir: str, filename: str):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    torch.save(model.state_dict(), path)
    print(f"CNN weights saved to: {path}")


def main(dataset_path: str = AUGMENTED_DATA_DIR):
    print("\n" + "=" * 25)
    print("PHASE 2: CNN Feature Extractor")
    print("=" * 25 + "\n")

    train_loader, val_loader, num_classes = prepare_datasets(AUGMENTED_DATA_DIR)

    if len(train_loader.dataset) == 0 and len(val_loader.dataset) == 0:
        print("No images found. Check DATASET_PATH.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model(num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
    )


    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, print_every=10)
        val_acc = validate(model, val_loader, device)

        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n", flush=True)

        if val_acc > best_val_acc:
          best_val_acc = val_acc
          epochs_without_improvement = 0
          save_model_state(model, MODELS_DIR, MODEL_FILENAME)
        else:
          epochs_without_improvement += 1
          if epochs_without_improvement >= patience:
              print(f"No improvement for {patience} epochs. Stopping early.")
              break

    print("\nTraining complete.")
    extract_and_cache_features(dataset_path=AUGMENTED_DATA_DIR, weights_path=os.path.join(MODELS_DIR, MODEL_FILENAME))


main()
