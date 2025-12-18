import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.preprocessing.feature_extractor import IMAGE_SIZE

class FeatureExtractor:
    def __init__(self):
        # Resolve project paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..', '..')

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------
        # Load trained CNN backbone
        # -------------------------------
        model_path = os.path.join(project_root, 'saved_models', 'cnn_feature_extractor.pth')

        model = models.resnet50(pretrained=False)
        model.fc = nn.Identity()  # remove classifier

        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state, strict=False)

        self.model = model.to(self.device)
        self.model.eval()

        # -------------------------------
        # Load feature scaler
        # -------------------------------
        scaler_path = os.path.join(project_root, 'saved_models', 'feature_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # -------------------------------
        # Image preprocessing
        # -------------------------------
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract a 2048-dim CNN feature vector from a single frame."""

        if frame is None or frame.size == 0:
            return np.zeros(2048)

        if frame.ndim != 3 or frame.shape[2] != 3:
            return np.zeros(2048)

        # Ensure uint8
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Preprocess
        img = self.transform(frame).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(img)

        features = features.cpu().numpy().reshape(1, -1)

        # Normalize with training scaler
        features = self.scaler.transform(features)

        return features.flatten()
