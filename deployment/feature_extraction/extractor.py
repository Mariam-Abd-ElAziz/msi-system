import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path

# Define constant here or import from config
IMAGE_SIZE = 128

class FeatureExtractor:
    def __init__(self):
        # Resolve project paths using Pathlib
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent.parent # adjust .parent depth based on your folder structure
        
        # Paths to artifacts
        self.model_path = self.project_root / 'saved_models' / 'cnn_feature_extractor.pth'
        self.scaler_path = self.project_root / 'saved_models' / 'feature_scaler.pkl'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Extractor] Using device: {self.device}")

        # 1. Load CNN Backbone
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        model = models.resnet50(pretrained=False)
        model.fc = nn.Identity()
        
        state = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state, strict=False)
        
        self.model = model.to(self.device)
        self.model.eval() # CRITICAL for inference

        # 2. Load Scaler
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
            
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # 3. Define Transform (Must match training!)
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
        if frame is None or frame.size == 0:
            return np.zeros(2048)

        # Preprocessing
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            features = self.model(img_tensor)

        features = features.cpu().numpy().reshape(1, -1)

        # Scaling
        features = self.scaler.transform(features)

        return features.flatten()