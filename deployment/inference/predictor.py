import sys
import os
import numpy as np
from pathlib import Path

# Adjust path to find src if needed
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.unified_predictor import UnifiedPredictor

CLASS_MAPPING = {
    0: 'cardboard', 1: 'glass', 2: 'metal', 
    3: 'paper', 4: 'plastic', 5: 'trash', 6: 'unknown'
}

class Predictor:
    def __init__(self, model_type='svm', threshold=0.6):
        """
        Args:
            model_type: 'svm' or 'knn'
            threshold: Confidence cutoff (probability for SVM, distance for KNN)
        """
        self.model_type = model_type.lower()
        self.threshold = threshold

        # Paths
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        saved_models_path = project_root / 'saved_models'
        
        print(f"[Predictor] Loading models from {saved_models_path}")
        print(f"[Predictor] Configured for {self.model_type.upper()} with threshold {self.threshold}")
        
        self.predictor = UnifiedPredictor(model_dir=str(saved_models_path))
        
    def predict(self, features):
        try:
            # 1. Get Probabilities / Distances depending on internal logic
            # Note: UnifiedPredictor usually returns probabilities. 
            # If you haven't updated UnifiedPredictor to handle KNN distance logic, 
            # this wrapper might need more logic like we did in the notebook.
            
            probs = self.predictor.predict_probability(features, model_name=self.model_type)
            
            # 2. Prediction Logic
            class_id = probs.argmax()
            confidence = probs.max()

            # --- DEBUG PRINT ---
            # This will show you exactly what the model "thinks"
            print(f"[DEBUG] Raw Class: {class_id} | Raw Conf: {confidence:.4f} | Threshold: {self.threshold}")
            # -------------------

            # 3. REJECTION MECHANISM
            # Logic: If confidence is LOWER than threshold, it is unknown
            # (Note: For KNN distance, logic is reversed: Higher distance = Unknown. 
            # Ensure UnifiedPredictor handles this or handle it here).
            
            is_unknown = False

            if self.model_type == 'svm':
                # Probability Logic: Low probability = Unknown
                if confidence < self.threshold:
                    is_unknown = True
            
            elif self.model_type == 'knn':
                # If your UnifiedPredictor returns probas for KNN, use same logic.
                # If it returns distances, you need to check if confidence > threshold
                # Assuming standard probability for now based on your previous code:
                if confidence < self.threshold:
                    is_unknown = True

            if is_unknown:
                class_id = 6 # Unknown
                # print(f"Rejected: {confidence:.2f}")

            # 4. Map to text
            class_name = CLASS_MAPPING.get(class_id, 'unknown')
            
            return class_name, confidence
            
        except Exception as e:
            print(f"[Predictor] Error: {e}")
            return 'error', 0.0