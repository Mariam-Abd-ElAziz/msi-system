import json
import pickle
import numpy as np
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score

class UnifiedPredictor:
    """
    Unified predictor for SVM, KNN, and ensemble methods.
    Handles loading models, making predictions, and combining results.
    """
    
    def __init__(self, model_dir="saved_models"):
        # Resolve to absolute path to prevent "File not found" errors
        self.model_dir = Path(model_dir).resolve()
        self.models = {}
        self.configs = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load both SVM and KNN models and their configs"""
        for model_name in ['svm', 'knn']:
            try:
                # Load Model
                model_path = self.model_dir / f"{model_name}_model.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        # Suppress scikit-learn version warnings if necessary
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.models[model_name] = pickle.load(f)
                else:
                    print(f"[WARNING] {model_name.upper()} model file not found at {model_path}")
                    continue

                # Load Config
                config_path = self.model_dir / f"{model_name}_config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        self.configs[model_name] = json.load(f)
                
                print(f"[OK] Loaded {model_name.upper()}")

            except Exception as e:
                print(f"[ERROR] Failed to load {model_name}: {e}")
    
    def _ensure_2d(self, features):
        """Helper to ensure features are (1, N) shape"""
        if isinstance(features, list):
            features = np.array(features)
        if len(features.shape) == 1:
            return features.reshape(1, -1)
        return features

    def predict_single(self, features, model_name='svm'):
        """
        Predict class ID using a specific model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available: {list(self.models.keys())}")
        
        features = self._ensure_2d(features)
        return int(self.models[model_name].predict(features)[0])
    
    def predict_probability(self, features, model_name='svm'):
        """
        Get prediction probabilities [p_0, p_1, ..., p_k].
        Works for both KNN and SVM (if trained with probability=True).
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        features = self._ensure_2d(features)
        
        if hasattr(self.models[model_name], 'predict_proba'):
            # Return the first row (for single sample)
            return self.models[model_name].predict_proba(features)[0]
        else:
            raise ValueError(f"{model_name} does not support probability prediction")
    
    def predict_ensemble(self, features, method='soft_voting'):
        """
        Ensemble prediction using multiple models.
        
        Args:
            features: Input feature vector
            method: 
                'soft_voting' (Recommended) - Averages probabilities.
                'hard_voting' - Majority vote of class labels.
        """
        features = self._ensure_2d(features)
        
        if not self.models:
            return {'ensemble_prediction': 6, 'error': 'No models loaded'}

        if method == 'soft_voting':
            # AVERAGE PROBABILITIES (Best for SVM + KNN)
            all_probs = []
            individual_preds = {}
            
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features)[0]
                    all_probs.append(prob)
                    individual_preds[name] = int(prob.argmax())
            
            if not all_probs:
                return self.predict_ensemble(features, method='hard_voting')

            # Calculate average probability vector
            avg_probs = np.mean(all_probs, axis=0)
            ensemble_pred = int(avg_probs.argmax())
            confidence = float(avg_probs.max())

            return {
                'ensemble_prediction': ensemble_pred,
                'confidence': confidence,
                'method': 'soft_voting',
                'individual_predictions': individual_preds
            }

        elif method == 'hard_voting':
            # MAJORITY VOTE (Backup)
            votes = []
            individual_preds = {}
            for name, model in self.models.items():
                pred = int(model.predict(features)[0])
                votes.append(pred)
                individual_preds[name] = pred
            
            # Find most frequent
            ensemble_pred = max(set(votes), key=votes.count)
            
            return {
                'ensemble_prediction': ensemble_pred,
                'method': 'hard_voting',
                'individual_predictions': individual_preds
            }
            
        else:
            raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    # Test script to verify it works
    print("--- Testing UnifiedPredictor ---")
    
    # 1. Initialize
    # Note: Ensure you have a 'saved_models' folder in the same directory or adjust path
    try:
        predictor = UnifiedPredictor()
        
        # 2. Create Dummy Features (2048 is standard ResNet50 output)
        dummy_features = np.random.rand(2048).astype(np.float32)
        
        # 3. Test Single Prediction
        if 'svm' in predictor.models:
            print(f"SVM Prediction: {predictor.predict_single(dummy_features, 'svm')}")
            probs = predictor.predict_probability(dummy_features, 'svm')
            print(f"SVM Probabilities: {probs[:3]}... (Length: {len(probs)})")

        # 4. Test Ensemble
        print("\n--- Ensemble Result ---")
        result = predictor.predict_ensemble(dummy_features, method='soft_voting')
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"\n[Test Error] {e}")
        print("Make sure 'saved_models' directory exists with .pkl files.")