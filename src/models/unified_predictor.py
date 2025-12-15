import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score


class UnifiedPredictor:
    """Unified predictor for SVM, KNN, and ensemble methods"""
    
    def __init__(self, model_dir="saved_models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.configs = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load both SVM and KNN models"""
        for model_name in ['svm', 'knn']:
            try:
                with open(self.model_dir / f"{model_name}_model.pkl", "rb") as f:
                    self.models[model_name] = pickle.load(f)
                
                with open(self.model_dir / f"{model_name}_config.json") as f:
                    self.configs[model_name] = json.load(f)
                print(f"[OK] Loaded {model_name.upper()}")
            except FileNotFoundError:
                print(f"[WARNING] {model_name.upper()} model not found")
    
    def predict_single(self, features, model_name='svm'):
        """Predict using specific model
        
        Args:
            features: Input feature vector (1D array)
            model_name: 'svm' or 'knn'
        
        Returns:
            Predicted class label
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return self.models[model_name].predict(features)[0]
    
    def predict_probability(self, features, model_name='knn'):
        """Get prediction probabilities (KNN only)
        
        Args:
            features: Input feature vector
            model_name: Model to use
        
        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # KNN can provide probability estimates
        if hasattr(self.models[model_name], 'predict_proba'):
            return self.models[model_name].predict_proba(features)[0]
        else:
            raise ValueError(f"{model_name} model doesn't support probability prediction")
    
    def predict_ensemble(self, features, method='voting'):
        """Ensemble prediction using multiple models
        
        Args:
            features: Input feature vector
            method: 'voting' for majority vote or 'average' for averaged probabilities
        
        Returns:
            Dictionary with ensemble prediction and individual predictions
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = int(model.predict(features)[0])
        
        if method == 'voting':
            # Majority voting
            votes = list(predictions.values())
            ensemble_pred = max(set(votes), key=votes.count)
        elif method == 'average':
            # For continuous predictions - use probability-based averaging if available
            ensemble_pred = int(np.mean(list(predictions.values())))
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_method': method,
            'individual_predictions': predictions
        }
    
    def batch_predict(self, features_array, model_name='svm'):
        """Predict on multiple samples
        
        Args:
            features_array: 2D array of features
            model_name: Model to use
        
        Returns:
            Array of predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.models[model_name].predict(features_array)
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Evaluate all models on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with accuracies
        """
        results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = {
                'accuracy': float(accuracy),
                'predictions': y_pred.tolist()
            }
        
        # Ensemble prediction
        ensemble_preds = []
        for features in X_test:
            result = self.predict_ensemble(features, method='voting')
            ensemble_preds.append(result['ensemble_prediction'])
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
        results['ensemble'] = {
            'accuracy': float(ensemble_accuracy),
            'predictions': ensemble_preds
        }
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"MODEL EVALUATION ON TEST SET")
        print(f"{'='*50}")
        for model_name, metrics in results.items():
            print(f"{model_name.upper():12} Accuracy: {metrics['accuracy']:.4f}")
        print(f"{'='*50}\n")
        
        return results
    
    def get_model_configs(self):
        """Get all loaded model configurations"""
        return self.configs


if __name__ == "__main__":
    predictor = UnifiedPredictor()
    
    # Test with dummy features
    dummy_features = np.random.rand(8255)
    
    print("\n--- Single Model Prediction ---")
    print(f"SVM prediction: {predictor.predict_single(dummy_features, 'svm')}")
    print(f"KNN prediction: {predictor.predict_single(dummy_features, 'knn')}")
    
    print("\n--- Ensemble Prediction ---")
    result = predictor.predict_ensemble(dummy_features)
    print(json.dumps(result, indent=2))
