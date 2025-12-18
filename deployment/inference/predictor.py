# wrapper/adapter code for the prediction module
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import UnifiedPredictor from src
from src.models.unified_predictor import UnifiedPredictor

CLASS_MAPPING = {
    0: 'glass', 1: 'paper', 2: 'cardboard', 
    3: 'plastic', 4: 'metal', 5: 'trash', 6: 'unknown'
}

class Predictor:
    def __init__(self):
        """
        Initialize the predictor with UnifiedPredictor
        """
        # Initialize UnifiedPredictor - it automatically loads models from saved_models/
        # Calculate the correct path to saved_models directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..', '..')
        saved_models_path = os.path.join(project_root, 'saved_models')
        
        # Initialize UnifiedPredictor with correct path
        self.predictor = UnifiedPredictor(model_dir=saved_models_path)
        

    def predict(self, features):
        """
        Predict using the loaded model
        
        Args:
            features: 1-dimensional data structure of numbers (normalized features)
        
        Returns:
            class_name: textual class label
            confidence: Confidence score (0.0 to 1.0)
        """
        try:
            # Get prediction probabilities
            probs = self.predictor.predict_probability(features, model_name='svm')
            
            # Get predicted class and confidence
            class_id = probs.argmax()
            confidence = probs[class_id]
            class_name = CLASS_MAPPING.get(class_id, 'unknown')
            return class_name, confidence
            
        except Exception as e:
            # Fallback in case of issues
            print(f"Prediction error: {e}")
            return 6, 0.0  # Return 'unknown' class with 0 confidence
