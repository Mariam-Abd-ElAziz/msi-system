# wrapper/adapter code for the prediction module

import joblib

class Predictor:
    def __init__(self, model_path):
        # Load the model only once
        self.model = joblib.load(model_path)

    # features: a 1-dimensional data structure of numbers that represents the important characteristics of one camera frame (sample).
    def predict(self, features):
        # Get predicted class and confidence
        # [features]: to make it 2D. one sample only (one frame) with multiple features
        # [0] to get the first (and only) sample's classes' probabilities
        probs = self.model.predict_proba([features])[0]
        class_id = probs.argmax()
        confidence = probs[class_id]
        return class_id, confidence
