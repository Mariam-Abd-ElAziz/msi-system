import cv2
import time
import sys
import os
import numpy as np

# Add src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from camera.camera import Camera
from feature_extraction.extractor import FeatureExtractor
from inference.predictor import Predictor

# ==========================================
# CONFIGURATION
# ==========================================
# Choose your model: 'svm' or 'knn'
# SELECTED_MODEL = 'svm'
SELECTED_MODEL = 'knn'

# Set your threshold here (from your Notebook results)
# SVM Example: 0.60  (Probability)
# KNN Example: 0.70  (If using probability) or 18.4 (If using distance)
# REJECTION_THRESHOLD = 0.35
REJECTION_THRESHOLD = 0.5
# REJECTION_THRESHOLD = 70.94
# REJECTION_THRESHOLD = 18.4
# ==========================================

def main():
    print("="*40)
    print("      STARTING DEPLOYMENT SYSTEM      ")
    print("="*40)

    try:
        camera = Camera(device_index=0)
        extractor = FeatureExtractor()
        
        # PASS CONFIGURATION HERE
        predictor = Predictor(model_type=SELECTED_MODEL, threshold=REJECTION_THRESHOLD)
        
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # Model Warmup
    print("[App] Warming up AI models...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = extractor.extract(dummy_frame)
    predictor.predict(dummy_feats)
    print("[App] System Ready!")

    prev_time = time.perf_counter()
    fps_avg = 0.0
    
    try:
        while True:
            frame = camera.read()
            
            features = extractor.extract(frame)
            classLabel, confidence = predictor.predict(features)

            # FPS Calculation
            curr_time = time.perf_counter()
            time_diff = curr_time - prev_time
            prev_time = curr_time
            if time_diff > 0:
                fps = 1.0 / time_diff
                fps_avg = 0.9 * fps_avg + 0.1 * fps 

            # Visualization
            color = (0, 255, 0) if classLabel != 'unknown' else (0, 0, 255)
            
            # Info Bar
            cv2.rectangle(frame, (0, 0), (640, 45), (0, 0, 0), -1)
            
            # Display Model Used + Prediction
            status_text = f"[{SELECTED_MODEL.upper()}] Pred: {classLabel.upper()}"
            conf_text = f"Conf: {confidence:.2f} | FPS: {int(fps_avg)}"
            
            cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, conf_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Material Classifier", frame)

            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()