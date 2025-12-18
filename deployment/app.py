import cv2
import time
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from camera.camera import Camera
from feature_extraction.extractor import FeatureExtractor
from inference.predictor import Predictor

def main():

    camera = Camera()
    extractor = FeatureExtractor()
    predictor = Predictor()  # model_path is now ignored, but kept for compatibility

    # initialize FPS calculation
    prev_time = time.perf_counter()

    while True:
        frame = camera.read()
        
        # --- Feature extraction
        features = extractor.extract(frame)

        # --- Inference
        classLabel, confidence = predictor.predict(features)

        # --- FPS calculation
        curr_time = time.perf_counter()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # --- Visualization
        text = f"Class: {classLabel} | Conf: {confidence:.2f} | FPS: {fps:.1f}"
        cv2.putText(frame, text,
                    (10, 30), #position in frame to display text
                    cv2.FONT_HERSHEY_SIMPLEX, #font style
                    0.7, # font size
                    (0, 255, 0), #RGB color
                    2 #thickness
                    )

        # open a window and show the frame
        cv2.imshow("Live Classification", frame)

        # wait for 1 ms and check if ESC key is pressed to exit
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    # release resources
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
