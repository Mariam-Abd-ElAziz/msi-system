import cv2
import time

class Camera:
    def __init__(self, device_index=0, width=-1, height=-1):
        self.device_index = device_index
        
        # 1. Open the camera first
        print(f"[Camera] Opening device {device_index}...")
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW) # CAP_DSHOW helps on Windows, remove if on Linux/Mac

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device_index}")

        # 2. Set Resolution
        if width != -1 and height != -1:
            print(f"[Camera] Requesting resolution: {width}x{height}")
            self.set_resolution(width, height)
        else:
            print("[Camera] Finding optimal resolution...")
            self.find_and_set_optimal_resolution()

        # 3. Verify final resolution
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Camera] Active resolution: {int(actual_w)}x{int(actual_h)}")

        # 4. Warmup (drain buffer to allow auto-focus/brightness adjustment)
        self.warmup(frames=10)

    def set_resolution(self, width, height):
        """Try to set the resolution on the current capture object"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def find_and_set_optimal_resolution(self):
        """Test common resolutions and pick the best supported one"""
        test_resolutions = [
            (1280, 720), (640, 480), (640, 360), (320, 240)
        ]
        
        for w, h in test_resolutions:
            self.set_resolution(w, h)
            # Check if it stuck
            act_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            act_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if int(act_w) == w and int(act_h) == h:
                print(f"[Camera] Success setting {w}x{h}")
                return
        
        print("[Camera] Could not set standard resolutions, using driver default.")

    def warmup(self, frames=10):
        print("[Camera] Warming up sensor...")
        for _ in range(frames):
            self.cap.read()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()