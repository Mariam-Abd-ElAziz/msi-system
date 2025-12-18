# code for setting up and managing camera deployment

import cv2

DEFAULT_CAMERA_WIDTH = 480
DEFAULT_CAMERA_HEIGHT = 360
class Camera:
    def __init__(self,
                device_index=0, # 0 : default web camera
                width=-1,      # desired width
                height=-1      # desired height
        ):

        optimal_width, optimal_height = 0, 0

        if width != -1 and height != -1:
            print(f"testing the requested resolution: {width}x{height}")
            if self.is_resolution_supported(device_index, width, height):
                print(f"Using the nearest valid resolution to the requested resolution: {width}x{height}")
                optimal_width, optimal_height = width, height
            else:
                print("Requested resolution not supported")
                optimal_width, optimal_height = self.find_optimal_resolution(device_index)
        else:
            # Find optimal resolution automatically
            optimal_width, optimal_height = self.find_optimal_resolution(device_index)

        # establish a connection to the camera and make it ready
        self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device_index}")
        
        # set the suitable resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, optimal_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, optimal_height)

    def read(self):
        #capture one frame from the camera
        # ret: boolean indicating success
        # frame: the captured frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self):
        self.cap.release()

    """Find the best supported resolution for the camera of the running machine"""
    def find_optimal_resolution(self, device_index=0):
        
        # Common resolutions to test (in order of preference)
        test_resolutions = [
            (1280, 720),   # HD
            (640, 480),    # VGA  
            (640, 360),    # 16:9 VGA
            (480, 360),    # 4:3
            (320, 240),    # QVGA
            (320, 180),    # Your working resolution
            (160, 120),    # Low resolution fallback
        ]
        
        print("Finding the optimal camera resolution...")
        
        for width, height in test_resolutions:
            if self.is_resolution_supported(device_index, width, height):
                print(f"Using resolution: {width}x{height}")
                return width, height

        print("⚠️ No working resolution found, using defaults")
        return DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT  # Your working fallback

    def is_resolution_supported(self, device_index, width, height):
        cap = cv2.VideoCapture(device_index)
        
        if cap.isOpened():
            # Try to set the resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Read a test frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                print(f"✅ {width}x{height} -> {actual_width:.0f}x{actual_height:.0f}")
                cap.release()
                return True
            else:
                print(f"❌ {width}x{height} -> Failed to read frame")
                cap.release()
                return False
        else:
            print(f"❌ {width}x{height} -> Cannot open camera")
            cap.release()
            return False
        