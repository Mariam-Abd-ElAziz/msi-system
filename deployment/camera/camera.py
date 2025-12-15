# code for setting up and managing camera deployment

import cv2

class Camera:
    def __init__(self,
                device_index=0, # 0 : default web camera
                # what resolution does the feature extractor expect??????
                width=640, height=480 # frame resolution as desired
        ):

        # establish a connection to the camera and make it ready
        self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device_index}")
        
        # set desired resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
