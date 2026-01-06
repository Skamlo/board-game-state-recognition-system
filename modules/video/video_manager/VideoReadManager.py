import cv2
import numpy as np
from typing import Generator
from modules.video import VideoManagerInterface


class VideoReadManager(VideoManagerInterface):
    def open_connection(self):
        if self.is_connection_open:
            self.close_connection()

        self.connection = cv2.VideoCapture(self.path)
        
        if not self.connection.isOpened():
            raise IOError(f"Error: Could not open video file at {self.path}")
        
        self.is_connection_open = True

    def close_connection(self):
        if self.connection:
            self.connection.release()
            self.connection = None
            self.is_connection_open = False

    def read(self) -> Generator[np.ndarray, None, None]:
        if not self.is_connection_open:
            self.open_connection()

        try:
            while True:
                ret, frame = self.connection.read()
                if not ret:
                    break

                if self.color_system == "RGB":
                    yield frame
                elif self.color_system == "BGR":
                    yield frame[:, :, ::-1]
        finally:
            self.close_connection()

    __call__ = read
