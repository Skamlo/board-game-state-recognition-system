import cv2
import numpy as np
from modules.video import VideoManagerInterface


class VideoWriteManager(VideoManagerInterface):
    def __init__(self, path:str, fps:int=30, codec:str="mp4v"):
        super().__init__(path)
        self.fps = fps
        self.codec = codec
        self.height = None
        self.width = None

    def open_connection(self):
        if self.is_connection_open:
            self.close_connection()

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.connection = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))

        if not self.connection.isOpened():
            raise IOError(f"Could not open output video for writing: {self.path}")
        
        self.is_connection_open = True

    def close_connection(self):
        if self.connection:
            self.connection.release()
            self.connection = None
            self.is_connection_open = False

    def write_frame(self, frame:np.ndarray):
        if self.height is None or self.width is None:
            self.height, self.width = frame.shape[:2]
        elif frame.shape[:2] != (self.height, self.width):
            raise ValueError("All frames must have identical dimensions.")
        
        if not self.is_connection_open:
            self.open_connection()

        if self.color_system == "RGB":
            self.connection.write(frame[:, :, ::-1])
        elif self.color_system == "BGR":
            self.connection.write(frame)

    def __enter__(self):
        return self
        
    def __call__(self, frame:np.ndarray):
        self.write_frame(frame)
