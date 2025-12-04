import cv2
import numpy as np
from typing import Generator


class VideoManagerInterface:
    def __init__(self, path:str, color_system:str="RGB"):
        self.path = path
        self.color_system = color_system
        self.connection = None
        self.is_connection_open = False

    def open_connection(self):
        raise NotImplementedError("Method not implemented. This class is only interface class.")

    def close_connection(self):
        raise NotImplementedError("Method not implemented. This class is only interface class.")

    def __del__(self):
        self.close_connection()

    def __enter__(self):
        self.open_connection()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close_connection()


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
                    yield frame[:, :, ::-1]
                elif self.color_system == "BGR":
                    yield frame
        finally:
            self.close_connection()

    __call__ = read


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
