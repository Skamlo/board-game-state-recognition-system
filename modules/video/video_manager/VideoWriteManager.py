import cv2
import numpy as np
import os
from .VideoManagerInterface import VideoManagerInterface

class VideoWriteManager(VideoManagerInterface):
    def __init__(self, path: str, fps: int = 30, codec: str = "mp4v"):
        super().__init__(path)
        self.fps = fps
        self.codec = codec
        self.height = None
        self.width = None
        self.connection = None
        
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def open_connection(self):
        if self.is_connection_open:
            self.close_connection()

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.connection = cv2.VideoWriter(self.path, fourcc, self.fps, (self.width, self.height))

        if not self.connection.isOpened():
            raise IOError(f"Could not open output video for writing: {self.path}")
        
        self.is_connection_open = True
        print(f"[INFO] Video Writer initialized: {self.path} ({self.width}x{self.height} @ {self.fps}fps)")

    def close_connection(self):
        if self.connection:
            self.connection.release()
            self.connection = None
            self.is_connection_open = False
            print("[INFO] Video Writer released.")

    def write_composite(self, main_frame: np.ndarray, side_frames: list[np.ndarray]):
        if main_frame is None: return

        # Целевая высота видео
        target_h = main_frame.shape[0]
        left_img = main_frame
        
        # --- ЛОГИКА ПРАВОЙ КОЛОНКИ ---
        # Высота одной планшетки (половина экрана)
        side_slot_h = target_h // 2
        
        frames_to_show = side_frames[-2:] if len(side_frames) >= 2 else side_frames
        right_stack = []
        
        # Вычисляем целевую ширину правой колонки на основе пропорций первого кадра
        # Если кадров нет, делаем ширину равной высоте (квадрат)
        side_slot_w = side_slot_h 
        if frames_to_show:
            ref_h, ref_w = frames_to_show[0].shape[:2]
            aspect_ratio = ref_w / ref_h
            side_slot_w = int(side_slot_h * aspect_ratio)

        # Функция для ресайза и заливки черным (если слотов меньше 2)
        def process_side_frame(img):
            if img is None:
                return np.zeros((side_slot_h, side_slot_w, 3), dtype=np.uint8)
            return cv2.resize(img, (side_slot_w, side_slot_h))

        # Заполняем стек (всегда 2 слота)
        if len(frames_to_show) == 2:
            right_stack.append(process_side_frame(frames_to_show[0]))
            right_stack.append(process_side_frame(frames_to_show[1]))
        elif len(frames_to_show) == 1:
            right_stack.append(process_side_frame(None)) # Верх пустой
            right_stack.append(process_side_frame(frames_to_show[0]))
        else:
            right_stack.append(process_side_frame(None))
            right_stack.append(process_side_frame(None))
            
        right_col = np.vstack(right_stack)
        
        # Склейка
        composite = np.hstack([left_img, right_col])
        self.write_frame(composite)

    def write_frame(self, frame: np.ndarray):
        if self.height is None or self.width is None:
            self.height, self.width = frame.shape[:2]
            self.open_connection()
        elif frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        
        if not self.is_connection_open:
            self.open_connection()

        self.connection.write(frame)

    def write(self, frame):
        self.write_frame(frame)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()
        
    def __call__(self, frame: np.ndarray):
        self.write_frame(frame)