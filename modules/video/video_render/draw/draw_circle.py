import numpy as np
import cv2
from typing import Tuple
from modules.object_detection.objects import Circle


def draw_circle(frame:np.array, circle:Circle, color:Tuple[int]=(0, 255, 0)):
    if not circle.is_visible or circle.last_data is None:
        return
    
    x, y, r = map(int, circle.last_data)
    color = color if circle.lost_frames == 0 else (128, 128, 128)
    
    cv2.circle(frame, (x, y), r, color, 2)
    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    
    if circle.name:
        cv2.putText(
            frame, circle.name.upper(), (x - 20, y - 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
    
    cv2.putText(
        frame, f"ID:{circle.id}", (x - 10, y - r - 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 
    )
    
    coord_text = f"{x},{y}"
    cv2.putText(
        frame, coord_text, (x - 25, y + 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )
