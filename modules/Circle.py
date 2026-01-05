import cv2
import numpy as np
from modules.Object import TrackedObject

class Circle(TrackedObject):
    def __init__(self, obj_id, pos=None, radius=None, max_lost=30):
        super().__init__(obj_id, max_lost)
        if pos is not None and radius is not None:
            self.last_data = (pos[0], pos[1], radius)
            self.is_visible = True

    def draw(self, frame, color=(0, 255, 0)):
        if not self.is_visible or self.last_data is None:
            return
        
        x, y, r = map(int, self.last_data)
        draw_color = color if self.lost_frames == 0 else (128, 128, 128)
        
        cv2.circle(frame, (x, y), r, draw_color, 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        cv2.putText(frame, f"ID:{self.id}", (x - 10, y - r - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
        
        coord_text = f"{x},{y}"
        cv2.putText(frame, coord_text, (x - 25, y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)