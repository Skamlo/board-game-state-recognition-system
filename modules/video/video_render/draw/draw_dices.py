import numpy as np
import cv2
from typing import Tuple, List
from modules.object_detection.objects import Dice


def draw_dices(frame: np.array, dices: List[Dice], color: Tuple[int] = (0, 255, 0)):
    for dice in dices:
        if dice.contour is None:
            continue

        x, y, w, h = cv2.boundingRect(dice.contour)
        
        center_x = x + w // 2
        center_y = y + h // 2
    
        cv2.drawContours(frame, [dice.contour], -1, color, 3)

        text = f"{dice.label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        text_x = int(center_x - text_width // 2)
        text_y = int(center_y + text_height // 2)
        cv2.putText(frame, text, (text_x + 1, text_y + 1), font, font_scale, (0, 255, 0), thickness)