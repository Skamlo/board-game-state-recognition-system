import numpy as np
import cv2
from typing import Tuple, List
from modules.object_detection.objects import Circle


def draw_circles(frame:np.array, circles:List[Circle], color:Tuple[int]=(0, 255, 0)):
    visible = 0
    for c in circles:
        if c.is_visible:
            visible += 1
            x, y, r = c.last_data
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Count: {visible}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )
