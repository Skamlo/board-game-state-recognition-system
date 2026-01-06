import numpy as np
import cv2
from typing import Tuple
from modules.object_detection.objects import Dice


def draw_dice(frame:np.array, dice:Dice, color:Tuple[int]=(0, 255, 0)):
    cv2.drawContours(frame, [dice.contour], -1, color, 3)
