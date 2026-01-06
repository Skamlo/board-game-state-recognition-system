import numpy as np
import cv2
from typing import Tuple
from modules.object_detection.objects import Board


def draw_board(frame:np.array, board:Board, color:Tuple[int]=(0, 255, 0)):
    cv2.drawContours(frame, [board.last_data], -1, color, 3)
