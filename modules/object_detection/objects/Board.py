import numpy as np
import cv2
from modules.object_detection.objects import Object


class Board(Object):
    def __init__(self, obj_id, target_size=600, max_lost=50):
        super().__init__(obj_id, max_lost)
        self.target_size = target_size
        self.name = "board"

    def get_warped(self, frame):
        if not self.is_visible or self.last_data is None:
            return None

        pts = self.last_data.reshape(4, 2).astype("float32")

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]

        dst = np.array([
            [0, 0],
            [self.target_size - 1, 0],
            [self.target_size - 1, self.target_size - 1],
            [0, self.target_size - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (self.target_size, self.target_size))
