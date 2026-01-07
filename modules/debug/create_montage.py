import cv2
import numpy as np
import math


def create_montage(images, size=(100, 100), cols=5):
    """
    This function was created for debbuging, shows masks of chips to be predicted.
    """
    if not images: return np.zeros((100, 100, 3), dtype='uint8')
    
    resized = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized.append(cv2.resize(img, size))
    
    rows = math.ceil(len(resized) / cols)
    montage_h = rows * size[1]
    montage_w = cols * size[0]
    
    if rows == 1: montage_w = len(resized) * size[0]
        
    montage = np.zeros((montage_h, montage_w, 3), dtype='uint8')
    
    for i, img in enumerate(resized):
        r, c = i // cols, i % cols
        y1, y2 = r * size[1], (r + 1) * size[1]
        x1, x2 = c * size[0], (c + 1) * size[0]
        montage[y1:y2, x1:x2] = img
        
    return montage


