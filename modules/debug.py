import cv2
import numpy as np
import math

def create_montage(images, size=(100, 100), cols=5):
    """
    this function was created for debbuging, shows masks of chips to be predicted 
    """
    if not images:
        return np.zeros((100, 100, 3), dtype='uint8')

    resized_imgs = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized_imgs.append(cv2.resize(img, size))

    n_images = len(resized_imgs)
    rows = math.ceil(n_images / cols)
    
    montage_height = rows * size[1]
    montage_width = cols * size[0]
    montage = np.zeros((montage_height, montage_width, 3), dtype='uint8')

    for i, img in enumerate(resized_imgs):
        r = i // cols
        c = i % cols
        y1, y2 = r * size[1], (r + 1) * size[1]
        x1, x2 = c * size[0], (c + 1) * size[0]
        montage[y1:y2, x1:x2] = img

    return montage