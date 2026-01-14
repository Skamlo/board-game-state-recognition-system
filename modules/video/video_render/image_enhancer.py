import cv2
import numpy as np

def apply_lighting_correction(frame, clip_limit=2.0, tile_grid_size=(4, 4), gamma_target=120):
    if frame is None:
        return None

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    mean_brightness = np.mean(l_clahe)
    if mean_brightness == 0:
        mean_brightness = 1

    gamma = np.log(gamma_target / 255.0) / np.log(mean_brightness / 255.0)
    gamma = np.clip(gamma, 0.6, 2.0)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    l_final = cv2.LUT(l_clahe, table)

    merged = cv2.merge((l_final, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return result
