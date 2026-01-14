import cv2
import numpy as np


def find_circles_hough(warped_frame, difficulty):
    if warped_frame is None:
        return []
    
    h_img, w_img = warped_frame.shape[:2]
    gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    if difficulty == "easy":
        minrad = 46 
        maxrad = 50
        mind = 50
    elif difficulty == 'spec':
        minrad = 20 
        maxrad = 30
        mind = 50
    else:
        minrad = 50 
        maxrad = 54 
        mind = 54
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=mind,
        param1=50, param2=30, minRadius=minrad, maxRadius=maxrad
    )

    results = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
                y1 = max(0, y - r)
                y2 = min(h_img, y + r)
                x1 = max(0, x - r)
                x2 = min(w_img, x + r)
                roi = warped_frame[y1:y2, x1:x2]
                if roi.size == 0: continue

                mask_h, mask_w = roi.shape[:2]
                mask = np.zeros((mask_h, mask_w), dtype="uint8")
                cv2.circle(mask, (x - x1, y - y1), r, 255, -1)

                results.append((x, y, r, roi, mask))

    return results
