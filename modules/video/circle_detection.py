import cv2
import numpy as np

def find_circles_hough(warped_frame):
    if warped_frame is None:
        return []
    gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=46, maxRadius=50
    )

    detected = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detected.append((x, y, r)) # (x, y, radius)
            
    return detected