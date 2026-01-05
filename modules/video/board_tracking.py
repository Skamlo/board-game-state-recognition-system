import cv2
import numpy as np

MIN_SIDE_LENGTH = 600
MAX_SIDE_LENGTH = 800

def find_boards_geometry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boards = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = cv2.minAreaRect(c)
            (x, y), (w, h), angle = rect
            if (MIN_SIDE_LENGTH <= w <= MAX_SIDE_LENGTH) and \
               (MIN_SIDE_LENGTH <= h <= MAX_SIDE_LENGTH):
                
                valid_boards.append(approx)
            
    return valid_boards