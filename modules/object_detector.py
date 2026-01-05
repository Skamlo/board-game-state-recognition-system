import cv2
import numpy as np

def detect_by_color(warped_img, lower_color, upper_color):
    if warped_img is None: return []
    
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for c in contours:
        if cv2.contourArea(c) > 200: 
            x, y, w, h = cv2.boundingRect(c)
            objects.append({'rect': (x, y, w, h), 'center': (x + w//2, y + h//2)})
            
    return objects