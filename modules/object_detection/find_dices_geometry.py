import cv2
import numpy as np
from modules.object_detection.objects import Dice

MIN_RADIUS = 15
MAX_RADIUS = 55
MIN_AREA = 800
MAX_AREA = 8500

ORANGE_LOWER1 = np.array([0, 120, 100])
ORANGE_UPPER1 = np.array([25, 255, 255])
ORANGE_LOWER2 = np.array([160, 120, 100])
ORANGE_UPPER2 = np.array([180, 255, 255])

BLUE_LOWER = np.array([95, 120, 100])
BLUE_UPPER = np.array([135, 255, 255])

def find_dices_geometry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask_orange1 = cv2.inRange(hsv, ORANGE_LOWER1, ORANGE_UPPER1)
    mask_orange2 = cv2.inRange(hsv, ORANGE_LOWER2, ORANGE_UPPER2)
    mask_orange = cv2.bitwise_or(mask_orange1, mask_orange2)
    
    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    
    valid_candidates = []
    
    for cnt in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        
        if not (MIN_RADIUS <= radius <= MAX_RADIUS):
            continue
            
        area = cv2.contourArea(cnt)
        
        if area < MIN_AREA or area > MAX_AREA: 
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity <= 0.65: 
            continue

        mask_cnt = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        
        total_pixels = cv2.countNonZero(mask_cnt)
        if total_pixels == 0: continue
        
        orange_px = cv2.countNonZero(cv2.bitwise_and(mask_orange, mask_orange, mask=mask_cnt))
        blue_px = cv2.countNonZero(cv2.bitwise_and(mask_blue, mask_blue, mask=mask_cnt))
        
        orange_ratio = orange_px / total_pixels
        blue_ratio = blue_px / total_pixels
        
        detected_label = None
        
        if orange_ratio > 0.2 and orange_ratio > blue_ratio:
            detected_label = "orange"
        elif blue_ratio > 0.2 and blue_ratio > orange_ratio:
            detected_label = "blue"
            
        if detected_label is not None:
            valid_candidates.append({'cnt': cnt, 'label': detected_label})

    final_dice_data = []
    
    for candidate in valid_candidates:
        cnt = candidate['cnt']
        x, y, w, h = cv2.boundingRect(cnt) 
        cx, cy = x + w // 2, y + h // 2
        
        neighbors = 0
        for other in valid_candidates:
            if cnt is other['cnt']: continue
            ox, oy, ow, oh = cv2.boundingRect(other['cnt'])
            
            dist = np.hypot(cx - (ox + ow // 2), cy - (oy + oh // 2))
            if dist < (MAX_RADIUS * 1.5): neighbors += 1
            
        if neighbors <= 1:
            final_dice_data.append(candidate)

    dice_objects = []
    for i, data in enumerate(final_dice_data):
        cnt = data['cnt']
        label_text = data['label']
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        new_dice = Dice(
            obj_id=i, 
            box=(x, y, w, h), 
            label=label_text, 
            contour=cnt
        )
        new_dice.is_visible = True
        dice_objects.append(new_dice)

    return tuple(dice_objects)