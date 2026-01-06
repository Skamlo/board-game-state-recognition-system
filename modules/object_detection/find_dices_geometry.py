import cv2
import numpy as np
from modules.object_detection.TokenClassifier import TokenClassifier
from modules.object_detection.objects import Dice

ORANGE_RANGE = (np.array([31, 98, 68]), np.array([11, 94, 19]))
BLUE_RANGE = (np.array([186, 93, 84]), np.array([200, 54, 15]))


def find_dices_geometry(frame, classifier: TokenClassifier):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blur, 0, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- Geometry Filtering ---
    circular_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 10000: continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            circular_contours.append(cnt)

    final_dice_contours = []
    for cnt in circular_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        
        neighbors = 0
        for other in circular_contours:
            if cnt is other: continue
            ox, oy, ow, oh = cv2.boundingRect(other)
            dist = np.hypot(cx - (ox + ow // 2), cy - (oy + oh // 2))
            if dist < 150: neighbors += 1
            
        if neighbors <= 1:
            final_dice_contours.append(cnt)

    # --- Classification & Object Creation ---
    dice_objects = []
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i, cnt in enumerate(final_dice_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Extract ROI with margin for SIFT
        margin = 5
        y1, y2 = max(0, y-margin), min(frame.shape[0], y+h+margin)
        x1, x2 = max(0, x-margin), min(frame.shape[1], x+w+margin)
        roi = frame[y1:y2, x1:x2]
        roi_hsv = hsv_frame[y1:y2, x1:x2]
        
        if roi.size == 0: continue

        # Classify Color
        o_mask = cv2.inRange(roi_hsv, ORANGE_RANGE[0], ORANGE_RANGE[1])
        b_mask = cv2.inRange(roi_hsv, BLUE_RANGE[0], BLUE_RANGE[1])
        
        o_score = cv2.countNonZero(o_mask)
        b_score = cv2.countNonZero(b_mask)
        
        color_res = "Unknown"
        if o_score > b_score and o_score > 50:
            color_res = "Orange"
        elif b_score > o_score and b_score > 50:
            color_res = "Blue"

        # Classify Label (SIFT)
        label_res = classifier.predict(roi)
        if label_res is None: label_res = "Unknown"

        # Instantiate Dice Object
        new_dice = Dice(
            obj_id=i, 
            box=(x, y, w, h), 
            color=color_res, 
            label=label_res, 
            contour=cnt
        )
        new_dice.is_visible = True
        dice_objects.append(new_dice)

    return tuple(dice_objects)
