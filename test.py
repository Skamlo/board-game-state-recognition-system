import cv2
import numpy as np
import math
import time

from modules.circle_detection import find_circles_hough
from modules.GameBoard import GameBoard
from modules.TokenClassifier import TokenClassifier
MIN_SIDE_LENGTH = 600
MAX_SIDE_LENGTH = 800
TARGET_WARPED_SIZE = 600

def create_montage(images, size=(100, 100), cols=5):
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

def find_boards_geometry_hardcoded(frame):
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

def create_resizable_window(name, width, height):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)


create_resizable_window("Main Stream", 1000, 700)

cap = cv2.VideoCapture("./data/clips/medium.mp4")

tracked_boards = [] 
board_id_counter = 0
circle_classifier = TokenClassifier("./data/elements/circles")

ref_imgs = circle_classifier.get_masked_references_images()
if 1:
    montage_refs = create_montage(ref_imgs, size=(120, 120), cols=5)
    cv2.imshow("REFERENCES (MASKED)", montage_refs)
    cv2.waitKey(1) 

last_debug_time = 0
DEBUG_INTERVAL = 0.5 
while True:
    ret, frame = cap.read()
    if not ret: break

    candidates = find_boards_geometry_hardcoded(frame)
    for cand in candidates:
        matched = False
        cand_center = np.mean(cand.reshape(4, 2), axis=0)
        
        for board in tracked_boards:
            if board.last_data is not None:
                curr_center = np.mean(board.last_data.reshape(4, 2), axis=0)
                if np.linalg.norm(cand_center - curr_center) < 100:
                    board.update(cand)
                    matched = True
                    break
        
        if not matched:
            new_board = GameBoard(board_id_counter, target_size=TARGET_WARPED_SIZE)
            new_board.update(cand)
            tracked_boards.append(new_board)
            board_id_counter += 1
    all_orb_inputs = []

    for board in tracked_boards:
        if board.lost_frames > 20: continue
        
        if board.last_data is not None:
             cv2.drawContours(frame, [board.last_data], -1, (0, 255, 0), 3)

        warped = board.get_warped(frame)
        if warped is not None:    
        
            detection_results = find_circles_hough(warped)
            board.update_circles(detection_results)
            board.draw_circles(warped)
            cv2.imshow(f"Board {board.id}", warped)
            for circle in board.circles:
                if not circle.is_visible: continue
                circle.frames_since_recognition += 1
                if circle.name is None or circle.frames_since_recognition > 30:
                    
                    if hasattr(circle, 'last_roi') and circle.last_roi is not None:
                        prediction = circle_classifier.predict(circle.last_roi, mask=circle.last_mask)
                        
                        if prediction and prediction != "Unknown":
                            circle.name = prediction
                            circle.frames_since_recognition = 0
                        elif prediction == "Unknown":
                            circle.name = "Unknown" 
                        elif circle.name is None:
                            circle.name = "None"
                
                if hasattr(circle, 'last_roi') and circle.last_roi is not None:
                    roi = circle.last_roi
                    mask = circle.last_mask
                    if len(roi.shape) == 3:
                        gray_input = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_input = roi.copy()
                    masked_debug = cv2.bitwise_and(gray_input, gray_input, mask=mask)
                    kp = circle_classifier.orb.detect(gray_input, mask=mask)
                    debug_view = cv2.drawKeypoints(masked_debug, kp, None, color=(0, 255, 0), flags=0)
                    label = circle.name if circle.name else "?"
                    cv2.putText(debug_view, f"{circle.id}:{label}", (5, 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    all_orb_inputs.append(debug_view)
    if time.time() - last_debug_time > DEBUG_INTERVAL:
        if all_orb_inputs:
            montage_debug = create_montage(all_orb_inputs, size=(100, 100), cols=6)
            cv2.imshow("DEBUG: ORB Inputs + Keypoints", montage_debug)
        last_debug_time = time.time()

    cv2.imshow("Main Stream", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()