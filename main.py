import cv2
import numpy as np
import time
from modules.object_detection.find_circles_hough import find_circles_hough
from modules.object_detection.objects import Board
from modules.object_detection.TokenClassifier import TokenClassifier
from modules.video import VideoReadManager, draw_board, draw_circle, draw_circles
from modules.game_engine import BoardLogic

# --- CONFIGURATION ---
MIN_SIDE_LENGTH = 600
MAX_SIDE_LENGTH = 800
TARGET_WARPED_SIZE = 600
VIDEO_PATH = "./data/clips/easy2.mp4"

# --- AUXILIARY FUNCTIONS ---
from modules.debug import create_montage

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

# --- INITIALIZATION ---
create_resizable_window("Main Stream", 1000, 700)
tracked_boards = []
board_id_counter = 0
circle_classifier = TokenClassifier("./data/elements/circles")
ref_imgs = circle_classifier.get_masked_references_images()

montage_refs = create_montage(ref_imgs, size=(120, 120), cols=5)
cv2.imshow("REFERENCES (MASKED)", montage_refs)
cv2.waitKey(1)

# --- MAIN LOOP ---
last_debug_time = 0
DEBUG_INTERVAL = 0.5

with VideoReadManager(VIDEO_PATH) as reader:
    for frame in reader.read():
        candidates = find_boards_geometry_hardcoded(frame)

        # 1. Update Board Tracking
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
                new_board = Board(board_id_counter, target_size=TARGET_WARPED_SIZE)
                new_board.update(cand)
                new_board.logic = BoardLogic(target_size=TARGET_WARPED_SIZE)
                tracked_boards.append(new_board)
                board_id_counter += 1
        
        all_orb_inputs = []

        # 2. Render Visualizations
        for board in tracked_boards:
            if not board.is_visible:
                continue
            
            # Visualizing a board
            draw_board(frame, board, color=(0, 255, 0))

            warped = board.get_warped(frame)
            if warped is None:
                continue
            
            # Detect circles on warped board
            detection_results = find_circles_hough(warped)
            board.logic.update_circles(detection_results)
            
            # VISUALIZATION: Draw detailed info for each circle on the warped board
            for circle in board.logic.circles:
                if circle.is_visible:
                    # This uses your draw_circle.py which draws ID, Name, and Coords
                    draw_circle(warped, circle)
                    
                    # Classification Logic
                    circle.frames_since_recognition += 1
                    if circle.name is None or circle.frames_since_recognition > 30:
                        if hasattr(circle, 'last_roi') and circle.last_roi is not None:
                            pred = circle_classifier.predict(circle.last_roi, mask=circle.last_mask)
                            circle.name = pred if pred else "Unknown"
                            circle.frames_since_recognition = 0

            # Show the top-down view with circles
            cv2.imshow(f"Board {board.id}", warped)

        # Final display of the main camera stream
        cv2.imshow("Main Stream", frame)
        
        if cv2.waitKey(1) == ord('q'): 
            break

cv2.destroyAllWindows()
