import cv2
import numpy as np
import time
from modules.object_detection.find_circles_hough import find_circles_hough
from modules.object_detection.objects import Board
from modules.object_detection.TokenClassifier import TokenClassifier
from modules.video import VideoReadManager, draw_board, draw_circle, draw_circles
from modules.game_engine import BoardLogic
from modules.debug.hist_des_debbuging import generate_debug_image 
from modules.debug import create_montage

# --- CONFIGURATION ---
MIN_SIDE_LENGTH = 600
MAX_SIDE_LENGTH = 800
TARGET_WARPED_SIZE = 600
VIDEO_PATH = "./data/clips/easy2.mp4"
ELEMENTS_PATH = "./data/elements/circles"

# --- AUXILIARY FUNCTIONS ---
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

def draw_hist_plot(hist, name, size=(150, 80)):
    canvas = np.zeros((size[1], size[0], 3), dtype="uint8")
    if hist is None: return canvas
    
    disp_hist = hist.copy()
    cv2.normalize(disp_hist, disp_hist, alpha=0, beta=size[1], norm_type=cv2.NORM_MINMAX)
    
    bin_w = max(1, size[0] // len(hist))
    
    for i in range(len(hist)):
        val = int(disp_hist[i])
        cv2.rectangle(canvas, (i * bin_w, size[1] - val), 
                      ((i + 1) * bin_w, size[1]), (0, 255, 0), -1)
    
    cv2.putText(canvas, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas

# --- INITIALIZATION ---
circle_classifier = TokenClassifier(ELEMENTS_PATH)
create_resizable_window("Main Stream", 1000, 700)
tracked_boards = []
board_id_counter = 0

ref_imgs = circle_classifier.get_masked_references_images()
if ref_imgs:
    montage_refs = create_montage(ref_imgs, size=(120, 120), cols=5)
    cv2.imshow("REFERENCES (MASKED)", montage_refs)

hist_plots = []
sorted_refs = sorted(circle_classifier.references.items()) 
for name, data in sorted_refs:
    if 'hist' in data:
        plot = draw_hist_plot(data['hist'], name)
        hist_plots.append(plot)
if hist_plots:
    montage_hists = create_montage(hist_plots, size=(150, 80), cols=4)
    cv2.imshow("REFERENCES (HISTOGRAMS)", montage_hists)

cv2.waitKey(1)

# --- MAIN LOOP ---
paused = False

with VideoReadManager(VIDEO_PATH) as reader:
    for frame in reader.read():
        if paused:
            key = cv2.waitKey(100)
            if key == ord(' '): paused = not paused
            if key == ord('q'): break
            continue

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
                new_board = Board(board_id_counter, target_size=TARGET_WARPED_SIZE)
                new_board.update(cand)
                new_board.logic = BoardLogic(target_size=TARGET_WARPED_SIZE)
                tracked_boards.append(new_board)
                board_id_counter += 1
        
        debug_panels_list = []
        
        for board in tracked_boards:
            if not board.is_visible:
                continue
            
            draw_board(frame, board, color=(0, 255, 0))

            warped = board.get_warped(frame)
            if warped is None: continue
            
            detection_results = find_circles_hough(warped)
            board.logic.update_circles(detection_results)
            
            for circle in board.logic.circles:
                if circle.is_visible:
                    # 1. Classification Logic
                    circle.frames_since_recognition += 1
                    
                    if circle.name is None or circle.frames_since_recognition > 30:
                        if hasattr(circle, 'last_roi') and circle.last_roi is not None:
                            pred = circle_classifier.predict(circle.last_roi, mask=circle.last_mask)
                            
                            if pred and pred != "Unknown": 
                                circle.name = pred 
                                circle.since_not_unknown = 0
                            elif circle.since_not_unknown > 2:
                                circle.name = "Unknown"
                            else: 
                                circle.since_not_unknown += 1
                            
                            circle.frames_since_recognition = 0

                    if hasattr(circle, 'last_roi') and circle.last_roi is not None:
                        panel = generate_debug_image(
                            circle_classifier, 
                            circle.last_roi, 
                            circle.last_mask, 
                            circle.name if circle.name else "Unknown",
                            circle.id
                        )
                        if panel is not None:
                            debug_panels_list.append(panel)
                    
                    draw_circle(warped, circle)

            # Show the top-down view
            detection_results = find_circles_hough(warped)
            board.logic.update_circles(detection_results)
            board.logic.draw_notification(warped)
            cv2.imshow(f"Board {board.id+1}", warped)
            
        # 3. Global Debug View 
        if debug_panels_list:
            full_debug_view = create_montage(debug_panels_list, size=(160, 270), cols=8)
            cv2.imshow("DEBUG: All Tokens Analysis", full_debug_view)
            
        cv2.imshow("Main Stream", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'): 
            break
        elif key == ord(' '):
            paused = not paused

cv2.destroyAllWindows()