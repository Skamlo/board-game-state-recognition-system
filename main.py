import cv2
import numpy as np
import time

# --- MODULE IMPORTS ---
from modules.object_detection.find_circles_hough import find_circles_hough
from modules.object_detection.find_boards_geometry import find_boards_geometry
from modules.object_detection.find_dices_geometry import find_dices_geometry
from modules.object_detection.objects import Board, Dice
from modules.object_detection.TokenClassifier import TokenClassifier
from modules.video import VideoReadManager, draw_board, draw_circle, draw_circles, draw_dices
from modules.game_engine import BoardLogic
from modules.debug import create_montage

# --- CONFIGURATION ---
MIN_SIDE_LENGTH = 600
MAX_SIDE_LENGTH = 800
TARGET_WARPED_SIZE = 600
VIDEO_PATH = "./data/clips/easy2.mp4"
ELEMENTS_PATH = "./data/elements/circles"
DICE_SAMPLES_PATH = "./data/elements/dice/labeled"

def create_resizable_window(name, width, height):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)

# --- INITIALIZATION ---
# 1. Classifiers
circle_classifier = TokenClassifier(ELEMENTS_PATH)
dice_classifier = TokenClassifier(DICE_SAMPLES_PATH)

# 2. Windows & Visuals
create_resizable_window("Main Stream", 1000, 700)

ref_imgs_circles = circle_classifier.get_masked_references_images()
montage_circles = create_montage(ref_imgs_circles, size=(120, 120), cols=5)
cv2.imshow("REFERENCES (Circles)", montage_circles)

ref_imgs_dices = dice_classifier.get_masked_references_images()
if ref_imgs_dices:
    montage_dices = create_montage(ref_imgs_dices, size=(120, 120), cols=5)
    cv2.imshow("REFERENCES (Dices)", montage_dices)

cv2.waitKey(1)

# 3. Tracking Lists
tracked_boards = []
board_id_counter = 0

tracked_dices = []
dice_id_counter = 0

# --- MAIN LOOP ---
with VideoReadManager(VIDEO_PATH) as reader:
    for frame in reader.read():
        
        # BOARD DETECTION & LOGIC
        board_candidates = find_boards_geometry(frame)

        # Update Board Tracking
        for cand in board_candidates:
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

        # Render Boards & Circles
        for board in tracked_boards:
            if not board.is_visible: continue
            
            draw_board(frame, board, color=(0, 255, 0))

            warped = board.get_warped(frame)
            if warped is None: continue
            
            detection_results = find_circles_hough(warped)
            board.logic.update_circles(detection_results)
            
            for circle in board.logic.circles:
                if circle.is_visible:
                    draw_circle(warped, circle)
                    
                    # Periodic Re-classification for Circles
                    circle.frames_since_recognition += 1
                    if circle.name is None or circle.frames_since_recognition > 30:
                        if hasattr(circle, 'last_roi') and circle.last_roi is not None:
                            pred = circle_classifier.predict(circle.last_roi, mask=circle.last_mask)
                            circle.name = pred if pred else "Unknown"
                            circle.frames_since_recognition = 0

            cv2.imshow(f"Board {board.id+1}", warped)

        # DICE DETECTION & LOGIC
        # 1. Detect dice in current frame
        dice_candidates_objects = find_dices_geometry(frame, dice_classifier)
        
        # 2. Match candidates to existing tracked dice (Tracking Logic)
        for d in tracked_dices:
            d.update(None)

        for new_dice_obj in dice_candidates_objects:
            matched = False
            # Try to match with existing tracked dice based on distance
            new_center = np.array(new_dice_obj.center)
            
            for tracked_die in tracked_dices:
                if tracked_die.center is not None:
                    dist = np.linalg.norm(new_center - np.array(tracked_die.center))
                    if dist < 50:
                        # Update the existing object with new data
                        tracked_die.update(new_dice_obj.contour)
                        tracked_die.box = new_dice_obj.box
                        tracked_die.center = new_dice_obj.center
                        tracked_die.contour = new_dice_obj.contour
                        
                        # Only update color/label if known, or if previous was unknown
                        if new_dice_obj.color != "Unknown":
                            tracked_die.color = new_dice_obj.color
                        if new_dice_obj.label != "Unknown":
                            tracked_die.label = new_dice_obj.label
                            
                        matched = True
                        break
            
            if not matched:
                # Register new dice
                new_dice_obj.id = dice_id_counter
                tracked_dices.append(new_dice_obj)
                dice_id_counter += 1

        # 3. Clean up list (remove dice lost for too long)
        tracked_dices = [d for d in tracked_dices if d.lost_frames < d.max_lost]

        # 4. Draw Dices
        visible_dices = [d for d in tracked_dices if d.is_visible]
        draw_dices(frame, visible_dices)

        # FINAL DISPLAY
        cv2.imshow("Main Stream", frame)
        
        if cv2.waitKey(1) == ord('q'): 
            break

cv2.destroyAllWindows()
