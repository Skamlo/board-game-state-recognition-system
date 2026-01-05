import cv2
import numpy as np
from modules.circle_detection import find_circles_hough
from modules.GameBoard import GameBoard

MIN_SIDE_LENGTH = 600
MAX_SIDE_LENGTH = 800
TARGET_WARPED_SIZE = 600

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

cap = cv2.VideoCapture("./data/clips/easy2.mp4")

tracked_boards = [] 
board_id_counter = 0

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

    for board in tracked_boards:
        if board.lost_frames > 20: continue
        if board.last_data is not None:
             cv2.drawContours(frame, [board.last_data], -1, (0, 255, 0), 3)
        warped = board.get_warped(frame)
        if warped is not None:
            
            raw_circles = find_circles_hough(warped)
            board.update_circles(raw_circles)
            board.draw_circles(warped)
            cv2.imshow(f"Board {board.id}", warped)
            # print(f"Board {board.id} has {len(board.circles)} objects")

    cv2.imshow("Main Stream", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()