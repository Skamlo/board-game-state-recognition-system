import cv2
import numpy as np
import time
from collections import deque, Counter

from modules.object_detection.find_circles_hough import find_circles_hough
from modules.object_detection.objects import Board
from modules.object_detection.TokenClassifier import TokenClassifier
from modules.video import VideoReadManager, draw_board, draw_circle
from modules.game_engine import BoardLogic
from modules.video.video_manager.VideoWriteManager import VideoWriteManager

# --- CONFIGURATION ---
MIN_SIDE_LENGTH = 600
MAX_SIDE_LENGTH = 800
TARGET_WARPED_SIZE = 600
VIDEO_PATH = "./data/clips/easy2.mp4"
OUTPUT_VIDEO_PATH = "./data/output/result_with_scores.mp4"
ELEMENTS_PATH = "./data/elements/circles"

# --- CONSTANTS ---
HISTORY_LEN = 20        
CONFIDENCE_THRESH = 8   
PERSISTENCE_THRESH = 15 
PERSISTENCE_FRAMES = 20 
SKIP_FRAMES = 5

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

# --- ФУНКЦИЯ ОТРИСОВКИ ОЧКОВ ---
def draw_stats_panel(image, circles):
    """
    Принимает изображение доски и список кругов.
    Считает количество животных и добавляет панель слева.
    Считает TOTAL в кроликах.
    """
    if image is None: return None
    
    h, w = image.shape[:2]
    panel_w = 220 # Немного расширил панель для цифр
    
    # 1. Считаем животных
    # Фильтруем: только видимые (или в памяти) и только валидные имена
    valid_names = []
    for c in circles:
        # Учитываем персистентность или видимость
        is_active = c.is_visible or (getattr(c, 'persistence_timer', 0) > 0)
        
        if is_active and c.name and c.name not in ["free", "Unknown", None]:
            valid_names.append(c.name)
            
    counts = Counter(valid_names)
    
    # --- СТОИМОСТЬ ЖИВОТНЫХ (В КРОЛИКАХ) ---
    # Rabbit = 1
    # Sheep = 6
    # Pig = 12 (2 sheep)
    # Cow = 36 (3 pigs)
    # Horse = 72 (2 cows)
    points_map = {
        'rabbit': 1,
        'sheep': 6,
        'pig': 12,
        'cow': 36,
        'horse': 72,
        'small_dog': 0, 
        'big_dog': 0
    }

    # 2. Создаем панель
    # Делаем темно-серый фон
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40) 
    
    # Заголовок
    cv2.putText(panel, "SCORE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.line(panel, (10, 50), (panel_w - 10, 50), (100, 100, 100), 2)
    
    # Выводим счет
    y_offset = 90
    total_score = 0
    
    # Сортируем для стабильности порядка
    for name in sorted(counts.keys()):
        count = counts[name]
        
        # Считаем очки
        value = points_map.get(name, 0)
        total_score += count * value
        
        # Текст (например PIG: 2)
        text = f"{name.upper()}: {count}"
        
        # Цвет текста (можно сделать зависимым от типа животного)
        color = (255, 255, 255)
        if name == 'pig': color = (150, 150, 255) # Розоватый
        elif name == 'sheep': color = (200, 255, 200) 
        elif name == 'cow': color = (200, 200, 255) 
        elif name == 'horse': color = (100, 100, 255)

        cv2.putText(panel, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 40
        
    # Итого
    cv2.line(panel, (10, y_offset + 10), (panel_w - 10, y_offset + 10), (100, 100, 100), 1)
    # Выводим сумму в кроликах
    cv2.putText(panel, f"TOTAL: {total_score}", (15, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 3. Склеиваем панель и доску
    combined = np.hstack([panel, image])
    return combined

# --- INITIALIZATION ---
circle_classifier = TokenClassifier(ELEMENTS_PATH)
create_resizable_window("Main Stream", 1000, 700)
tracked_boards = []
board_id_counter = 0

cv2.waitKey(1)

# --- MAIN LOOP ---
paused = False
frame_count = 0 

with VideoReadManager(VIDEO_PATH) as reader, \
     VideoWriteManager(OUTPUT_VIDEO_PATH, fps=30) as writer:
    
    for frame in reader.read():
        if paused:
            key = cv2.waitKey(100)
            if key == ord(' '): paused = not paused
            if key == ord('q'): break
            continue
        
        frame_count += 1
        candidates = find_boards_geometry_hardcoded(frame)

        # 1. Board Tracking
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
        
        current_warps = []

        # 2. Process Boards
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
                    # --- Logic Start ---
                    if not hasattr(circle, 'pred_history'):
                        circle.pred_history = deque(maxlen=HISTORY_LEN)

                    if frame_count % SKIP_FRAMES == 0:
                        raw_pred = "Unknown"
                        if getattr(circle, 'lost_frames', 0) == 0:
                            if hasattr(circle, 'last_roi') and circle.last_roi is not None:
                                p = circle_classifier.predict(circle.last_roi, mask=circle.last_mask)
                                if p: raw_pred = p
                            
                            if raw_pred != "Unknown":
                                circle.pred_history.append(raw_pred)

                    if len(circle.pred_history) > 0:
                        most_common, count = Counter(circle.pred_history).most_common(1)[0]
                        old_name = circle.name
                        
                        if count >= CONFIDENCE_THRESH:
                            circle.name = most_common
                        
                        if count >= PERSISTENCE_THRESH and most_common != "free": 
                            circle.persistence_timer = PERSISTENCE_FRAMES
                        
                        board.logic.check_state_change(circle, old_name)
                    # --- Logic End ---
                    
                    # Drawing
                    display_color = (0, 255, 0)
                    label_text = circle.name
                    
                    is_persisting = getattr(circle, 'persistence_timer', 0) > 0 and getattr(circle, 'lost_frames', 0) > 0
                    
                    if is_persisting:
                        display_color = (255, 0, 255)
                    elif circle.name is None:
                         display_color = (128, 128, 128)
                         label_text = "..."
                    elif circle.name == "free":
                        display_color = (0, 255, 255)
                    
                    draw_circle(warped, circle, label=label_text, color=display_color)

            board.logic.draw_visual_events(warped)
            
            # --- ГЛАВНОЕ ИЗМЕНЕНИЕ: Добавляем панель статистики к варпу ---
            warped_with_stats = draw_stats_panel(warped, board.logic.circles)
            current_warps.append(warped_with_stats)
            
            cv2.imshow(f"Board {board.id+1}", warped)
            
        # Запись видео
        writer.write_composite(frame, current_warps)

        cv2.imshow("Main Stream", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'): 
            break
        elif key == ord(' '):
            paused = not paused

cv2.destroyAllWindows()