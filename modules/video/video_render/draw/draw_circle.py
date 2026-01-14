import numpy as np
import cv2
from typing import Tuple
from modules.object_detection.objects import Circle, Board

def draw_circle(frame: np.array, circle: Circle, color: Tuple[int] = (0, 255, 0), label: str = None):
    """
    Отрисовывает круг. 
    Аргумент label позволяет передать кастомный текст (например "FREE"), 
    игнорируя circle.name.
    """
    if not circle.is_visible or circle.last_data is None:
        return
    
    x, y, r = map(int, circle.last_data)
    
    # Если круг потерян (lost_frames > 0), рисуем серым
    final_color = color if circle.lost_frames == 0 else (128, 128, 128)
    
    # 1. Рисуем сам круг
    cv2.circle(frame, (x, y), r, final_color, 2)
    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    
    # 2. Определяем, какой текст писать
    # Если передали label извне (из main.py) - используем его. Иначе берем имя из круга.
    text_to_show = label if label is not None else circle.name

    if text_to_show:
        cv2.putText(
            frame, 
            str(text_to_show).upper(), 
            (x - 20, y - 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            final_color, # Используем тот же цвет, что и у круга (Зеленый или Желтый)
            2
        )
    
    # 3. Рисуем ID
    cv2.putText(
        frame, f"ID:{circle.id}", (x - 10, y - r - 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, final_color, 
    )
    
    # 4. Рисуем координаты (мелким шрифтом)
    coord_text = f"{x},{y}"
    cv2.putText(
        frame, coord_text, (x - 25, y + 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )

def draw_circles(frame: np.array, circles: list[Circle]):
    for circle in circles:
        draw_circle(frame, circle)

def draw_board(frame: np.array, board: Board, color: Tuple[int] = (0, 255, 0)):
    if board.last_data is None:
        return
    
    # Рисуем контур доски
    cnt = board.last_data.astype(int)
    cv2.drawContours(frame, [cnt], -1, color, 2)
    
    # Рисуем название доски в центре
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(frame, f"Board {board.id+1}", (cX - 20, cY), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)