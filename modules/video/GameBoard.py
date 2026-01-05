import cv2
import numpy as np
from modules.Object import TrackedObject
from modules.Circle import Circle

class GameBoard(TrackedObject):
    def __init__(self, obj_id, target_size=600, max_lost=50):
        super().__init__(obj_id, max_lost)
        self.target_size = target_size
        
        # --- МАССИВ ДЛЯ ХРАНЕНИЯ КРУГОВ ---
        # Здесь будут лежать объекты класса Circle
        self.circles = [] 
        self.next_circle_id = 0
        self.relative_zones = [
            (0.14, 0.16), (0.32, 0.15), (0.50, 0.14), (0.68, 0.15), (0.86, 0.16),
            (0.22, 0.35), (0.40, 0.33), (0.59, 0.33), (0.77, 0.35),
            (0.31, 0.52), (0.50, 0.50), (0.69, 0.52),
            (0.41, 0.69), (0.59, 0.69),
            (0.50, 0.86)
        ]
        self.search_radius = 45
        self.zones = []
        for (rx, ry) in self.relative_zones:
            self.zones.append((int(rx * self.target_size), int(ry * self.target_size)))
            
        self.search_radius = 45
    
    
    def get_warped(self, frame):
        if not self.is_visible or self.last_data is None:
            return None
        
        pts = self.last_data.reshape(4, 2).astype("float32")
        
        # Упорядочиваем углы: TL, TR, BR, BL
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        
        dst = np.array([
            [0, 0], [self.target_size-1, 0], 
            [self.target_size-1, self.target_size-1], [0, self.target_size-1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (self.target_size, self.target_size))

    def update_circles(self, raw_detections):
        """
        Принимает список (x, y, r) от HoughCircles и обновляет массив self.circles
        """
        if not raw_detections:
            # Если ничего не нашли, помечаем все текущие круги как потерянные
            for circle in self.circles:
                circle.update(None)
        else:
            # Логика сопоставления (Matching):
            # Пытаемся найти для каждого старого круга ближайший новый
            used_detections = [False] * len(raw_detections)
            
            for circle in self.circles:
                # Если круг уже потерян давно, пропускаем
                if not circle.is_visible: continue

                best_idx = -1
                min_dist = 50 # Максимальная дистанция сдвига (пиксели)

                cx, cy, _ = circle.last_data
                
                for i, (nx, ny, nr) in enumerate(raw_detections):
                    if used_detections[i]: continue
                    
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array([nx, ny]))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                
                if best_idx != -1:
                    circle.update(raw_detections[best_idx])
                    used_detections[best_idx] = True
                else:
                    circle.update(None) # Не нашли пару -> потерян
            
            # Создаем новые круги для оставшихся детекций
            for i, (nx, ny, nr) in enumerate(raw_detections):
                if not used_detections[i]:
                    new_circle = Circle(self.next_circle_id, pos=(nx, ny), radius=nr)
                    self.circles.append(new_circle)
                    self.next_circle_id += 1

        # Очистка мусора: удаляем круги, которые потеряны слишком давно
        self.circles = [c for c in self.circles if c.lost_frames < c.max_lost]

    def draw_circles(self, image):
        """Рисует все активные круги из массива на переданном изображении"""
        for circle in self.circles:
            if circle.is_visible:
                circle.draw(image)
        
        # Выводим общее количество
        cv2.putText(image, f"Count: {len([c for c in self.circles if c.is_visible])}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)