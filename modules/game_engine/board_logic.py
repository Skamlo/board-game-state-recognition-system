import numpy as np
import cv2
from modules.object_detection.objects import Circle

class BoardLogic:
    def __init__(self, target_size=600, search_radius=45):
        self.target_size = target_size
        self.search_radius = search_radius

        self.circles = []
        self.next_circle_id = 0

        self.relative_zones = [
            (0.14, 0.16), (0.32, 0.15), (0.50, 0.14), (0.68, 0.15), (0.86, 0.16),
            (0.22, 0.35), (0.40, 0.33), (0.59, 0.33), (0.77, 0.35),
            (0.31, 0.52), (0.50, 0.50), (0.69, 0.52),
            (0.41, 0.69), (0.59, 0.69),
            (0.50, 0.86)
        ]
        
        self.zones = [
            (int(x * target_size), int(y * target_size))
            for x, y in self.relative_zones
        ]

        # --- VISUAL DEBUG VARIABLES ---
        self.visual_events = []         
        self.frame_counter = 0          
        self.last_event_times = {}      
        self.notification_message = ""  
        self.notification_timer = 0

    def add_visual_event(self, event_type, pos, label=None, circle_id=None):
        self.visual_events.append({
            'type': event_type,      
            'pos': (int(pos[0]), int(pos[1])),
            'timer': 45              
        })
        
        if label:
            action = "Added" if event_type == 'plus' else "Removed"
            self.notification_message = f"{action}: {label}"
            self.notification_timer = 60 
        
        if circle_id is not None:
            self.last_event_times[circle_id] = self.frame_counter

    def update_circles(self, detection_results):
        self.frame_counter += 1
        
        # Список ID кругов, которые нужно удалить в этом кадре
        circles_to_remove = set()

        if not detection_results:
            # Если вообще ничего не нашли, уменьшаем таймеры у всех
            for circle in self.circles:
                persistence = getattr(circle, 'persistence_timer', 0)
                if persistence > 0:
                    circle.persistence_timer -= 1
                    circle.lost_frames = getattr(circle, 'lost_frames', 0) + 1
                else:
                    # Время вышло - удаляем
                    circles_to_remove.add(circle.id)
        else:
            used = [False] * len(detection_results)
            
            for circle in self.circles:
                cx, cy, _ = circle.last_data
                best_idx, min_dist = -1, self.search_radius
                
                for i, (nx, ny, nr, _, _) in enumerate(detection_results):
                    if used[i]: continue
                    dist = np.linalg.norm([cx - nx, cy - ny])
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i

                if best_idx != -1:
                    # НАШЛИ: Обновляем данные и сбрасываем счетчик потери
                    x, y, r, roi, mask = detection_results[best_idx]
                    circle.update((x, y, r))
                    circle.last_roi = roi
                    circle.last_mask = mask
                    circle.lost_frames = 0 # Сброс, так как мы видим его
                    used[best_idx] = True
                else:
                    # ПОТЕРЯЛИ: Проверяем память
                    persistence = getattr(circle, 'persistence_timer', 0)
                    if persistence > 0:
                        circle.persistence_timer -= 1
                        circle.lost_frames = getattr(circle, 'lost_frames', 0) + 1
                    else:
                        # Время вышло - удаляем
                        circles_to_remove.add(circle.id)

            # Добавляем новые
            for i, (x, y, r, roi, mask) in enumerate(detection_results):
                if not used[i]:
                    c = Circle(self.next_circle_id, pos=(x, y), radius=r)
                    c.last_roi = roi
                    c.last_mask = mask
                    c.lost_frames = 0
                    self.circles.append(c)
                    self.next_circle_id += 1
        
        # --- ГЕНЕРАЦИЯ СОБЫТИЙ УДАЛЕНИЯ (Removed) ---
        for circle in self.circles:
            if circle.id in circles_to_remove:
                # Если удаляемый круг был НЕ "free" и НЕ "Unknown", пишем Removed
                if circle.name and circle.name != "free" and circle.name != "Unknown":
                    # Рисуем минус на последней известной позиции
                    if circle.last_data is not None:
                        x, y, _ = circle.last_data
                        self.add_visual_event('minus', (x, y), label=circle.name, circle_id=circle.id)

        # --- ФИЗИЧЕСКОЕ УДАЛЕНИЕ ---
        # Оставляем только те круги, которых нет в списке на удаление
        self.circles = [c for c in self.circles if c.id not in circles_to_remove]

    def check_state_change(self, circle, old_name):
        new_name = circle.name
        if not new_name: return

        last_time = self.last_event_times.get(circle.id, -999)
        if self.frame_counter - last_time < 30: 
            return

        if circle.last_data is None: return
        current_pos = (circle.last_data[0], circle.last_data[1])

        def is_occupied(name):
            return name is not None and name != "free" and name != "Unknown"
        def is_free(name):
            return name is None or name == "free" or name == "Unknown"

        if is_free(old_name) and is_occupied(new_name):
            self.add_visual_event('plus', current_pos, label=new_name, circle_id=circle.id)

        if is_occupied(old_name) and is_free(new_name):
            self.add_visual_event('minus', current_pos, label=old_name, circle_id=circle.id)

    def draw_visual_events(self, image):
        for event in self.visual_events:
            x, y = event['pos']
            timer = event['timer']
            color = (255, 255, 255) 
            thickness = 3
            
            if event['type'] == 'plus':
                sz = 15
                cv2.line(image, (x - sz, y), (x + sz, y), color, thickness) 
                cv2.line(image, (x, y - sz), (x, y + sz), color, thickness) 
            elif event['type'] == 'minus':
                sz = 15
                cv2.line(image, (x - sz, y), (x + sz, y), color, thickness)

            event['timer'] -= 1
        
        self.visual_events = [e for e in self.visual_events if e['timer'] > 0]

        if self.notification_timer > 0 and self.notification_message:
            cv2.rectangle(image, (5, 5), (400, 45), (0, 0, 0), -1)
            cv2.putText(image, self.notification_message, (15, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.notification_timer -= 1
        else:
            self.notification_message = ""