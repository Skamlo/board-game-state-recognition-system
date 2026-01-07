import numpy as np
import cv2
from modules.object_detection.objects import Circle


class BoardLogic:
    def __init__(self, target_size=600, search_radius=45):
        self.target_size = target_size
        self.search_radius = search_radius

        self.circles = []
        self.next_circle_id = 0

        self.notification = ""
        self.notification_timer = 0
        
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

    def update_circles(self, detection_results):
        visible_before = {c.id for c in self.circles if c.is_visible}
        
        if not detection_results:
            for circle in self.circles:
                circle.update(None)
        else:
            used = [False] * len(detection_results)

            for circle in self.circles:
                if not circle.is_visible:
                    continue

                cx, cy, _ = circle.last_data
                best_idx = -1
                min_dist = self.search_radius

                for i, (nx, ny, nr, _, _) in enumerate(detection_results):
                    if used[i]:
                        continue

                    dist = np.linalg.norm([cx - nx, cy - ny])
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i

                if best_idx != -1:
                    x, y, r, roi, mask = detection_results[best_idx]
                    circle.update((x, y, r))
                    circle.last_roi = roi
                    circle.last_mask = mask
                    used[best_idx] = True
                else:
                    circle.update(None)

            for i, (x, y, r, roi, mask) in enumerate(detection_results):
                if not used[i]:
                    c = Circle(self.next_circle_id, pos=(x, y), radius=r)
                    c.last_roi = roi
                    c.last_mask = mask
                    self.circles.append(c)
                    self.next_circle_id += 1
        
        visible_after = {c.id for c in self.circles if c.is_visible}
        added_ids = visible_after - visible_before
        removed_ids = visible_before - visible_after

        if added_ids or removed_ids:
            msgs = []
            if added_ids:
                msgs.append(f"Added : {list(added_ids)}")
            if removed_ids:
                msgs.append(f"Lost: {list(removed_ids)}")
            
            self.notification = " | ".join(msgs)
            self.notification_timer = 120 
        
        if self.notification_timer > 0:
            self.notification_timer -= 1
        else:
            self.notification = ""
        
    def draw_notification(self, image):
        if self.notification:
            cv2.rectangle(image, (10, 10), (600, 50), (0, 0, 0), -1)
            cv2.putText(image, self.notification, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)