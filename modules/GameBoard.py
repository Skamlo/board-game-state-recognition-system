import cv2
import numpy as np
from modules.Object import TrackedObject
from modules.Circle import Circle

class GameBoard(TrackedObject):
    def __init__(self, obj_id, target_size=600, max_lost=50):
        super().__init__(obj_id, max_lost)
        self.target_size = target_size
        self.circles = [] 
        self.next_circle_id = 0
        self.relative_zones = [
            (0.14, 0.16), (0.32, 0.15), (0.50, 0.14), (0.68, 0.15), (0.86, 0.16),
            (0.22, 0.35), (0.40, 0.33), (0.59, 0.33), (0.77, 0.35),
            (0.31, 0.52), (0.50, 0.50), (0.69, 0.52),
            (0.41, 0.69), (0.59, 0.69),
            (0.50, 0.86)
        ] # local points on a board
        self.name = "board"
        self.search_radius = 45
        self.zones = []
        for (rx, ry) in self.relative_zones:
            self.zones.append((int(rx * self.target_size), int(ry * self.target_size)))
            
        self.search_radius = 45
    
    
    def get_warped(self, frame):
        if not self.is_visible or self.last_data is None:
            return None
        
        pts = self.last_data.reshape(4, 2).astype("float32")
        
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

    def update_circles(self, detection_results):
        if not detection_results:
            for circle in self.circles:
                circle.update(None)
        else:
            used_detections = [False] * len(detection_results)
            
            for circle in self.circles:
                if not circle.is_visible: continue

                best_idx = -1
                min_dist = 50 
                cx, cy, _ = circle.last_data
                for i, (nx, ny, nr, n_roi, n_mask) in enumerate(detection_results):
                    if used_detections[i]: continue
                    
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array([nx, ny]))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                
                if best_idx != -1:
                    x, y, r, roi, mask = detection_results[best_idx]
                    circle.update((x, y, r))
                    
                    circle.last_roi = roi
                    circle.last_mask = mask
                    
                    used_detections[best_idx] = True
                else:
                    circle.update(None)
            
            for i, (nx, ny, nr, n_roi, n_mask) in enumerate(detection_results):
                if not used_detections[i]:
                    new_circle = Circle(self.next_circle_id, pos=(nx, ny), radius=nr)
                    
                    new_circle.last_roi = n_roi
                    new_circle.last_mask = n_mask
                    
                    self.circles.append(new_circle)
                    self.next_circle_id += 1
        


    def draw_circles(self, image):
        for circle in self.circles:
            if circle.is_visible:
                circle.draw(image)
        cv2.putText(image, f"Count: {len([c for c in self.circles if c.is_visible])}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)