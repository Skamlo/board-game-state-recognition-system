import numpy as np
import cv2
from collections import Counter
from modules.object_detection.objects import Circle

class BoardLogic:
    def __init__(self, target_size=600, search_radius=45):
        self.target_size = target_size
        self.search_radius = search_radius

        self.circles = []
        self.next_circle_id = 0

        self.visual_events = []         
        self.frame_counter = 0          
        self.last_event_times = {}      
        self.notification_queue = [] 

        self.snapshot_counts = None 
        self.stability_timer = 0
        self.last_frame_counts = Counter()
        self.manipulation_freeze_timer = 0

        self.points_map = {
            'rabbit': 1, 'sheep': 6, 'pig': 12, 
            'cow': 36, 'horse': 72, 'small_dog': 6, 'big_dog': 36
        }

    def add_visual_event(self, event_type, pos, label=None, circle_id=None):
        self.visual_events.append({
            'type': event_type,      
            'pos': (int(pos[0]), int(pos[1])),
            'timer': 45              
        })
        if circle_id is not None:
            self.last_event_times[circle_id] = self.frame_counter

    def add_notification(self, text, color=(0, 255, 255), duration=90):
        if not any(n[0] == text for n in self.notification_queue):
            self.notification_queue.append([text, color, duration])

    def update_circles(self, detection_results):
        self.frame_counter += 1
        
        if not detection_results:
            for circle in self.circles:
                persistence = getattr(circle, 'persistence_timer', 0)
                if persistence > 0:
                    circle.persistence_timer -= 1
                    circle.lost_frames = getattr(circle, 'lost_frames', 0) + 1
                else:
                    circle.update(None)
        else:
            used = [False] * len(detection_results)
            for circle in self.circles:
                if not (circle.is_visible or getattr(circle, 'persistence_timer', 0) > 0):
                    continue
                
                if circle.last_data is None: continue
                cx, cy, _ = circle.last_data

                best_idx, min_dist = -1, self.search_radius
                for i, (nx, ny, nr, _, _) in enumerate(detection_results):
                    if used[i]: continue
                    dist = np.linalg.norm([cx - nx, cy - ny])
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i

                if best_idx != -1:
                    x, y, r, roi, mask = detection_results[best_idx]
                    circle.update((x, y, r))
                    circle.last_roi = roi
                    circle.last_mask = mask
                    circle.lost_frames = 0 
                    used[best_idx] = True
                else:
                    persistence = getattr(circle, 'persistence_timer', 0)
                    if persistence > 0:
                        circle.persistence_timer -= 1
                        circle.lost_frames = getattr(circle, 'lost_frames', 0) + 1
                    else:
                        circle.update(None)

            for i, (x, y, r, roi, mask) in enumerate(detection_results):
                if not used[i]:
                    c = Circle(self.next_circle_id, pos=(x, y), radius=r)
                    c.last_roi = roi
                    c.last_mask = mask
                    c.lost_frames = 0
                    self.circles.append(c)
                    self.next_circle_id += 1
        
        self.circles = [c for c in self.circles if c.is_visible or getattr(c, 'persistence_timer', 0) > 0]
        
        self._process_game_state()

    def _get_current_counts(self):
        counts = Counter()
        for c in self.circles:
            is_active = c.is_visible or (getattr(c, 'persistence_timer', 0) > 0)
            if is_active and c.name and c.name not in ["free", "Unknown", None]:
                counts[c.name] += 1
        return counts

    def _process_game_state(self):
        curr_counts = self._get_current_counts()

        if self.snapshot_counts is None:
            self.snapshot_counts = curr_counts
            self.last_frame_counts = curr_counts
            return

        total_curr = sum(curr_counts.values())
        total_snap = sum(self.snapshot_counts.values())
        
        if (total_snap - total_curr) >= 3:
            if self.manipulation_freeze_timer == 0 or total_curr < sum(self.last_frame_counts.values()):
                self.manipulation_freeze_timer = 50

        if self.manipulation_freeze_timer > 0:
            self.manipulation_freeze_timer -= 1
            self.last_frame_counts = curr_counts
            return 

        has_changed_since_last_frame = (curr_counts != self.last_frame_counts)

        if has_changed_since_last_frame:
            self.stability_timer = 0
        else:
            self.stability_timer += 1

        self.last_frame_counts = curr_counts 

        TARGET_STABILITY = 20 
        
        if self.stability_timer >= TARGET_STABILITY:
            if curr_counts != self.snapshot_counts:
                self._resolve_changes(self.snapshot_counts, curr_counts)
                self.snapshot_counts = curr_counts
                self.stability_timer = 0 

        required = {'rabbit', 'sheep', 'pig', 'cow', 'horse'}
        if all(curr_counts[anim] >= 1 for anim in required):
             if "WINNER!" not in [n[0] for n in self.notification_queue]:
                 self.add_notification("WINNER!", color=(0, 255, 0), duration=120)

    def _resolve_changes(self, old_counts, new_counts):
        delta = {}
        all_keys = set(old_counts.keys()) | set(new_counts.keys())
        
        lost_items = []
        gained_items = []
        
        value_lost = 0
        value_gained = 0
        
        for k in all_keys:
            diff = new_counts[k] - old_counts[k]
            if diff != 0:
                delta[k] = diff
                val = self.points_map.get(k, 0)
                
                if diff < 0:
                    lost_items.append(f"{abs(diff)} {k}")
                    value_lost += abs(diff) * val
                else:
                    gained_items.append(f"{diff} {k}")
                    value_gained += diff * val

        total_old_animals = sum(old_counts.values())
        total_new_animals = sum(new_counts.values())
        
        if total_old_animals > 0 and total_new_animals == 0:
            self.add_notification("WOLF ATTACK!", (0, 0, 255), 120)
            return
        
        if delta.get('rabbit', 0) < 0 and new_counts.get('rabbit', 0) == 0: 
            if 'sheep' not in str(gained_items) and 'pig' not in str(gained_items):
                self.add_notification("FOX ATTACK!", (0, 165, 255), 120)
                return
                 

        if len(lost_items) > 0 and len(gained_items) > 0:
            
            TRADE_TOLERANCE = 0.4
            
            diff_value = abs(value_lost - value_gained)
            
            if diff_value / (value_gained + 0.1) <= TRADE_TOLERANCE:
                lost_str = ", ".join(lost_items)
                gained_str = ", ".join(gained_items)
                
                color = (200, 200, 200)
                if value_gained > 12: color = (255, 215, 0)
                
                self.add_notification(f"Trade: {lost_str} -> {gained_str}", color, 120)
            
            else:
                lost_str = ", ".join(lost_items)
                gained_str = ", ".join(gained_items)
                
                self.add_notification(f"Change: {lost_str} -> {gained_str} (Diff: {value_gained - value_lost})", (100, 100, 255), 120)
            
        elif len(gained_items) > 0:
            self.add_notification(f"Born: {', '.join(gained_items)}", (0, 255, 0), 60)

    def check_state_change(self, circle, old_name):
        new_name = circle.name
        if not new_name: return

        last_time = self.last_event_times.get(circle.id, -999)
        if self.frame_counter - last_time < 30: return
        if circle.last_data is None: return
        
        pos = (circle.last_data[0], circle.last_data[1])
        
        def is_occ(n): return n and n not in ["free", "Unknown"]
        def is_fr(n): return not n or n in ["free", "Unknown"]

        if is_fr(old_name) and is_occ(new_name):
            self.add_visual_event('plus', pos, label=new_name, circle_id=circle.id)
        if is_occ(old_name) and is_fr(new_name):
            self.add_visual_event('minus', pos, label=old_name, circle_id=circle.id)

    def draw_visual_events(self, image):
        for event in self.visual_events:
            x, y = event['pos']
            color = (255, 255, 255) 
            thickness = 3
            sz = 15
            if event['type'] == 'plus':
                cv2.line(image, (x - sz, y), (x + sz, y), color, thickness) 
                cv2.line(image, (x, y - sz), (x, y + sz), color, thickness) 
            elif event['type'] == 'minus':
                cv2.line(image, (x - sz, y), (x + sz, y), color, thickness)
            event['timer'] -= 1
        self.visual_events = [e for e in self.visual_events if e['timer'] > 0]

        y_start = 300
        h, w = image.shape[:2]
        
        active_notes = []
        for i, note in enumerate(self.notification_queue):
            text, color, timer = note
            if timer <= 0: continue
            
            font_scale = 1.0
            if len(text) > 25: font_scale = 0.6 
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = y_start + (i * 40)
            
            overlay = image.copy()
            cv2.rectangle(overlay, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
            
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            
            note[2] -= 1
            active_notes.append(note)
            if i >= 4: break 
            
        self.notification_queue = active_notes