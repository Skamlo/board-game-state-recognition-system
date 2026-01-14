import cv2
import numpy as np
from collections import deque, Counter

from modules.object_detection.objects import Board
from modules.game_engine import BoardLogic
from modules.object_detection import (
    find_circles_hough,
    TokenClassifier,
    find_dices_geometry
)
from modules.video import (
    VideoReadManager,
    VideoWriteManager,
    draw_board,
    draw_circle,
    draw_dices,
    apply_lighting_correction
)

class Pipeline:
    def __init__(self, config):
        self.cfg = config
        self.classifier = None
        self.tracked_boards = []
        self.board_id_counter = 0
        self.frame_count = 0

        self.points_map = {
            'rabbit': 1, 'sheep': 6, 'pig': 12,
            'cow': 36, 'horse': 72, 'small_dog': 0, 'big_dog': 0
        }

        self._init_components()

    def _init_components(self):
        self.classifier = TokenClassifier(self.cfg['ELEMENTS_PATH'])
        cv2.namedWindow("Main Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Main Stream", 1000, 700)

    def run(self):
        paused = False

        with VideoReadManager(self.cfg['VIDEO_PATH']) as reader, \
             VideoWriteManager(self.cfg['OUTPUT_VIDEO_PATH'], fps=30) as writer:

            for frame in reader.read():
                if paused:
                    key = cv2.waitKey(100)
                    if key == ord(' '):
                        paused = not paused
                    if key == ord('q'):
                        break
                    continue

                self.frame_count += 1

                processed_frame, board_views = self.process_frame(frame)

                writer.write_composite(processed_frame, board_views)

                cv2.imshow("Main Stream", processed_frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused

        cv2.destroyAllWindows()

    def process_frame(self, frame):
        frame = apply_lighting_correction(frame)
        candidates = self._find_boards_geometry(frame)
        self._update_tracked_boards(candidates)

        current_board_views = []

        for board in self.tracked_boards:
            if not board.is_visible:
                continue

            draw_board(frame, board, color=(0, 255, 0))
            warped = board.get_warped(frame)
            if warped is None:
                continue

            self._process_board_logic(board, warped)

            board.logic.draw_visual_events(warped)
            warped_with_stats = self._draw_stats_panel(warped, board.logic.circles)
            current_board_views.append(warped_with_stats)

            cv2.imshow(f"Board {board.id+1}", warped)

        dices = find_dices_geometry(frame)
        draw_dices(frame, dices, color=(0, 165, 255))
        return frame, current_board_views

    def _process_board_logic(self, board, warped_img):
        detection_results = find_circles_hough(warped_img, self.cfg["DIFFICULTY"])

        board.logic.update_circles(detection_results)

        for circle in board.logic.circles:
            if circle.is_visible:
                if not hasattr(circle, 'pred_history'):
                    circle.pred_history = deque(maxlen=self.cfg['HISTORY_LEN'] + 2)

                if self.frame_count % self.cfg['SKIP_FRAMES'] == 0:
                    raw_pred = "Unknown"
                    if getattr(circle, 'lost_frames', 0) == 0:
                        if hasattr(circle, 'last_roi') and circle.last_roi is not None:
                            p = self.classifier.predict(circle.last_roi, mask=circle.last_mask)
                            if p:
                                raw_pred = p

                        if raw_pred != "Unknown":
                            circle.pred_history.append(raw_pred)
                            if raw_pred != "free":
                                circle.pred_history.append(raw_pred)

                if len(circle.pred_history) > 0:
                    most_common, count = Counter(circle.pred_history).most_common(1)[0]
                    old_name = circle.name

                    threshold = self.cfg['CONFIDENCE_THRESH']
                    is_currently_occupied = (old_name not in [None, "free", "Unknown"])

                    if is_currently_occupied and most_common == "free":
                        threshold += 2
                    elif not is_currently_occupied and most_common != "free":
                        threshold = max(1, threshold - 1)

                    if count >= threshold:
                        circle.name = most_common

                    if count >= self.cfg['PERSISTENCE_THRESH'] and most_common != "free":
                        circle.persistence_timer = self.cfg['PERSISTENCE_FRAMES']

                    board.logic.check_state_change(circle, old_name)

                self._draw_token_on_board(warped_img, circle)

    def _draw_token_on_board(self, image, circle):
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

        draw_circle(image, circle, label=label_text, color=display_color)

    def _draw_stats_panel(self, image, circles):
        if image is None:
            return None
        h, w = image.shape[:2]
        panel_w = 220

        valid_names = []
        for c in circles:
            is_active = c.is_visible or (getattr(c, 'persistence_timer', 0) > 0)
            if is_active and c.name and c.name not in ["free", "Unknown", None]:
                valid_names.append(c.name)
        counts = Counter(valid_names)

        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        cv2.putText(panel, "SCORE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.line(panel, (10, 50), (panel_w - 10, 50), (100, 100, 100), 2)

        y_offset = 90
        total_score = 0

        for name in sorted(counts.keys()):
            count = counts[name]
            value = self.points_map.get(name, 0)
            total_score += count * value

            text = f"{name.upper()}: {count}"
            color = (255, 255, 255)
            if name == 'pig':
                color = (150, 150, 255)
            elif name == 'sheep':
                color = (200, 255, 200)
            elif name == 'cow':
                color = (200, 200, 255)
            elif name == 'horse':
                color = (100, 100, 255)

            cv2.putText(panel, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 40

        cv2.line(panel, (10, y_offset + 10), (panel_w - 10, y_offset + 10), (100, 100, 100), 1)
        cv2.putText(panel, f"TOTAL: {total_score}", (15, y_offset + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return np.hstack([panel, image])

    def _find_boards_geometry(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=100, sigmaSpace=100)
        edged = cv2.Canny(blurred, 10, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        edged = closed
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_boards = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                rect = cv2.minAreaRect(c)
                (x, y), (w, h), angle = rect
                if (self.cfg['MIN_SIDE'] <= w <= self.cfg['MAX_SIDE']) and \
                   (self.cfg['MIN_SIDE'] <= h <= self.cfg['MAX_SIDE']):
                    valid_boards.append(approx)
        return valid_boards

    def _update_tracked_boards(self, candidates):
        for cand in candidates:
            matched = False
            cand_center = np.mean(cand.reshape(4, 2), axis=0)

            for board in self.tracked_boards:
                if board.last_data is not None:
                    curr_center = np.mean(board.last_data.reshape(4, 2), axis=0)
                    if np.linalg.norm(cand_center - curr_center) < 100:
                        board.update(cand)
                        matched = True
                        break

            if not matched:
                new_board = Board(self.board_id_counter, target_size=self.cfg['TARGET_WARPED_SIZE'])
                new_board.update(cand)
                new_board.logic = BoardLogic(target_size=self.cfg['TARGET_WARPED_SIZE'])
                self.tracked_boards.append(new_board)
                self.board_id_counter += 1
