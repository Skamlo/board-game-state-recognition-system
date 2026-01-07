import cv2
import os
import numpy as np


class TokenClassifier:
    def __init__(self, references_path):
        self.orb = cv2.SIFT_create()  #cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  #cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.references = {} 
        
        self._load_references(references_path)

    def _calc_histogram(self, image, mask):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 12, 3], [0, 180, 0, 256, 0, 256])
        if cv2.countNonZero(mask) > 0:
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()
    def _calc_average_color(self, image, mask):
        """
        Recounts Lab.
        returns numpy array [L, a, b].
        """
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        mean_val = cv2.mean(lab_image, mask=mask)[:3]
        
        return np.array(mean_val, dtype=float)
    
    def _load_references(self, path):
        if not os.path.exists(path):
            print(f"Warning: Path {path} not found!")
            return

        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img_color = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype="uint8")
            
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2 - 12, 255, -1)

            kp, des = self.orb.detectAndCompute(img, mask=mask)
            avg_color = self._calc_average_color(img_color, mask)
            hist = self._calc_histogram(img_color, mask)
            if des is not None:
                name = os.path.splitext(filename)[0]        
                self.references[name] = {
                    'des': des,
                    'img': img_gray,
                    'hist': hist,
                    'color': avg_color
                }
                print(f"Loaded reference: {name} ({len(kp)} features)")
    
    def predict(self, image_roi, mask=None):
        # --- 1. БАЗОВЫЕ ПРОВЕРКИ ---
        if image_roi is None or image_roi.size == 0: return None
        
        if len(image_roi.shape) == 3:
            gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        else: return None 

        h, w = gray.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Радиус рабочей области
        r_outer = min(w, h) // 2 - 12

        if mask is None:
            mask = np.zeros((h, w), dtype="uint8")
            cv2.circle(mask, (cx, cy), r_outer, 255, -1)

        # --- 2. SIFT (ОПРЕДЕЛЕНИЕ ТИПА ЖИВОТНОГО) ---
        kp, des = self.orb.detectAndCompute(gray, mask=mask)
        if des is None or len(des) < 10: return None 

        best_sift_name = None
        max_matches = 0
        for name, data in self.references.items():
            ref_des = data['des']
            if ref_des is None: continue
            matches = self.matcher.knnMatch(des, ref_des, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) > max_matches:
                max_matches = len(good)
                best_sift_name = name

        if max_matches < 10: return "Unknown"
        
        base_name = best_sift_name.split('_')[0]

        # --- 3. АНАЛИЗ ЦВЕТА (LAB) ---
        lab_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2Lab)
        
        # ==========================================
        # ЛОГИКА ДЛЯ СВИНЬИ (PIG) - 8 ЗОН
        # ==========================================
        if base_name == 'pig':
            half_side = int(r_outer / np.sqrt(2))
            lines = {
                'top': cy - half_side, 'bot': cy + half_side,
                'left': cx - half_side, 'right': cx + half_side
            }

            zones_rects = [
                ((lines['left'], 0), (lines['right'], lines['top'])),       # N
                ((lines['left'], lines['bot']), (lines['right'], h)),       # S
                ((0, lines['top']), (lines['left'], lines['bot'])),         # W
                ((lines['right'], lines['top']), (w, lines['bot'])),        # E
                ((0, 0), (lines['left'], lines['top'])),                    # NW
                ((lines['right'], 0), (w, lines['top'])),                   # NE
                ((0, lines['bot']), (lines['left'], h)),                    # SW
                ((lines['right'], lines['bot']), (w, h))                    # SE
            ]

            blue_zones = 0
            for (pt1, pt2) in zones_rects:
                m_zone = np.zeros((h, w), dtype="uint8")
                cv2.rectangle(m_zone, pt1, pt2, 255, -1)
                final_mask = cv2.bitwise_and(m_zone, mask)
                
                if cv2.countNonZero(final_mask) < 20: continue
                
                mean_val = cv2.mean(lab_image, mask=final_mask)
                b = mean_val[2]
                
                if b < 120: blue_zones += 1

            # --- РЕШЕНИЕ (Возвращаем free или pig) ---
            
            # pig_blue (Синяя) -> Free
            if blue_zones >= 5: return "free"
            
            # pig (Обычная, есть небо) -> Занято
            if blue_zones >= 1: return "pig"
            
            # pig_red (Сепия, нет неба) -> Free
            return "free"

        # ==========================================
        # ЛОГИКА ДЛЯ ЛОШАДИ (HORSE)
        # ==========================================
        elif base_name == 'horse':
            mean_lab = cv2.mean(lab_image, mask=mask)[:3]
            b = mean_lab[2]
            
            # horse_blue -> Free
            if b < 125: return "free"
            
            # Обычная лошадь
            return "horse"
        elif base_name == 'cow':
            mean_lab = cv2.mean(lab_image, mask=mask)[:3]
            a, b = mean_lab[1], mean_lab[2]
            
            # _blue (Синий вариант) -> Free
            if b < 122: return "free"
            
            # _red (Красный вариант, a >= 127) -> Free
            if a >= 127: return "free"
            
            # Обычный вариант (кролик, овца, корова)
            return base_name
        
        # ==========================================
        # ЛОГИКА ДЛЯ ОСТАЛЬНЫХ (С ТРАВОЙ)
        # ==========================================
        else:
            mean_lab = cv2.mean(lab_image, mask=mask)[:3]
            a, b = mean_lab[1], mean_lab[2]
            
            # _blue (Синий вариант) -> Free
            if b < 118: return "free"
            
            # _red (Красный вариант, a >= 127) -> Free
            if a >= 127: return "free"
            
            # Обычный вариант (кролик, овца, корова)
            return base_name  
    

    def get_masked_references_images(self):
        visuals = []
        for name, data in self.references.items():
            img = data['img'] 
            
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype="uint8")
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2 - 12, 255, -1)
            
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            
            masked_img_color = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(masked_img_color, name, (5, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            visuals.append(masked_img_color)
            
        return visuals
