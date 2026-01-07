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
        if image_roi is None or image_roi.size == 0: return None
        
        if len(image_roi.shape) == 3:
            gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        else: return None 

        if mask is None:
            h, w = gray.shape[:2]
            mask = np.zeros((h, w), dtype="uint8")
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2 - 12, 255, -1)

        # SIFT
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

        # Lab part
        lab_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2Lab)
        mean_lab = cv2.mean(lab_image, mask=mask)[:3]
        L, a, b = mean_lab
        
        if base_name == 'pig':
            h, w = gray.shape[:2]
            cx, cy = w // 2, h // 2
            r_outer = min(w, h) // 2 - 12
            half_side = int(r_outer / np.sqrt(2))
            line_top = cy - half_side
            line_bot = cy + half_side
            line_left = cx - half_side
            line_right = cx + half_side
            masks_8 = []
            
            m = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(m, (line_left, 0), (line_right, line_top), 255, -1)
            masks_8.append(cv2.bitwise_and(m, mask))
            m = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(m, (line_left, line_bot), (line_right, h), 255, -1)
            masks_8.append(cv2.bitwise_and(m, mask))
            m = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(m, (0, line_top), (line_left, line_bot), 255, -1)
            masks_8.append(cv2.bitwise_and(m, mask))
            m = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(m, (line_right, line_top), (w, line_bot), 255, -1)
            masks_8.append(cv2.bitwise_and(m, mask))
            m = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(m, (0, 0), (line_left, line_top), 255, -1)
            masks_8.append(cv2.bitwise_and(m, mask))
            m = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(m, (line_right, 0), (w, line_top), 255, -1)
            masks_8.append(cv2.bitwise_and(m, mask))
            m = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(m, (0, line_bot), (line_left, h), 255, -1)
            masks_8.append(cv2.bitwise_and(m, mask))
            m = np.zeros((h, w), dtype="uint8")
            cv2.rectangle(m, (line_right, line_bot), (w, h), 255, -1)
            masks_8.append(cv2.bitwise_and(m, mask))

            blue_zones = 0            
            for zone_mask in masks_8:
                if cv2.countNonZero(zone_mask) < 20: continue
                
                mean_val = cv2.mean(lab_image, mask=zone_mask)
                b = mean_val[2]
                if b < 120:
                    blue_zones += 1
            if blue_zones >= 5:
                return "pig_blue"
            if blue_zones >= 1:
                return "pig"
            return "pig_red"
        

        elif base_name == 'horse':
            if b < 120: 
                return "horse_blue"
            return "horse"

        else:
            if b < 118:
                candidate = f"{base_name}_blue"
                return candidate if candidate in self.references else base_name
            if a >= 127:
                candidate = f"{base_name}_red"
                return candidate if candidate in self.references else base_name
            else:
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
