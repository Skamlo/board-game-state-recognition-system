import cv2
import os
import numpy as np


class TokenClassifier:
    def __init__(self, references_path):
        self.orb = cv2.SIFT_create()  #cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  #cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.references = {} 
        
        self._load_references(references_path)

    def _load_references(self, path):
        if not os.path.exists(path):
            print(f"Warning: Path {path} not found!")
            return

        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype="uint8")
            
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 2
            if radius <= 0: radius = 1
            cv2.circle(mask, center, radius, 255, -1)

            kp, des = self.orb.detectAndCompute(img, mask=mask)
            
            if des is not None:
                name = os.path.splitext(filename)[0]        
                self.references[name] = {
                    'des': des,
                    'img': img
                }
                print(f"Loaded reference: {name} ({len(kp)} features)")

    def predict(self, image_roi, mask=None):
        
        if len(image_roi.shape) == 3:
            gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_roi

        if mask is None:
            h, w = gray.shape[:2]
            mask = np.zeros((h, w), dtype="uint8")
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)

        kp, des = self.orb.detectAndCompute(gray, mask=mask)
        
        if des is None or len(des) < 5:
            return None 

        best_match_name = None
        max_good_matches = 0
        
        for name, data in self.references.items():
            ref_des = data['des']
            if ref_des is None: continue
            
            matches = self.matcher.knnMatch(des, ref_des, k=2)
            
            good = []
            for m, n in matches:
                if m.distance < 0.85 * n.distance:
                    good.append(m)
            
            max_good_matches = len(good)
            best_match_name = name

        if max_good_matches > 4: 
            return best_match_name
        
        return "Unknown"

    def get_masked_references_images(self):
        visuals = []
        for name, data in self.references.items():
            img = data['img'] 
            
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype="uint8")
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2 - 2, 255, -1)
            
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            
            masked_img_color = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(masked_img_color, name, (5, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            visuals.append(masked_img_color)
            
        return visuals
