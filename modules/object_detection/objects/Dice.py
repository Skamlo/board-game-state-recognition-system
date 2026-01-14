from modules.object_detection.objects import Object


class Dice(Object):
    def __init__(self, obj_id, box=None, color="Unknown", label="Unknown", contour=None, max_lost=50):
        super().__init__(obj_id, max_lost)
        self.box = box        # (x, y, w, h)
        self.color = color    # "Orange" or "Blue"
        self.label = label    # From TokenClassifier
        self.contour = contour
        
        if box is not None:
            self.center = (box[0] + box[2] // 2, box[1] + box[3] // 2)
        else:
            self.center = None