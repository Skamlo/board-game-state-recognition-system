from modules.object_detection.objects import Object


class Circle(Object):
    def __init__(self, obj_id, pos=None, radius=None, max_lost=30):
        super().__init__(obj_id, max_lost)
        if pos is not None and radius is not None:
            self.last_data = (pos[0], pos[1], radius)
            self.is_visible = True

        self.last_roi = None
        self.last_mask = None
