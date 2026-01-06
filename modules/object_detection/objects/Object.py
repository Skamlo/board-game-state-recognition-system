class Object:
    def __init__(self, obj_id, max_lost=50):
        self.id = obj_id
        self.lost_frames = 0
        self.max_lost = max_lost
        self.is_visible = False
        self.last_data = None
        self.name = None
        self.frames_since_recognition = 0

    def update(self, detected_data):
        if detected_data is not None:
            self.last_data = detected_data
            self.lost_frames = 0
            self.is_visible = True
            return True
        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost:
                self.is_visible = False
            return self.is_visible
