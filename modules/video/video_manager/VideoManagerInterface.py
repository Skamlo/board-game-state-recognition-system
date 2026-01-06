class VideoManagerInterface:
    def __init__(self, path:str, color_system:str="RGB"):
        self.path = path
        self.color_system = color_system
        self.connection = None
        self.is_connection_open = False

    def open_connection(self):
        raise NotImplementedError("Method not implemented. This class is only interface class.")

    def close_connection(self):
        raise NotImplementedError("Method not implemented. This class is only interface class.")

    def __del__(self):
        self.close_connection()

    def __enter__(self):
        self.open_connection()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close_connection()
