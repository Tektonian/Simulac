class Stuff:
    def __init__(self, obj_uri_or_prebuilt_name: str, name: str = "") -> None:
        self.name = name
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Robot:
    def __init__(self, obj_uri_or_prebuilt_name: str, name: str = "") -> None:
        self.name = name
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Camera:
    def __init__(self):
        """Not implemented yet"""


class Light:
    def __init__(self):
        """Not implemented yet"""
