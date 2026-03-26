from tt.sdk import obtain_runtime


class Stuff:
    def __init__(self, obj_uri_or_prebuilt_name: str, name: str = "") -> None:
        world_maker = obtain_runtime().world_maker
        self._entity = world_maker.create_entity(name, obj_uri_or_prebuilt_name, "", "")


class Robot:
    def __init__(self, obj_uri_or_prebuilt_name: str, name: str = "") -> None:
        world_maker = obtain_runtime().world_maker
        self._entity = world_maker.create_entity(name, obj_uri_or_prebuilt_name, "", "")


class Camera:
    def __init__(self):
        """Not implemented yet"""


class Light:
    def __init__(self):
        """Not implemented yet"""
