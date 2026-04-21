from __future__ import annotations

from typing import Literal


class Stuff:
    def __init__(self, obj_uri_or_prebuilt_name: str, name: str | None = None) -> None:
        self.name = name
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Robot:
    def __init__(self, obj_uri_or_prebuilt_name: str, name: str | None = None) -> None:
        self.name = name
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Camera:
    def __init__(
        self,
        name: str,
        type: Literal[
            "rgb", "tactile", "depth", "pointcloud", "normal", "segmentation"
        ] = "rgb",
        mode: Literal["fixed", "track"] = "track",
        lookat: tuple[float, float, float] = (0, 0, 0),
        fov: float = 50.0,
        aspect: float = 1.0,
        near: float = 100.0,
        far: float = 1000.0,
    ):
        self.name = name
        self.type = type
        self.mode = mode
        self.lookat = lookat
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far


class Light:
    def __init__(
        self,
        name: str,
        type: Literal["ambient", "pointlight", "reactarea", "spot"] = "spot",
        color: tuple[int, int, int] = (0xFF, 0xFF, 0xFF),
        intensity: float = 0.8,
    ):
        self.name = name
        self.type = type
        self.color = color
