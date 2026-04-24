from __future__ import annotations

from typing import Generic, Literal, Sequence, TypeVar, Union

from typing_extensions import TypeVar

from .randomize import RandomizableFloat, RandomizableVec3

ActionT = TypeVar("ActionT", bound=Sequence[float], default=list[float])


class Stuff:
    def __init__(self, obj_uri_or_prebuilt_name: str, name: str | None = None) -> None:
        self.name = name
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Robot(Generic[ActionT]):
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
        lookat: Union[RandomizableVec3, tuple[float, float, float]] = (0, 0, 0),
        fov: Union[RandomizableFloat, float] = 50.0,
        aspect: Union[RandomizableFloat, float] = 1.0,
        near: Union[RandomizableFloat, float] = 100.0,
        far: Union[RandomizableFloat, float] = 1000.0,
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
        color: Union[RandomizableVec3, tuple[int, int, int]] = (0xFF, 0xFF, 0xFF),
        intensity: Union[RandomizableFloat, float] = 0.8,
    ):
        self.name = name
        self.type = type
        self.color = color
        self.intensity = intensity
