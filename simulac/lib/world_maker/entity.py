from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Sequence, Union

from typing_extensions import TypeVar

from simulac.sdk.environment_service.common.model.entity import (
    AmbientLightSpec,
    AreaLightSpec,
    CameraMode,
    CameraSpec,
    CameraType,
    LightSpec,
    PointLightSpec,
    SpotLightSpec,
)

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.randomize import (
        RandomizableColor,
        RandomizableFloat,
        RandomizableVec3,
    )

ActionT = TypeVar("ActionT", bound=Sequence[float], default=list[float])


class Stuff:
    def __init__(self, obj_uri_or_prebuilt_name: str) -> None:
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Robot(Generic[ActionT]):
    def __init__(self, obj_uri_or_prebuilt_name: str) -> None:
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Camera:
    def __init__(
        self,
        type: CameraType = "rgb",
        mode: CameraMode = "fixed",
        lookat: Union[RandomizableVec3, tuple[float, float, float]] = (0, 0, 0),
        fov: Union[RandomizableFloat, float] = 50.0,
        aspect: Union[RandomizableFloat, float] = 1.0,
        near: Union[RandomizableFloat, float] = 100.0,
        far: Union[RandomizableFloat, float] = 1000.0,
    ):
        self.__spec = CameraSpec(type, mode, lookat, fov, aspect, near, far)

    def _to_spec(self) -> CameraSpec:
        return self.__spec


@dataclass(frozen=True, slots=True)
class _Light:
    color: RandomizableColor = (255, 255, 255)
    intensity: RandomizableFloat = 0.8

    def _to_spec(self) -> LightSpec:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SpotLight(_Light):
    angle: RandomizableFloat = 45.0
    penumbra: RandomizableFloat = 0.0

    def _to_spec(self) -> SpotLightSpec:
        return SpotLightSpec(
            color=self.color,
            intensity=self.intensity,
            angle=self.angle,
            penumbra=self.penumbra,
        )


@dataclass(frozen=True, slots=True)
class AreaLight(_Light):
    width: RandomizableFloat = 1.0
    height: RandomizableFloat = 1.0

    def _to_spec(self) -> AreaLightSpec:
        return AreaLightSpec(
            color=self.color,
            intensity=self.intensity,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True, slots=True)
class AmbientLight(_Light):
    def _to_spec(self) -> AmbientLightSpec:
        return AmbientLightSpec(
            color=self.color,
            intensity=self.intensity,
        )


@dataclass(frozen=True, slots=True)
class PointLight(_Light):
    range: RandomizableFloat | None = None
    decay: RandomizableFloat = 2.0

    def _to_spec(self) -> PointLightSpec:
        return PointLightSpec(
            color=self.color,
            intensity=self.intensity,
            range=self.range,
            decay=self.decay,
        )


type LightType = SpotLight | AreaLight | AmbientLight | PointLight
