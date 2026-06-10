from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Sequence, Union

from typing_extensions import TypeVar

from simulac.sdk.environment_service.common.model.entity import (
    AmbientLightSpec,
    AreaLightSpec,
    CameraMode,
    CameraSpec,
    LightSpec,
    PointLightSpec,
    SpotLightSpec,
    TCameraType,
)

if TYPE_CHECKING:
    from simulac.base.types.geometry import Vec3
    from simulac.sdk.environment_service.common.randomize import (
        RandomizableColor,
        RandomizableFloat,
        RandomizableVec3,
    )

ActionT = TypeVar("ActionT", bound=Sequence[float], default=list[float])


class Stuff:
    """Build-time asset descriptor for a non-robot object.

    See https://tektonian.com/asset for the prebuilt asset list.

    Args:
        obj_uri_or_prebuilt_name:
            Asset URI or registered prebuilt asset name.
    """

    def __init__(self, obj_uri_or_prebuilt_name: str) -> None:
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Robot(Generic[ActionT]):
    """Build-time asset descriptor for an controllable robot.

    See https://tektonian.com/asset for the prebuilt asset list.
    Args:
        obj_uri_or_prebuilt_name: Asset URI or registered prebuilt asset name.

    """

    def __init__(self, obj_uri_or_prebuilt_name: str) -> None:
        self.obj_uri_or_prebuilt_name = obj_uri_or_prebuilt_name


class Camera(Generic[TCameraType]):
    """Build-time camera descriptor converted into an SDK CameraSpec.

    Args:
        type: Camera output type.
        mode: Camera tracking mode.
        lookat: Default target point used by fixed/look-at camera modes.
        fov: Vertical field of view.
        aspect: Camera aspect ratio.
        near: Near clipping plane.
        far: Far clipping plane.

    Image::
        ![image](https://postfiles.pstatic.net/20140323_62/jidon333_13955736063040iGSH_PNG/FOV.png?type=w1)
    """

    def __init__(
        self,
        type: "TCameraType" = "rgb",
        mode: CameraMode = "fixed",
        lookat: Vec3 = (0, 0, 0),
        fov: float = 50.0,
        aspect: float = 1.0,
        near: float = 100.0,
        far: float = 1000.0,
    ):
        self.__spec = CameraSpec(type, mode, lookat, fov, aspect, near, far)

    def _to_spec(self) -> CameraSpec:
        """Return the SDK camera spec for this descriptor.

        Returns:
            CameraSpec used by the environment service.
        """
        return self.__spec


@dataclass(frozen=True, slots=True)
class _Light:
    """Base build-time light descriptor.

    Attributes:
        color: Public RGB color tuple or randomizable color spec.
        intensity: Public light intensity or randomizable intensity spec.
    """

    color: RandomizableColor = (255, 255, 255)
    intensity: RandomizableFloat = 0.8

    def _to_spec(self) -> LightSpec:
        """Return the SDK light spec for this descriptor.

        Raises:
            NotImplementedError: Always raised by the abstract base descriptor.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SpotLight(_Light):
    """Spot light descriptor with cone angle and penumbra.

    Diagram::

        [ SpotLight ]
               o
              / \\
             /   \\
            /     \\
        +---------------+
        |    object     |
        +---------------+

    Attributes:
        angle: Full cone angle in degrees.
        penumbra: Edge softness value.
    """

    angle: RandomizableFloat = 45.0
    penumbra: RandomizableFloat = 0.0

    def _to_spec(self) -> SpotLightSpec:
        """Return the SDK spot light spec for this descriptor.

        Returns:
            SpotLightSpec used by the environment service.
        """
        return SpotLightSpec(
            color=self.color,
            intensity=self.intensity,
            angle=self.angle,
            penumbra=self.penumbra,
        )


@dataclass(frozen=True, slots=True)
class AreaLight(_Light):
    """Rectangular area light descriptor.

    Diagram::

        [ AreaLight ]
        +-----------+
        | light box |
        +-----------+
          v   v   v
        +---------------+
        |    object     |
        +---------------+

    Attributes:
        width: Requested light width.
        height: Requested light height.
    """

    width: RandomizableFloat = 1.0
    height: RandomizableFloat = 1.0

    def _to_spec(self) -> AreaLightSpec:
        """Return the SDK area light spec for this descriptor.

        Returns:
            AreaLightSpec used by the environment service.
        """
        return AreaLightSpec(
            color=self.color,
            intensity=self.intensity,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True, slots=True)
class AmbientLight(_Light):
    """Scene-wide ambient light descriptor.

    Diagram::

        [ AmbientLight ]
          v   v   v   v
        +---------------+
        |    object     |
        +---------------+
    """

    def _to_spec(self) -> AmbientLightSpec:
        """Return the SDK ambient light spec for this descriptor.

        Returns:
            AmbientLightSpec used by the environment service.
        """
        return AmbientLightSpec(
            color=self.color,
            intensity=self.intensity,
        )


@dataclass(frozen=True, slots=True)
class PointLight(_Light):
    """Point light descriptor with optional range and decay.

    Diagram::

        [ PointLight ]
            \\ | /
          --  o  --
            / | \\
        +---------------+
        |    object     |
        +---------------+

    Attributes:
        range: Optional effective distance for the light.
        decay: Public attenuation decay scalar.
    """

    range: RandomizableFloat | None = None
    decay: RandomizableFloat = 2.0

    def _to_spec(self) -> PointLightSpec:
        """Return the SDK point light spec for this descriptor.

        Returns:
            PointLightSpec used by the environment service.
        """
        return PointLightSpec(
            color=self.color,
            intensity=self.intensity,
            range=self.range,
            decay=self.decay,
        )


type LightType = SpotLight | AreaLight | AmbientLight | PointLight
