from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Tuple

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.model.component import (
        RenderingComponent,
    )
    from simulac.sdk.environment_service.common.model.ref import BuildOpType
    from simulac.sdk.environment_service.common.randomize import (
        RandomizableBool,
        RandomizableColor,
        RandomizableFloat,
        RandomizableFloatList,
        RandomizableQuat,
        RandomizableVec3,
    )


@dataclass(slots=True)
class EnvironmentMachineEntity:
    id: str | None = None
    description: str = ""
    asset_uri: str | None = None
    pos: RandomizableVec3 = (0, 0, 0)
    rot: RandomizableVec3 = (0, 0, 0)

    init_position: RandomizableFloatList | None = None
    action_max: list[float] | None = None
    action_min: list[float] | None = None


@dataclass(slots=True)
class EnvironmentStuffEntity:
    id: str | None = None
    description: str = ""
    asset_uri: str | None = None
    pos: RandomizableVec3 = (0, 0, 0)
    rot: RandomizableVec3 = (0, 0, 0)
    size: RandomizableVec3 = (1, 1, 1)
    fixed: RandomizableBool = True
    mass: RandomizableFloat | None = None
    friction: RandomizableFloat | None = None
    density: RandomizableFloat | None = None


@dataclass(slots=True)
class TransformSpec:
    pos: RandomizableVec3 = (0, 0, 0)
    rot: RandomizableVec3 = (0, 0, 0)


@dataclass(slots=True)
class EntityRef:
    entity_id: str


@dataclass(slots=True)
class AnchorRef:
    entity_id: str
    anchor: str


@dataclass(slots=True)
class ColliderRef:
    entity_id: str
    collider: str


@dataclass(slots=True)
class WorldPointRef:
    pos: RandomizableVec3


type SpatialRef = EntityRef | AnchorRef | ColliderRef | WorldPointRef


@dataclass(slots=True)
class AttachSpec:
    target: EntityRef | AnchorRef
    offset: TransformSpec = field(default_factory=TransformSpec)
    offset_frame: Literal["target", "world"] = "target"


@dataclass(slots=True)
class LookAtSpec:
    target: SpatialRef
    up: RandomizableVec3 = (0, 0, 1)
    offset: RandomizableVec3 = (0, 0, 0)
    offset_frame: Literal["target", "world"] = "world"


@dataclass(slots=True)
class TrackSpec:
    target: SpatialRef
    keep_offset: bool = True
    offset_frame: Literal["target", "world"] = "world"


@dataclass(slots=True)
class AmbientLightSpec:
    type: Literal["ambient"] = "ambient"
    enabled: RandomizableBool = True
    color: RandomizableColor = (255, 255, 255)
    intensity: RandomizableFloat = 0.8


@dataclass(slots=True)
class PointLightSpec:
    type: Literal["pointlight"] = "pointlight"
    color: RandomizableColor = (255, 255, 255)
    enabled: RandomizableBool = True
    intensity: RandomizableFloat = 0.8
    range: RandomizableFloat | None = None
    decay: RandomizableFloat = 2.0


@dataclass(slots=True)
class SpotLightSpec:
    type: Literal["spot"] = "spot"
    color: RandomizableColor = (255, 255, 255)
    enabled: RandomizableBool = True
    intensity: RandomizableFloat = 0.8
    range: RandomizableFloat | None = None
    decay: RandomizableFloat = 2.0
    angle: RandomizableFloat = 45.0
    penumbra: RandomizableFloat = 0.0


@dataclass(slots=True)
class AreaLightSpec:
    type: Literal["area"] = "area"
    color: RandomizableColor = (255, 255, 255)
    enabled: RandomizableBool = True
    intensity: RandomizableFloat = 0.8
    width: RandomizableFloat = 1.0
    height: RandomizableFloat = 1.0


type LightType = Literal["ambient", "pointlight", "rectarea", "spot"]
type LightSpec = AmbientLightSpec | PointLightSpec | SpotLightSpec | AreaLightSpec


@dataclass(slots=True)
class EnvironmentLightEntity:
    id: str | None = None
    description: str = ""
    pos: RandomizableVec3 = (0, 0, 0)
    rot: RandomizableVec3 = (0, 0, 0)
    spec: LightSpec = field(default_factory=PointLightSpec)

    attach: AttachSpec | None = None
    look_at: LookAtSpec | None = None
    track: TrackSpec | None = None


type CameraType = Literal[
    "rgb", "tactile", "depth", "pointcloud", "normal", "segmentation"
]
type CameraMode = Literal["fixed", "track"]


@dataclass(frozen=True, slots=True)
class CameraSpec:
    type: CameraType = "rgb"
    mode: CameraMode = "track"
    lookat: RandomizableVec3 = (0, 0, 0)
    fov: RandomizableFloat = 50.0
    aspect: RandomizableFloat = 1.0
    near: RandomizableFloat = 100.0
    far: RandomizableFloat = 1000.0


@dataclass(slots=True)
class EnvironmentCameraEntity:
    id: str | None = None
    description: str = ""
    spec: CameraSpec = field(default_factory=CameraSpec)
    pos: RandomizableVec3 = (0, 0, 0)
    rot: RandomizableVec3 = (0, 0, 0)

    attach: AttachSpec | None = None
    look_at: LookAtSpec | None = None
    track: TrackSpec | None = None
