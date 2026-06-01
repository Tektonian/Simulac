from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.randomize import (
        RandomizableBool,
        RandomizableColor,
        RandomizableFloat,
        RandomizableFloatList,
        RandomizableVec3,
    )


AxisSide = Literal["right", "left", "front", "back", "up", "down"]
SupportFrame = Literal["world", "local"]
OffsetFrame = Literal["target", "world"]
CameraType = Literal["rgb", "tactile", "depth", "pointcloud", "normal", "segmentation"]
LightType = Literal["ambient", "pointlight", "rectarea", "reactarea", "spot", "area"]


class RefBase:
    """Runtime-checkable base for all engine-neutral references."""


class ObjectRefBase(RefBase):
    """Reference to a backend object such as an entity, geom, joint, site, or camera."""


class PointRefBase(RefBase):
    """Reference that resolves to a world-space point."""


class VectorRefBase(RefBase):
    """Reference that resolves to a vector."""


@dataclass(frozen=True, slots=True)
class EntityRef(ObjectRefBase):
    entity_id: str

    @property
    def pos(self) -> EntityPosRef:
        return EntityPosRef(self.entity_id)

    @property
    def rot(self) -> EntityRotRef:
        return EntityRotRef(self.entity_id)

    @property
    def quat(self) -> EntityQuatRef:
        return EntityQuatRef(self.entity_id)


@dataclass(frozen=True, slots=True)
class EntityPosRef(PointRefBase):
    entity_id: str


@dataclass(frozen=True, slots=True)
class EntityRotRef(VectorRefBase):
    entity_id: str


@dataclass(frozen=True, slots=True)
class EntityQuatRef(VectorRefBase):
    entity_id: str


@dataclass(frozen=True, slots=True)
class BodyRef(ObjectRefBase):
    entity_id: str
    name: str | None = None

    @property
    def pos(self) -> BodyPosRef:
        return BodyPosRef(self.entity_id, self.name)


@dataclass(frozen=True, slots=True)
class BodyPosRef(PointRefBase):
    entity_id: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ColliderRef(ObjectRefBase):
    entity_id: str
    name: str

    @property
    def center(self) -> ColliderCenterRef:
        return ColliderCenterRef(self.entity_id, self.name)

    @property
    def pos(self) -> ColliderPosRef:
        return ColliderPosRef(self.entity_id, self.name)

    @property
    def bounds(self) -> ColliderBoundsRef:
        return ColliderBoundsRef(self.entity_id, self.name)

    def surface(self, side: AxisSide) -> SurfaceRef:
        return SurfaceRef(self.entity_id, self.name, side)

    def support(
        self,
        direction: RandomizableVec3,
        *,
        frame: SupportFrame = "world",
    ) -> SupportPointRef:
        return SupportPointRef(self.entity_id, self.name, direction, frame)

    def set_friction(self, friction: RandomizableFloat) -> SetColliderFrictionOp:
        return SetColliderFrictionOp(self._target(), friction)

    def _target(self) -> ColliderRef:
        return ColliderRef(self.entity_id, self.name)


@dataclass(frozen=True, slots=True)
class ColliderCenterRef(PointRefBase):
    entity_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ColliderPosRef(RefBase):
    entity_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ColliderBoundsRef(RefBase):
    entity_id: str
    collider_name: str

    @property
    def center(self) -> BoundsCenterRef:
        return BoundsCenterRef(self.entity_id, self.collider_name)

    @property
    def min(self) -> BoundsMinRef:
        return BoundsMinRef(self.entity_id, self.collider_name)

    @property
    def max(self) -> BoundsMaxRef:
        return BoundsMaxRef(self.entity_id, self.collider_name)

    @property
    def size(self) -> BoundsSizeRef:
        return BoundsSizeRef(self.entity_id, self.collider_name)


@dataclass(frozen=True, slots=True)
class BoundsCenterRef(PointRefBase):
    entity_id: str
    collider_name: str


@dataclass(frozen=True, slots=True)
class BoundsMinRef(PointRefBase):
    entity_id: str
    collider_name: str


@dataclass(frozen=True, slots=True)
class BoundsMaxRef(PointRefBase):
    entity_id: str
    collider_name: str


@dataclass(frozen=True, slots=True)
class BoundsSizeRef(VectorRefBase):
    entity_id: str
    collider_name: str


@dataclass(frozen=True, slots=True)
class SurfaceRef(RefBase):
    entity_id: str
    collider_name: str
    side: AxisSide

    @property
    def center(self) -> SurfaceCenterRef:
        return SurfaceCenterRef(self.entity_id, self.collider_name, self.side)

    @property
    def normal(self) -> SurfaceNormalRef:
        return SurfaceNormalRef(self.entity_id, self.collider_name, self.side)

    def sample(
        self,
        using: str | "PointRefType" | None = None,
        margin: RandomizableFloat = 0.0,
        x: RandomizableFloat | None = None,
        y: RandomizableFloat | None = None,
        offset: RandomizableVec3 = (0.0, 0.0, 0.0),
        offset_frame: OffsetFrame = "target",
    ) -> SurfaceSampleRef:
        if isinstance(using, str):
            source: str | PointRefType | None = using
        else:
            source = using

        return SurfaceSampleRef(
            entity_id=self.entity_id,
            collider_name=self.collider_name,
            side=self.side,
            using=source,
            margin=margin,
            x=x,
            y=y,
            offset=offset,
            offset_frame=offset_frame,
        )


@dataclass(frozen=True, slots=True)
class SurfaceCenterRef(PointRefBase):
    entity_id: str
    collider_name: str
    side: AxisSide


@dataclass(frozen=True, slots=True)
class SurfaceSampleRef(PointRefBase):
    entity_id: str
    collider_name: str
    side: AxisSide
    using: str | PointRefType | None = None
    margin: RandomizableFloat = 0.0
    x: RandomizableFloat | None = None
    y: RandomizableFloat | None = None
    offset: RandomizableVec3 = (0.0, 0.0, 0.0)
    offset_frame: OffsetFrame = "target"


@dataclass(frozen=True, slots=True)
class SurfaceNormalRef(VectorRefBase):
    entity_id: str
    collider_name: str
    side: AxisSide


@dataclass(frozen=True, slots=True)
class SupportPointRef(PointRefBase):
    entity_id: str
    collider_name: str
    direction: RandomizableVec3
    frame: SupportFrame = "world"


@dataclass(frozen=True, slots=True)
class JointRef(ObjectRefBase):
    entity_id: str
    name: str

    @property
    def pos(self) -> JointPosRef:
        return JointPosRef(self.entity_id, self.name)

    @property
    def axis(self) -> JointAxisRef:
        return JointAxisRef(self.entity_id, self.name)

    @property
    def range(self) -> JointRangeRef:
        return JointRangeRef(self.entity_id, self.name)

    def set_pos(self, pos: RandomizableFloat) -> SetJointPosOp:
        return SetJointPosOp(self._target(), pos)

    def set_friction(self, friction: RandomizableFloat) -> SetJointFrictionOp:
        return SetJointFrictionOp(self._target(), friction)

    def set_damping(self, damping: RandomizableFloat) -> SetJointDampingOp:
        return SetJointDampingOp(self._target(), damping)

    def _target(self) -> JointRef:
        return JointRef(self.entity_id, self.name)


@dataclass(frozen=True, slots=True)
class JointPosRef(RefBase):
    entity_id: str
    name: str


@dataclass(frozen=True, slots=True)
class JointVelRef(RefBase):
    entity_id: str
    name: str


@dataclass(frozen=True, slots=True)
class JointAxisRef(VectorRefBase):
    entity_id: str
    name: str


@dataclass(frozen=True, slots=True)
class JointRangeRef(RefBase):
    entity_id: str
    name: str


@dataclass(frozen=True, slots=True)
class AnchorRef(ObjectRefBase):
    entity_id: str
    name: str

    @property
    def pos(self) -> AnchorPosRef:
        return AnchorPosRef(self.entity_id, self.name)


@dataclass(frozen=True, slots=True)
class AnchorPosRef(PointRefBase):
    entity_id: str
    name: str


@dataclass(frozen=True, slots=True)
class CameraRef(ObjectRefBase):
    entity_id: str
    name: str | None = None

    @property
    def output(self) -> CameraOutputRef:
        return CameraOutputRef(self.entity_id, self.name)

    @property
    def pos(self) -> CameraPosRef:
        return CameraPosRef(self.entity_id, self.name)


@dataclass(frozen=True, slots=True)
class CameraOutputRef(RefBase):
    entity_id: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class CameraPosRef(PointRefBase):
    entity_id: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class LightRef(ObjectRefBase):
    entity_id: str
    name: str | None = None

    @property
    def pos(self) -> LightPosRef:
        return LightPosRef(self.entity_id, self.name)


@dataclass(frozen=True, slots=True)
class LightPosRef(PointRefBase):
    entity_id: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class WorldPointRef(PointRefBase):
    pos: RandomizableVec3


type ObjectRefType = (
    EntityRef | BodyRef | ColliderRef | JointRef | AnchorRef | CameraRef | LightRef
)

type PointRefType = (
    EntityPosRef
    | BodyPosRef
    | ColliderCenterRef
    | BoundsCenterRef
    | BoundsMinRef
    | BoundsMaxRef
    | SurfaceCenterRef
    | SurfaceSampleRef
    | SupportPointRef
    | AnchorPosRef
    | CameraPosRef
    | LightPosRef
    | WorldPointRef
)

type VectorRefType = (
    EntityRotRef | EntityQuatRef | BoundsSizeRef | SurfaceNormalRef | JointAxisRef
)

type RefType = (
    ObjectRefType
    | PointRefType
    | VectorRefType
    | EntityPosRef
    | ColliderPosRef
    | ColliderBoundsRef
    | SurfaceRef
    | JointPosRef
    | JointVelRef
    | JointRangeRef
    | CameraOutputRef
)

type PlaceTargetRefType = (
    PointRefType | EntityRef | BodyRef | ColliderRef | SurfaceRef | AnchorRef
)


def as_place_target(
    ref: PlaceTargetRefType, *, margin: RandomizableFloat = 0.0
) -> PointRefType:
    if isinstance(ref, PointRefBase):
        return ref
    if isinstance(ref, SurfaceRef):
        return ref.sample(margin)
    if isinstance(ref, ColliderRef):
        return ref.surface("up").sample(margin)
    if isinstance(ref, AnchorRef):
        return ref.pos
    if isinstance(ref, EntityRef):
        return ref.pos
    if isinstance(ref, BodyRef):  # pyright: ignore[reportUnnecessaryIsInstance]
        return ref.pos
    raise TypeError(f"Unsupported place target ref: {ref!r}")


def as_place_source(ref: PlaceTargetRefType | None) -> PointRefType | None:
    if ref is None:
        return None
    if isinstance(ref, PointRefBase):
        return ref
    if isinstance(ref, SurfaceRef):
        return ref.center
    if isinstance(ref, ColliderRef):
        return ref.surface("down").center
    if isinstance(ref, AnchorRef):
        return ref.pos
    if isinstance(ref, EntityRef):
        return ref.pos
    if isinstance(ref, BodyRef):  # pyright: ignore[reportUnnecessaryIsInstance]
        return ref.pos
    raise TypeError(f"Unsupported place source ref: {ref!r}")


class BuildOpBase:
    """Runtime-checkable base for all deferred build operations."""


# region - Needed?

# @dataclass(frozen=True, slots=True)
# class AddEntityOp(BuildOpBase):
#     entity: EntityRef


# @dataclass(frozen=True, slots=True)
# class RemoveEntityOp(BuildOpBase):
#     entity: EntityRef


# @dataclass(frozen=True, slots=True)
# class ReplaceEntityOp(BuildOpBase):
#     entity: EntityRef
#     render_uri: str
#     physics_uri: str

# end-region


@dataclass(frozen=True, slots=True)
class SetEntityPosOp(BuildOpBase):
    entity: EntityRef
    pos: RandomizableVec3 | PointRefType


@dataclass(frozen=True, slots=True)
class SetEntityRotOp(BuildOpBase):
    entity: EntityRef
    rot: RandomizableVec3


@dataclass(frozen=True, slots=True)
class SetEntityQuatOp(BuildOpBase):
    entity: EntityRef
    quat: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SetEntitySizeOp(BuildOpBase):
    entity: EntityRef
    size: RandomizableVec3


@dataclass(frozen=True, slots=True)
class SetEntityFixedOp(BuildOpBase):
    entity: EntityRef
    fixed: RandomizableBool


@dataclass(frozen=True, slots=True)
class SetEntityMassOp(BuildOpBase):
    entity: EntityRef
    mass: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetEntityFrictionOp(BuildOpBase):
    entity: EntityRef
    friction: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetColliderSizeOp(BuildOpBase):
    target: ColliderRef
    size: RandomizableVec3


@dataclass(frozen=True, slots=True)
class SetColliderFrictionOp(BuildOpBase):
    target: ColliderRef
    friction: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetJointPosOp(BuildOpBase):
    target: JointRef
    pos: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetJointVelOp(BuildOpBase):
    target: JointRef
    vel: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetJointCtrlOp(BuildOpBase):
    target: JointRef
    ctrl: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetJointFrictionOp(BuildOpBase):
    target: JointRef
    friction: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetJointDampingOp(BuildOpBase):
    target: JointRef
    damping: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetActPosOp(BuildOpBase):
    entity: EntityRef
    pos: RandomizableFloatList


@dataclass(frozen=True, slots=True)
class PlaceOp(BuildOpBase):
    entity: EntityRef
    target: PointRefType
    source: PointRefType | None = None
    margin: RandomizableFloat = 0.0


@dataclass(frozen=True, slots=True)
class AttachOp(BuildOpBase):
    entity: EntityRef
    parent: EntityRef | AnchorRef
    offset: RandomizableVec3 = (0, 0, 0)
    rot: RandomizableVec3 = (0, 0, 0)
    offset_frame: OffsetFrame = "target"


@dataclass(frozen=True, slots=True)
class LookAtOp(BuildOpBase):
    entity: EntityRef
    target: PointRefType
    up: RandomizableVec3 = (0, 0, 1)
    offset: RandomizableVec3 = (0, 0, 0)
    offset_frame: OffsetFrame = "world"


@dataclass(frozen=True, slots=True)
class FollowOp(BuildOpBase):
    entity: EntityRef
    target: EntityRef | AnchorRef | ColliderRef
    offset: RandomizableVec3 = (0, 0, 0)
    frame: Literal["world", "local"] = "world"
    keep_offset: bool = True


@dataclass(frozen=True, slots=True)
class SetCameraPosOp(BuildOpBase):
    camera: CameraRef
    pos: RandomizableVec3 | PointRefType


@dataclass(frozen=True, slots=True)
class SetCameraRotOp(BuildOpBase):
    camera: CameraRef
    rot: RandomizableVec3


@dataclass(frozen=True, slots=True)
class SetCameraTypeOp(BuildOpBase):
    camera: CameraRef
    type: CameraType


@dataclass(frozen=True, slots=True)
class SetCameraFovOp(BuildOpBase):
    camera: CameraRef
    fov: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetCameraAspectOp(BuildOpBase):
    camera: CameraRef
    aspect: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetCameraNearOp(BuildOpBase):
    camera: CameraRef
    near: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetCameraFarOp(BuildOpBase):
    camera: CameraRef
    far: RandomizableFloat


# @dataclass(frozen=True, slots=True)
# class SetCameraResolutionOp(BuildOpBase):
#     camera: CameraRef
#     width: int
#     height: int


# @dataclass(frozen=True, slots=True)
# class SetCameraNoiseOp(BuildOpBase):
#     camera: CameraRef
#     enabled: RandomizableBool = True
#     std: RandomizableFloat = 0.0
#     model: Literal["gaussian", "depth", "salt_pepper"] = "gaussian"


# @dataclass(frozen=True, slots=True)
# class SetCameraExposureOp(BuildOpBase):
#     camera: CameraRef
#     exposure: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetLightPosOp(BuildOpBase):
    light: LightRef
    pos: RandomizableVec3 | PointRefType


@dataclass(frozen=True, slots=True)
class SetLightRotOp(BuildOpBase):
    light: LightRef
    rot: RandomizableVec3


@dataclass(frozen=True, slots=True)
class SetLightTypeOp(BuildOpBase):
    light: LightRef
    type: LightType


@dataclass(frozen=True, slots=True)
class SetLightIntensityOp(BuildOpBase):
    light: LightRef
    intensity: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetLightColorOp(BuildOpBase):
    light: LightRef
    color: RandomizableColor


@dataclass(frozen=True, slots=True)
class SetLightAngleOp(BuildOpBase):
    light: LightRef
    angle: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetLightAreaSizeOp(BuildOpBase):
    light: LightRef
    width: RandomizableFloat
    height: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetLightRangeOp(BuildOpBase):
    light: LightRef
    range: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetLightDecayOp(BuildOpBase):
    light: LightRef
    decay: RandomizableFloat


@dataclass(frozen=True, slots=True)
class SetLightPenumbraOp(BuildOpBase):
    light: LightRef
    penumbra: RandomizableFloat


type BuildOpType = (
    # AddEntityOp
    # | RemoveEntityOp
    # | ReplaceEntityOp
    SetEntityPosOp
    | SetEntityRotOp
    | SetEntityQuatOp
    | SetEntitySizeOp
    | SetEntityFixedOp
    | SetEntityMassOp
    | SetEntityFrictionOp
    | SetColliderSizeOp
    | SetColliderFrictionOp
    | SetJointPosOp
    | SetJointVelOp
    | SetJointCtrlOp
    | SetJointFrictionOp
    | SetJointDampingOp
    | SetActPosOp
    | PlaceOp
    | AttachOp
    | LookAtOp
    | FollowOp
    | SetCameraPosOp
    | SetCameraRotOp
    | SetCameraTypeOp
    | SetCameraFovOp
    | SetCameraAspectOp
    | SetCameraNearOp
    | SetCameraFarOp
    # | SetCameraResolutionOp
    # | SetCameraNoiseOp
    # | SetCameraExposureOp
    | SetLightPosOp
    | SetLightRotOp
    | SetLightTypeOp
    | SetLightIntensityOp
    | SetLightColorOp
    | SetLightAngleOp
    | SetLightAreaSizeOp
    | SetLightRangeOp
    | SetLightDecayOp
    | SetLightPenumbraOp
)
