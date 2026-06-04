from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from simulac.base.types.geometry import ColorRgb, Quat, Vec3


def _joint_state(ops: IJointRuntimeOps, name: str) -> JointState:
    """Common joint parts for all physics engine"""
    joint_type = ops.get_joint_type(name)
    if joint_type == "hinge":
        return HingeJointState(ops, name)
    if joint_type == "slide":
        return SlideJointState(ops, name)
    if joint_type == "ball":
        return BallJointState(ops, name)
    return FreeJointState(ops, name)


class IJointRuntimeOps(Protocol):
    def get_joint_type(
        self, name: str
    ) -> Literal["hinge", "slide", "ball", "free"]: ...
    def get_joint_scalar_pos(self, name: str) -> float: ...
    def get_joint_scalar_vel(self, name: str) -> float: ...
    def get_joint_free_pos(self, name: str) -> Vec3: ...
    def get_joint_quat(self, name: str) -> Quat: ...
    def get_joint_linear_vel(self, name: str) -> Vec3: ...
    def get_joint_angular_vel(self, name: str) -> Vec3: ...
    def get_joint_axis(self, name: str) -> Vec3: ...
    def get_joint_limited(self, name: str) -> bool: ...
    def get_joint_range(self, name: str) -> tuple[float, float] | None: ...
    def get_joint_force(self, name: str) -> float: ...


class LinkState:
    def __init__(self, ops: IRobotRuntimeOps, name: str) -> None:
        self._ops = ops
        self.name = name

    @property
    def pos(self) -> Vec3:
        return self._ops.get_link_pos(self.name)

    @property
    def quat(self) -> Quat:
        return self._ops.get_link_quat(self.name)

    @property
    def rot(self) -> Quat:
        return self.quat

    @property
    def linear_vel(self) -> Vec3:
        return self._ops.get_link_linear_vel(self.name)

    @property
    def angular_vel(self) -> Vec3:
        return self._ops.get_link_angular_vel(self.name)


class SiteState:
    def __init__(self, ops: IRobotRuntimeOps, name: str) -> None:
        self._ops = ops
        self.name = name

    @property
    def pos(self) -> Vec3:
        return self._ops.get_site_pos(self.name)

    @property
    def quat(self) -> Quat:
        return self._ops.get_site_quat(self.name)

    @property
    def rot(self) -> Quat:
        return self.quat

    @property
    def linear_vel(self) -> Vec3:
        return self._ops.get_site_linear_vel(self.name)

    @property
    def angular_vel(self) -> Vec3:
        return self._ops.get_site_angular_vel(self.name)


class _JointStateBase:
    def __init__(self, ops: IJointRuntimeOps, name: str) -> None:
        self._ops = ops
        self.name = name


class HingeJointState(_JointStateBase):
    @property
    def type(self) -> Literal["hinge"]:
        return "hinge"

    @property
    def pos(self) -> float:
        return self._ops.get_joint_scalar_pos(self.name)

    @property
    def vel(self) -> float:
        return self._ops.get_joint_scalar_vel(self.name)

    @property
    def axis(self) -> Vec3:
        return self._ops.get_joint_axis(self.name)

    @property
    def limited(self) -> bool:
        return self._ops.get_joint_limited(self.name)

    @property
    def range(self) -> tuple[float, float] | None:
        return self._ops.get_joint_range(self.name)

    @property
    def force(self) -> float:
        return self._ops.get_joint_force(self.name)


class SlideJointState(_JointStateBase):
    @property
    def type(self) -> Literal["slide"]:
        return "slide"

    @property
    def pos(self) -> float:
        return self._ops.get_joint_scalar_pos(self.name)

    @property
    def vel(self) -> float:
        return self._ops.get_joint_scalar_vel(self.name)

    @property
    def axis(self) -> Vec3:
        return self._ops.get_joint_axis(self.name)

    @property
    def limited(self) -> bool:
        return self._ops.get_joint_limited(self.name)

    @property
    def range(self) -> tuple[float, float] | None:
        return self._ops.get_joint_range(self.name)

    @property
    def force(self) -> float:
        return self._ops.get_joint_force(self.name)


class BallJointState(_JointStateBase):
    @property
    def type(self) -> Literal["ball"]:
        return "ball"

    @property
    def quat(self) -> Quat:
        return self._ops.get_joint_quat(self.name)

    @property
    def rot(self) -> Quat:
        return self.quat

    @property
    def angular_vel(self) -> Vec3:
        return self._ops.get_joint_angular_vel(self.name)


class FreeJointState(_JointStateBase):
    @property
    def type(self) -> Literal["free"]:
        return "free"

    @property
    def pos(self) -> Vec3:
        return self._ops.get_joint_free_pos(self.name)

    @property
    def quat(self) -> Quat:
        return self._ops.get_joint_quat(self.name)

    @property
    def rot(self) -> Quat:
        return self.quat

    @property
    def linear_vel(self) -> Vec3:
        return self._ops.get_joint_linear_vel(self.name)

    @property
    def angular_vel(self) -> Vec3:
        return self._ops.get_joint_angular_vel(self.name)


JointState: TypeAlias = (
    HingeJointState | SlideJointState | BallJointState | FreeJointState
)


_UNSET = object()


class ContactResult:
    def __init__(self, ops: IRuntimeStateOps, a: object, b: object) -> None:
        self._ops = ops
        self._a = a
        self._b = b
        self._indices: tuple[int, ...] | None = None
        self._points: tuple[Vec3, ...] | None = None
        self._normal: Vec3 | None | object = _UNSET
        self._max_force: float | None | object = _UNSET

    def _contact_indices(self) -> tuple[int, ...]:
        if self._indices is None:
            self._indices = self._ops.contact_indices(self._a, self._b)
        return self._indices

    @property
    def exists(self) -> bool:
        return len(self._contact_indices()) > 0

    @property
    def count(self) -> int:
        return len(self._contact_indices())

    @property
    def points(self) -> tuple[Vec3, ...]:
        if self._points is None:
            self._points = tuple(
                self._ops.contact_point(i) for i in self._contact_indices()
            )
        return self._points

    @property
    def normal(self) -> Vec3 | None:
        if self._normal is _UNSET:
            indices = self._contact_indices()
            self._normal = None if not indices else self._ops.contact_normal(indices[0])
        return self._normal

    @property
    def max_force(self) -> float | None:
        if self._max_force is _UNSET:
            forces = [self._ops.contact_force(i) for i in self._contact_indices()]
            forces = [f for f in forces if f is not None]
            self._max_force = None if not forces else max(forces)
        return self._max_force


class IRuntimeStateOps(Protocol):
    def get_time(self) -> float: ...
    def get_step_count(self) -> int: ...

    def contact_indices(self, a: object, b: object) -> tuple[int, ...]: ...
    def contact_point(self, contact_index: int) -> Vec3: ...
    def contact_normal(self, contact_index: int) -> Vec3: ...
    def contact_force(self, contact_index: int) -> float | None: ...


class RuntimeState:
    def __init__(self, ops: IRuntimeStateOps) -> None:
        self._ops = ops

    @property
    def time(self) -> float:
        return self._ops.get_time()

    @property
    def step_count(self) -> int:
        return self._ops.get_step_count()

    def contacts(self, a: object, b: object) -> ContactResult:
        return ContactResult(self._ops, a, b)


@runtime_checkable
class IStuffRuntimeOps(IJointRuntimeOps, Protocol):
    def get_pos(self) -> Vec3: ...
    def get_quat(self) -> Quat: ...
    def get_mass(self) -> float: ...
    def get_friction(self) -> float: ...

    def change_pos(self, pos: Vec3) -> None: ...
    def change_quat(self, quat: Quat) -> None: ...
    def change_mass(self, mass: float) -> None: ...
    def change_friction(self, friction: float) -> None: ...


class StuffRuntime:
    def __init__(self, entity_id: str, ops: IStuffRuntimeOps) -> None:
        self.id = entity_id
        self._ops = ops

    def get_pos(self) -> Vec3:
        return self._ops.get_pos()

    def get_quat(self) -> Quat:
        return self._ops.get_quat()

    def get_mass(self) -> float:
        return self._ops.get_mass()

    def get_friction(self) -> float:
        return self._ops.get_friction()

    def change_pos(self, pos: Vec3) -> None:
        self._ops.change_pos(pos)

    def change_quat(self, quat: Quat) -> None:
        self._ops.change_quat(quat)

    def change_mass(self, mass: float) -> None:
        self._ops.change_mass(mass)

    def change_friction(self, friction: float) -> None:
        self._ops.change_friction(friction)

    def joint(self, name: str) -> JointState:
        return _joint_state(self._ops, name)


class IRobotRuntimeOps(IJointRuntimeOps, Protocol):
    # Robot object itself
    def get_base_pos(self) -> Vec3: ...
    def get_base_quat(self) -> Quat: ...

    # Robot joint
    def get_joint_pos(self) -> list[float]: ...
    def get_joint_vel(self) -> list[float]: ...

    def get_joint_type(
        self,
        name: str,
    ) -> Literal["hinge", "slide", "ball", "free"]: ...
    def get_joint_scalar_pos(self, name: str) -> float: ...
    def get_joint_scalar_vel(self, name: str) -> float: ...
    def get_joint_free_pos(self, name: str) -> Vec3: ...
    def get_joint_quat(self, name: str) -> Quat: ...
    def get_joint_linear_vel(self, name: str) -> Vec3: ...
    def get_joint_angular_vel(self, name: str) -> Vec3: ...
    def get_joint_axis(self, name: str) -> Vec3: ...
    def get_joint_limited(self, name: str) -> bool: ...
    def get_joint_range(self, name: str) -> tuple[float, float] | None: ...
    def get_joint_force(self, name: str) -> float: ...

    # Sensors on robot
    def get_sensor_value(self, name: str) -> tuple[float, ...]: ...
    def get_sensor_dim(self, name: str) -> int: ...
    def get_sensor_type(self, name: str) -> int: ...

    # Sites
    def get_site_pos(self, name: str) -> Vec3: ...
    def get_site_quat(self, name: str) -> Quat: ...
    def get_site_linear_vel(self, name: str) -> Vec3: ...
    def get_site_angular_vel(self, name: str) -> Vec3: ...

    # Links
    def get_link_pos(self, name: str) -> Vec3: ...
    def get_link_quat(self, name: str) -> Quat: ...
    def get_link_linear_vel(self, name: str) -> Vec3: ...
    def get_link_angular_vel(self, name: str) -> Vec3: ...

    # control ops
    def change_joint_pos(self, joint_pos: list[float]) -> None: ...
    def change_joint_vel(self, joint_vel: list[float]) -> None: ...
    def set_control(self, action: list[float]) -> None: ...


class RobotRuntime:
    def __init__(self, entity_id: str, ops: IRobotRuntimeOps) -> None:
        self.id = entity_id
        self._ops = ops

    def get_pos(self) -> Vec3:
        return self._ops.get_base_pos()

    def get_quat(self) -> Quat:
        return self._ops.get_base_quat()

    def get_joint_pos(self) -> list[float]:
        return self._ops.get_joint_pos()

    def get_joint_vel(self) -> list[float]:
        return self._ops.get_joint_vel()

    def joint(self, name: str) -> JointState:
        return _joint_state(self._ops, name)

    def sensor(self, name: str) -> SensorState:
        return SensorState(self._ops, name)

    def site(self, name: str) -> SiteState:
        return SiteState(self._ops, name)

    def link(self, name: str) -> LinkState:
        return LinkState(self._ops, name)

    def get_site_pos(self, name: str) -> Vec3:
        return self._ops.get_site_pos(name)

    def get_site_quat(self, name: str) -> Quat:
        return self._ops.get_site_quat(name)

    def get_link_pos(self, name: str) -> Vec3:
        return self._ops.get_link_pos(name)

    def get_link_quat(self, name: str) -> Quat:
        return self._ops.get_link_quat(name)

    def change_joint_pos(self, joint_pos: list[float]) -> None:
        self._ops.change_joint_pos(joint_pos)

    def change_joint_vel(self, joint_vel: list[float]) -> None:
        self._ops.change_joint_vel(joint_vel)

    def set_control(self, action: list[float]) -> None:
        self._ops.set_control(action)


class ICameraRuntimeOps(Protocol):
    def get_pos(self) -> Vec3: ...
    def get_quat(self) -> Quat: ...

    def change_pos(self, pos: Vec3) -> None: ...
    def change_quat(self, quat: Quat) -> None: ...

    def get_fov(self) -> float: ...
    def change_fov(self, fov: float) -> None: ...
    def render(self, *, width: int = 640, height: int = 480) -> Any: ...


class CameraRuntime:
    def __init__(self, entity_id: str, ops: ICameraRuntimeOps) -> None:
        self.id = entity_id
        self._ops = ops

    def get_pos(self) -> Vec3:
        return self._ops.get_pos()

    def get_quat(self) -> Quat:
        return self._ops.get_quat()

    def change_pos(self, pos: Vec3) -> None:
        self._ops.change_pos(pos)

    def change_quat(self, quat: Quat) -> None:
        self._ops.change_quat(quat)

    def get_fov(self) -> float:
        return self._ops.get_fov()

    def change_fov(self, fov: float) -> None:
        self._ops.change_fov(fov)

    def render(self, *, width: int = 640, height: int = 480) -> Any:
        return self._ops.render(width=width, height=height)


class ILightRuntimeOps(Protocol):
    # common
    def get_pos(self) -> Vec3: ...
    def get_quat(self) -> Quat: ...
    def change_pos(self, pos: Vec3) -> None: ...
    def change_quat(self, quat: Quat) -> None: ...

    def get_color(self) -> ColorRgb: ...
    def change_color(self, color: ColorRgb) -> None: ...

    def get_intensity(self) -> float: ...
    def change_intensity(self, intensity: float) -> None: ...


class IAmbientLightRuntimeOps(ILightRuntimeOps, Protocol):
    pass


class IPointLightRuntimeOps(ILightRuntimeOps, Protocol):
    def get_range(self) -> float: ...
    def change_range(self, range: float) -> None: ...

    def get_decay(self) -> float: ...
    def change_decay(self, decay: float) -> None: ...


class IDirectionalLightRuntimeOps(ILightRuntimeOps, Protocol):
    def get_direction(self) -> Vec3: ...
    def change_direction(self, direction: Vec3) -> None: ...


class ISpotLightRuntimeOps(
    IPointLightRuntimeOps, IDirectionalLightRuntimeOps, Protocol
):
    def get_angle(self) -> float: ...
    def change_angle(self, angle: float) -> None: ...

    def get_penumbra(self) -> float: ...
    def change_penumbra(self, penumbra: float) -> None: ...


class IAreaLightRuntimeOps(IDirectionalLightRuntimeOps, Protocol):
    def get_area_size(self) -> tuple[float, float]: ...
    def change_area_size(self, width: float, height: float) -> None: ...


class LightRuntime:
    def __init__(self, entity_id: str, ops: ILightRuntimeOps) -> None:
        self.id = entity_id
        self._ops = ops

    def get_pos(self) -> Vec3:
        return self._ops.get_pos()

    def get_quat(self) -> Quat:
        return self._ops.get_quat()

    def change_pos(self, pos: Vec3) -> None:
        self._ops.change_pos(pos)

    def change_quat(self, quat: Quat) -> None:
        self._ops.change_quat(quat)

    def get_color(self) -> ColorRgb:
        return self._ops.get_color()

    def change_color(self, color: ColorRgb) -> None:
        self._ops.change_color(color)

    def get_intensity(self) -> float:
        return self._ops.get_intensity()

    def change_intensity(self, intensity: float) -> None:
        self._ops.change_intensity(intensity)


class AmbientLightRuntime(LightRuntime):
    def __init__(self, entity_id: str, ops: IAmbientLightRuntimeOps) -> None:
        super().__init__(entity_id, ops)
        self._ops = ops


class PointLightRuntime(LightRuntime):
    def __init__(self, entity_id: str, ops: IPointLightRuntimeOps) -> None:
        super().__init__(entity_id, ops)
        self._ops = ops

    def get_range(self) -> float:
        return self._ops.get_range()

    def change_range(self, range: float) -> None:
        self._ops.change_range(range)

    def get_decay(self) -> float:
        return self._ops.get_decay()

    def change_decay(self, decay: float) -> None:
        self._ops.change_decay(decay)


class SpotLightRuntime(PointLightRuntime):
    def __init__(self, entity_id: str, ops: ISpotLightRuntimeOps) -> None:
        super().__init__(entity_id, ops)
        self._ops = ops

    def get_angle(self) -> float:
        return self._ops.get_angle()

    def change_angle(self, angle: float) -> None:
        self._ops.change_angle(angle)

    def get_direction(self) -> Vec3:
        return self._ops.get_direction()

    def change_direction(self, direction: Vec3) -> None:
        self._ops.change_direction(direction)

    def get_penumbra(self) -> float:
        return self._ops.get_penumbra()

    def change_penumbra(self, penumbra: float) -> None:
        self._ops.change_penumbra(penumbra)


class AreaLightRuntime(LightRuntime):
    def __init__(self, entity_id: str, ops: IAreaLightRuntimeOps) -> None:
        super().__init__(entity_id, ops)
        self._ops = ops

    def get_area_size(self) -> tuple[float, float]:
        return self._ops.get_area_size()

    def change_area_size(self, width: float, height: float) -> None:
        self._ops.change_area_size(width, height)

    def get_direction(self) -> Vec3:
        return self._ops.get_direction()

    def change_direction(self, direction: Vec3) -> None:
        self._ops.change_direction(direction)


class SensorState:
    def __init__(self, ops: IRobotRuntimeOps, name: str) -> None:
        self._ops = ops
        self.name = name

    @property
    def value(self) -> tuple[float, ...]:
        return self._ops.get_sensor_value(self.name)

    @property
    def dim(self) -> int:
        return self._ops.get_sensor_dim(self.name)

    @property
    def type(self) -> int:
        return self._ops.get_sensor_type(self.name)
