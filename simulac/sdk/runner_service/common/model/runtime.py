from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable
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
class IStuffRuntimeOps(Protocol):
    def get_pos(self) -> tuple[float, float, float]: ...
    def get_quat(self) -> tuple[float, float, float, float]: ...
    def get_mass(self) -> float: ...
    def get_friction(self) -> float: ...

    def change_pos(self, pos: tuple[float, float, float]) -> None: ...
    def change_quat(self, quat: tuple[float, float, float, float]) -> None: ...
    def change_mass(self, mass: float) -> None: ...
    def change_friction(self, friction: float) -> None: ...


class StuffRuntime:
    def __init__(self, entity_id: str, ops: IStuffRuntimeOps) -> None:
        self.id = entity_id
        self._ops = ops

    def get_pos(self) -> tuple[float, float, float]:
        return self._ops.get_pos()

    def get_quat(self) -> tuple[float, float, float, float]:
        return self._ops.get_quat()

    def get_mass(self) -> float:
        return self._ops.get_mass()

    def get_friction(self) -> float:
        return self._ops.get_friction()

    def change_pos(self, pos: tuple[float, float, float]) -> None:
        self._ops.change_pos(pos)

    def change_quat(self, quat: tuple[float, float, float, float]) -> None:
        self._ops.change_quat(quat)

    def change_mass(self, mass: float) -> None:
        self._ops.change_mass(mass)

    def change_friction(self, friction: float) -> None:
        self._ops.change_friction(friction)


class IRobotRuntimeOps(Protocol):
    # below two are for mobile robot or floating boat
    def get_base_pos(self) -> tuple[float, float, float]: ...
    def get_base_quat(self) -> tuple[float, float, float, float]: ...

    def get_joint_pos(self) -> list[float]: ...
    def get_joint_vel(self) -> list[float]: ...

    def change_joint_pos(self, joint_pos: list[float]) -> None: ...
    def change_joint_vel(self, joint_vel: list[float]) -> None: ...

    def step(self, action: list[float]) -> None: ...
    def tick(self) -> None: ...


class RobotRuntime:
    def __init__(self, entity_id: str, ops: IRobotRuntimeOps) -> None:
        self.id = entity_id
        self._ops = ops

    def get_pos(self) -> tuple[float, float, float]:
        return self._ops.get_base_pos()

    def get_quat(self) -> tuple[float, float, float, float]:
        return self._ops.get_base_quat()

    def get_joint_pos(self) -> list[float]:
        return self._ops.get_joint_pos()

    def get_joint_vel(self) -> list[float]:
        return self._ops.get_joint_vel()

    def change_joint_pos(self, joint_pos: list[float]) -> None:
        self._ops.change_joint_pos(joint_pos)

    def change_joint_vel(self, joint_vel: list[float]) -> None:
        self._ops.change_joint_vel(joint_vel)

    def step(self, action: list[float]) -> None:
        self._ops.step(action)

    def tick(self) -> None:
        self._ops.tick()


class RobotJointRuntime:
    def get_pos(self) -> float: ...
    def get_vel(self) -> float: ...
    def change_pos(self, pos: float) -> None: ...
    def change_vel(self, vel: float) -> None: ...

    def change_target_pos(self, pos: float) -> None: ...
    # NOTE: below two are future use,
    # since our team concluded that we are focuing on `pos` control
    def _change_target_vel(self, vel: float) -> None: ...
    def _change_target_force(self, force: float) -> None: ...


class RobotLinkRuntime:
    def get_pos(self) -> tuple[float, float, float]: ...
    def get_quat(self) -> tuple[float, float, float, float]: ...


class ICameraRuntimeOps(Protocol):
    def get_pos(self) -> tuple[float, float, float]: ...
    def get_quat(self) -> tuple[float, float, float, float]: ...

    def change_pos(self, pos: tuple[float, float, float]) -> None: ...
    def change_quat(self, quat: tuple[float, float, float, float]) -> None: ...

    def get_fov(self) -> float: ...
    def change_fov(self, fov: float) -> None: ...


class CameraRuntime:
    def __init__(self, entity_id: str, ops: ICameraRuntimeOps) -> None:
        self.id = entity_id
        self._ops = ops

    def get_pos(self) -> tuple[float, float, float]:
        return self._ops.get_pos()

    def get_quat(self) -> tuple[float, float, float, float]:
        return self._ops.get_quat()

    def change_pos(self, pos: tuple[float, float, float]) -> None:
        self._ops.change_pos(pos)

    def change_quat(self, quat: tuple[float, float, float, float]) -> None:
        self._ops.change_quat(quat)

    def get_fov(self) -> float:
        return self._ops.get_fov()

    def change_fov(self, fov: float) -> None:
        self._ops.change_fov(fov)


class ILightRuntimeOps(Protocol):
    def get_pos(self) -> tuple[float, float, float]: ...
    def get_quat(self) -> tuple[float, float, float, float]: ...

    def change_pos(self, pos: tuple[float, float, float]) -> None: ...
    def change_quat(self, quat: tuple[float, float, float, float]) -> None: ...

    def get_color(self) -> tuple[float, float, float]: ...
    def change_color(self, color: tuple[float, float, float]) -> None: ...

    def get_intensity(self) -> float: ...
    def change_intensity(self, intensity: float) -> None: ...


class ISpotLightRuntimeOps(Protocol):
    def get_angle(self) -> float: ...
    def change_angle(self, angle: float) -> None: ...


class IAreaLightRuntimeOps(Protocol):
    def get_area_size(self) -> tuple[float, float]: ...
    def change_area_size(self, width: float, height: float) -> None: ...


class IDirectionalLightRuntimeOps(Protocol):
    def get_direction(self) -> tuple[float, float, float]: ...
    def change_direction(self, direction: tuple[float, float, float]) -> None: ...
