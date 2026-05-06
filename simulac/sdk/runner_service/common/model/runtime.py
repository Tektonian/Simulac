from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class IStuffRuntimeOps(Protocol):
    def get_pos(self) -> tuple[float, float, float]: ...
    def get_quat(self) -> tuple[float, float, float, float]: ...
    def get_mass(self) -> float: ...
    def get_friction(self) -> float: ...

    def change_pos(self, pos: tuple[float, float, float]) -> None: ...
    def change_quat(self, quat: tuple[float, float, float, float]) -> None: ...
    def change_mass(self, mass: float) -> None: ...
    def change_size(self, size: tuple[float, float, float]) -> None: ...
    def change_fixed(self, is_fixed: bool) -> None: ...
    def change_friction(self, friction: float) -> None: ...
    def change_density(self, density: float) -> None: ...


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

