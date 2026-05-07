from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.runner_service.common.model.runtime import IStuffRuntimeOps

if TYPE_CHECKING:
    import mujoco

    from .binding import MujocoStuffBinding


def _wxyz_to_xyzw(quat: tuple[float, float, float, float]) -> list[float]:
    return [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]


def _xyzw_to_wxyz(quat: tuple[float, float, float, float]) -> list[float]:
    return [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]


class MujocoStuffRuntimeOps(IStuffRuntimeOps):
    def __init__(
        self,
        entity_id: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        binding: MujocoStuffBinding,
    ):
        self.id = entity_id
        self._model = model
        self._data = data
        self._binding = binding

    def get_pos(self) -> tuple[float, float, float]:
        if self._binding.root_freejoint_id >= 0:
            qadr = int(self._model.jnt_qposadr[self._binding.root_freejoint_id])
            qpos = self._data.qpos[qadr : qadr + 3]
            return (float(qpos[0]), float(qpos[1]), float(qpos[2]))

        xpos = self._data.xpos[self._binding.root_body_id]
        return (float(xpos[0]), float(xpos[1]), float(xpos[2]))

    def get_quat(self) -> tuple[float, float, float, float]:
        if self._binding.root_freejoint_id >= 0:
            qadr = int(self._model.jnt_qposadr[self._binding.root_freejoint_id])
            quat_wxyz = self._data.qpos[qadr + 3 : qadr + 7]
        else:
            quat_wxyz = self._data.xquat[self._binding.root_body_id]

        return (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )

    def get_friction(self) -> float:
        if not self._binding.geom_ids:
            raise SimulacBaseError("Stuff has no geoms")

        values = [
            float(self._model.geom_friction[geom_id][0])
            for geom_id in self._binding.geom_ids
        ]

        first = values[0]
        if any(value != first for value in values):
            # Mujoco support three types of frictions
            # `slide`, `torsion`, and `roll`, we only support `slide`
            # So no multiple friction allowed
            raise SimulacBaseError(
                "Stuff has multiple friction values; use per-geom friction API"
            )

        return first

    def get_mass(self):
        binding = self._binding
        model = self._model
        return sum(float(model.body_mass[body_id]) for body_id in binding.body_ids)

    def change_pos(self, pos: tuple[float, float, float]):
        binding = self._binding
        model = self._model
        data = self._data
        if binding.root_freejoint_id >= 0:
            qadr = int(model.jnt_qposadr[binding.root_freejoint_id])
            data.qpos[qadr : qadr + 3] = list(pos)
        else:
            model.body_pos[binding.root_body_id] = list(pos)
        self._sync_model()

    def change_quat(self, quat: tuple[float, float, float, float]):
        binding = self._binding
        model = self._model
        data = self._data

        quat_wxyz = _xyzw_to_wxyz(quat)
        if binding.root_freejoint_id >= 0:
            qadr = int(model.jnt_qposadr[binding.root_freejoint_id])
            data.qpos[qadr + 3 : qadr + 7] = quat_wxyz
        else:
            model.body_quat[binding.root_body_id] = quat_wxyz
        self._sync_model()

    def change_mass(self, mass: float) -> None:
        if mass <= 0:
            raise SimulacBaseError("mass must be positive")

        binding = self._binding
        model = self._model
        data = self._data

        current_mass = self.get_mass()
        if current_mass <= 0:
            raise SimulacBaseError(
                "current mass must be positive. Something went wrong!"
            )

        ratio = mass / current_mass

        for body_id in binding.body_ids:
            model.body_mass[body_id] *= ratio
            model.body_inertia[body_id][:] *= ratio

        mujoco.mj_setConst(model, data)

        self._sync_model()

    def change_friction(self, friction: float) -> None:
        if friction < 0:
            raise SimulacBaseError("friction must be positive")

        binding = self._binding
        model = self._model
        for geom_id in binding.geom_ids:
            model.geom_friction[geom_id][0] = float(friction)

        self._sync_model()

    def _sync_model(self):
        """`mj_forward` doesn't increase time, only recalculate contact, geometry, etcs..."""
        mujoco.mj_forward(self._model, self._data)
