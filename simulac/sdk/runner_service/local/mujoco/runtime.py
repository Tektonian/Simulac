from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

from simulac.base.utils.rotation import euler_to_quat
from simulac.sdk.runner_service.common.model.runtime import IStuffRuntimeOps

if TYPE_CHECKING:
    import mujoco

    from .binding import MujocoEntityBinding


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
        binding: MujocoEntityBinding,
    ):
        self.id = entity_id
        self._model = model
        self._data = data
        self._binding = binding

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

    def change_rot(self, rot: tuple[float, float, float]):
        binding = self._binding
        model = self._model
        data = self._data

        quat_wxyz = _xyzw_to_wxyz(euler_to_quat(*rot))
        if binding.root_freejoint_id >= 0:
            qadr = int(model.jnt_qposadr[binding.root_freejoint_id])
            data.qpos[qadr + 3 : qadr + 7] = quat_wxyz
        else:
            model.body_quat[binding.root_body_id] = quat_wxyz
        self._sync_model()

    def change_mass(self, mass: float) -> None: ...
    def change_size(self, size: tuple[float, float, float]) -> None: ...
    def change_fixed(self, is_fixed: bool) -> None: ...
    def change_friction(self, friction: float) -> None: ...
    def change_density(self, density: float) -> None: ...

    def _sync_model(self):
        """`mj_forward` doesn't increase time, only recalculate contact, geometry, etcs..."""
        mujoco.mj_forward(self._model, self._data)
