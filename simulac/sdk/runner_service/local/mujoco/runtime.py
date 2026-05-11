from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.runner_service.common.model.runtime import IStuffRuntimeOps

if TYPE_CHECKING:
    import mujoco

    from .binding import MujocoRobotBinding, MujocoStuffBinding


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

    def change_friction(self, friction: float) -> None:
        if friction < 0:
            raise SimulacBaseError("friction must be positive")

        binding = self._binding
        model = self._model
        for geom_id in binding.geom_ids:
            model.geom_friction[geom_id][0] = float(friction)


class MujocoRobotRuntimeOps(IRobotRuntimeOps):
    def __init__(
        self,
        entity_id: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        binding: MujocoRobotBinding,
        *,
        on_after_step: Callable[[], None],
    ) -> None:
        self.id = entity_id
        self._model = model
        self._data = data
        self._binding = binding
        self._on_after_step = on_after_step
        self._actuator_by_id = {
            actuator.actuator_id: actuator for actuator in binding.actuators.values()
        }

    def get_base_pos(self) -> tuple[float, float, float]:
        binding = self._binding

        if binding.root_freejoint_id >= 0:
            qadr = int(self._model.jnt_qposadr[binding.root_freejoint_id])
            qpos = self._data.qpos[qadr : qadr + 3]
            return (float(qpos[0]), float(qpos[1]), float(qpos[2]))

        xpos = self._data.xpos[binding.root_body_id]
        return (float(xpos[0]), float(xpos[1]), float(xpos[2]))

    def get_base_quat(self) -> tuple[float, float, float, float]:
        binding = self._binding

        if binding.root_freejoint_id >= 0:
            qadr = int(self._model.jnt_qposadr[binding.root_freejoint_id])
            quat_wxyz = self._data.qpos[qadr + 3 : qadr + 7]
        else:
            quat_wxyz = self._data.xquat[binding.root_body_id]

        return (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )

    def get_joint_pos(self) -> list[float]:
        values: list[float] = []

        for joint_id in self._binding.joint_ids:
            if joint_id == self._binding.root_freejoint_id:
                continue

            joint_type = int(self._model.jnt_type[joint_id])
            qdim, _ = self.__joint_dims(joint_type)

            qadr = int(self._model.jnt_qposadr[joint_id])
            for offset in range(qdim):
                values.append(float(self._data.qpos[qadr + offset]))

        return values

    def get_joint_vel(self) -> list[float]:
        values: list[float] = []

        for joint_id in self._binding.joint_ids:
            if joint_id == self._binding.root_freejoint_id:
                continue

            joint_type = int(self._model.jnt_type[joint_id])
            _, vdim = self.__joint_dims(joint_type)

            dadr = int(self._model.jnt_dofadr[joint_id])
            for offset in range(vdim):
                values.append(float(self._data.qvel[dadr + offset]))

        return values

    def change_joint_pos(self, joint_pos: list[float]) -> None:
        cursor = 0

        for joint_id in self._binding.joint_ids:
            if joint_id == self._binding.root_freejoint_id:
                continue

            joint_type = int(self._model.jnt_type[joint_id])
            qdim, _ = self.__joint_dims(joint_type)

            qadr = int(self._model.jnt_qposadr[joint_id])
            if cursor + qdim > len(joint_pos):
                raise SimulacBaseError("joint_pos size mismatch")

            for offset in range(qdim):
                self._data.qpos[qadr + offset] = float(joint_pos[cursor + offset])

            cursor += qdim

        if cursor != len(joint_pos):
            raise SimulacBaseError("joint_pos size mismatch")

        mujoco.mj_forward(self._model, self._data)

    def change_joint_vel(self, joint_vel: list[float]) -> None:
        cursor = 0

        for joint_id in self._binding.joint_ids:
            if joint_id == self._binding.root_freejoint_id:
                continue

            joint_type = int(self._model.jnt_type[joint_id])
            _, vdim = self.__joint_dims(joint_type)

            dadr = int(self._model.jnt_dofadr[joint_id])
            if cursor + vdim > len(joint_vel):
                raise SimulacBaseError("joint_vel size mismatch")

            for offset in range(vdim):
                self._data.qvel[dadr + offset] = float(joint_vel[cursor + offset])

            cursor += vdim

        if cursor != len(joint_vel):
            raise SimulacBaseError("joint_vel size mismatch")

        mujoco.mj_forward(self._model, self._data)

    def _set_action(self, action: list[float]) -> None:
        actuator_ids = self._binding.actuator_ids

        if len(action) != len(actuator_ids):
            raise SimulacBaseError(
                f"Robot action size mismatch for {self.id!r}: "
                f"expected {len(actuator_ids)}, got {len(action)}"
            )

        for actuator_id, value in zip(actuator_ids, action):
            ctrl = float(value)
            self.__validate_ctrl(actuator_id, ctrl)
            self._data.ctrl[actuator_id] = ctrl

    def step(self, action: list[float]) -> None:
        self._set_action(action)
        mujoco.mj_step(self._model, self._data)
        self._on_after_step()

    def tick(self) -> None:
        mujoco.mj_step(self._model, self._data)
        self._on_after_step()

    def __validate_ctrl(self, actuator_id: int, value: float):
        actuator = self._actuator_by_id.get(actuator_id)
        if actuator is None:
            raise SimulacBaseError(
                f"Actuator {actuator_id} is not bound to robot {self.id!r}"
            )

        if actuator.ctrl_range is None:
            return

        lo, hi = actuator.ctrl_range
        if float(lo) <= value <= float(hi):
            return

        raise SimulacBaseError(
            f"Actuator control out of range for robot {self.id!r}: "
            f"actuator={actuator.name!r}, value={value}, "
            f"range=({float(lo)}, {float(hi)})"
        )

    def __joint_dims(self, joint_type: int) -> tuple[int, int]:
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        return 1, 1

