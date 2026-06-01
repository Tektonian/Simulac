from __future__ import annotations

from array import array
from math import sqrt
from typing import TYPE_CHECKING, Callable

import mujoco

from simulac.base.error.error import SimulacBaseError
from simulac.base.types.geometry import Vec3
from simulac.sdk.environment_service.common.model.ref import ColliderRef
from simulac.sdk.runner_service.common.model.runtime import (
    ICameraRuntimeOps,
    ILightRuntimeOps,
    IRobotRuntimeOps,
    IStuffRuntimeOps,
)


def _wxyz_to_xyzw(quat: tuple[float, float, float, float]) -> list[float]:
    return [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]


def _xyzw_to_wxyz(quat: tuple[float, float, float, float]) -> list[float]:
    return [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]


class MujocoRuntimeStateOps:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        step_count: Callable[[], int],
        stuff_bindings: dict[str, MujocoStuffBinding],
        machine_bindings: dict[str, MujocoRobotBinding],
    ) -> None:
        self._model = model
        self._data = data
        self._step_count = step_count
        self._stuff_bindings = stuff_bindings
        self._machine_bindings = machine_bindings

    def get_time(self) -> float:
        return float(self._data.time)

    def get_step_count(self) -> int:
        return int(self._step_count())

    def contact_indices(self, a: object, b: object) -> tuple[int, ...]:
        a_geom_ids = self._resolve_geom_ids(a)
        b_geom_ids = self._resolve_geom_ids(b)

        indices: list[int] = []
        for contact_idx in range(int(self._data.ncon)):
            contact = self._data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if (geom1 in a_geom_ids and geom2 in b_geom_ids) or (
                geom2 in a_geom_ids and geom1 in b_geom_ids
            ):
                indices.append(contact_idx)

        return tuple(indices)

    def contact_point(self, contact_index: int) -> Vec3:
        self._validate_contact_index(contact_index)
        point = self._data.contact[contact_index].pos
        return (float(point[0]), float(point[1]), float(point[2]))

    def contact_normal(self, contact_index: int) -> Vec3:
        self._validate_contact_index(contact_index)
        frame = self._data.contact[contact_index].frame
        return (float(frame[0]), float(frame[1]), float(frame[2]))

    def contact_force(self, contact_index: int) -> float | None:
        self._validate_contact_index(contact_index)
        contact_force = array("d", [0.0] * 6)
        try:
            mujoco.mj_contactForce(
                self._model,
                self._data,
                contact_index,
                contact_force,
            )
        except TypeError:
            return None

        return sqrt(
            float(contact_force[0]) ** 2
            + float(contact_force[1]) ** 2
            + float(contact_force[2]) ** 2
        )

    def _validate_contact_index(self, contact_index: int) -> None:
        if contact_index < 0 or contact_index >= int(self._data.ncon):
            raise SimulacBaseError(
                f"Contact index {contact_index} is out of range: ncon={self._data.ncon}"
            )

    def _resolve_geom_ids(self, target: object) -> set[int]:
        if isinstance(target, ColliderRef):
            return {self._named_geom_id(target.entity_id, target.name)}

        entity_id = self._entity_id_from_runtime(target)
        if entity_id is None:
            raise SimulacBaseError(f"Unsupported contact target: {target!r}")

        binding = self._stuff_bindings.get(entity_id)
        if binding is not None:
            return set(binding.geom_ids)

        machine_binding = self._machine_bindings.get(entity_id)
        if machine_binding is not None:
            return set(machine_binding.geom_ids)

        raise SimulacBaseError(f"No contact-capable entity named {entity_id!r}")

    def _named_geom_id(self, entity_id: str, name: str) -> int:
        full_name = f"{entity_id}/{name}"
        geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, full_name)
        if geom_id < 0:
            raise SimulacBaseError(f"No MuJoCo geom named {full_name!r}")
        return int(geom_id)

    def _entity_id_from_runtime(self, target: object) -> str | None:
        entity_id = getattr(target, "id", None)
        if isinstance(entity_id, str):
            return entity_id

        runtime = getattr(target, "_runtime", None)
        runtime_id = getattr(runtime, "id", None)
        if isinstance(runtime_id, str):
            return runtime_id

        return None


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

    def get_joint_type(self, name: str):
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            return "hinge"
        if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            return "slide"
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return "ball"
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return "free"
        raise SimulacBaseError(f"Unsupported MuJoCo joint type: {joint_type}")

    def get_joint_scalar_pos(self, name: str) -> float:
        joint = self.__require_joint(name)
        self.__require_scalar_joint(name, joint.joint_id)
        return float(self._data.qpos[joint.qpos_addr])

    def get_joint_scalar_vel(self, name: str) -> float:
        joint = self.__require_joint(name)
        self.__require_scalar_joint(name, joint.joint_id)
        return float(self._data.qvel[joint.qvel_addr])

    def get_joint_free_pos(self, name: str) -> tuple[float, float, float]:
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
            raise SimulacBaseError(f"Joint {name!r} is not a free joint")
        pos = self._data.qpos[joint.qpos_addr : joint.qpos_addr + 3]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def get_joint_quat(self, name: str) -> tuple[float, float, float, float]:
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            quat_wxyz = self._data.qpos[joint.qpos_addr : joint.qpos_addr + 4]
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            quat_wxyz = self._data.qpos[joint.qpos_addr + 3 : joint.qpos_addr + 7]
        else:
            raise SimulacBaseError(f"Joint {name!r} has no quaternion state")

        return (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )

    def get_joint_linear_vel(self, name: str) -> tuple[float, float, float]:
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
            raise SimulacBaseError(f"Joint {name!r} has no linear velocity vector")
        vel = self._data.qvel[joint.qvel_addr : joint.qvel_addr + 3]
        return (float(vel[0]), float(vel[1]), float(vel[2]))

    def get_joint_angular_vel(self, name: str) -> tuple[float, float, float]:
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            vel = self._data.qvel[joint.qvel_addr : joint.qvel_addr + 3]
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            vel = self._data.qvel[joint.qvel_addr + 3 : joint.qvel_addr + 6]
        else:
            raise SimulacBaseError(f"Joint {name!r} has no angular velocity vector")
        return (float(vel[0]), float(vel[1]), float(vel[2]))

    def get_joint_axis(self, name: str) -> tuple[float, float, float]:
        joint = self.__require_joint(name)
        self.__require_scalar_joint(name, joint.joint_id)
        return joint.axis

    def get_joint_limited(self, name: str) -> bool:
        joint = self.__require_joint(name)
        return bool(self._model.jnt_limited[joint.joint_id])

    def get_joint_range(self, name: str) -> tuple[float, float] | None:
        joint = self.__require_joint(name)
        if not bool(self._model.jnt_limited[joint.joint_id]):
            return None
        return (
            float(self._model.jnt_range[joint.joint_id][0]),
            float(self._model.jnt_range[joint.joint_id][1]),
        )

    def get_joint_force(self, name: str) -> float:
        joint = self.__require_joint(name)
        self.__require_scalar_joint(name, joint.joint_id)
        return float(self._data.qfrc_actuator[joint.qvel_addr])

    def __require_joint(self, name: str):
        joint = self._binding.joints.get(name)
        if joint is None:
            known = ", ".join(sorted(self._binding.joints)) or "<none>"
            raise SimulacBaseError(
                f"No joint {name!r} on stuff {self.id!r}. Known joints: {known}"
            )
        return joint

    def __require_scalar_joint(self, name: str, joint_id: int) -> None:
        joint_type = int(self._model.jnt_type[joint_id])
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise SimulacBaseError(f"Joint {name!r} has no scalar position/velocity")


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

    def get_joint_type(self, name: str):
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            return "hinge"
        if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            return "slide"
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return "ball"
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return "free"
        raise SimulacBaseError(f"Unsupported MuJoCo joint type: {joint_type}")

    def get_joint_scalar_pos(self, name: str) -> float:
        joint = self.__require_joint(name)
        self.__require_scalar_joint(name, joint.joint_id)
        return float(self._data.qpos[joint.qpos_addr])

    def get_joint_scalar_vel(self, name: str) -> float:
        joint = self.__require_joint(name)
        self.__require_scalar_joint(name, joint.joint_id)
        return float(self._data.qvel[joint.qvel_addr])

    def get_joint_free_pos(self, name: str) -> tuple[float, float, float]:
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
            raise SimulacBaseError(f"Joint {name!r} is not a free joint")
        pos = self._data.qpos[joint.qpos_addr : joint.qpos_addr + 3]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def get_joint_quat(self, name: str) -> tuple[float, float, float, float]:
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            quat_wxyz = self._data.qpos[joint.qpos_addr : joint.qpos_addr + 4]
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            quat_wxyz = self._data.qpos[joint.qpos_addr + 3 : joint.qpos_addr + 7]
        else:
            raise SimulacBaseError(f"Joint {name!r} has no quaternion state")
        return (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )

    def get_joint_linear_vel(self, name: str) -> tuple[float, float, float]:
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
            raise SimulacBaseError(f"Joint {name!r} has no linear velocity vector")
        linear_vel = self._data.qvel[joint.qvel_addr : joint.qvel_addr + 3]
        return (float(linear_vel[0]), float(linear_vel[1]), float(linear_vel[2]))

    def get_joint_angular_vel(self, name: str) -> tuple[float, float, float]:
        joint = self.__require_joint(name)
        joint_type = int(self._model.jnt_type[joint.joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            angular_vel = self._data.qvel[joint.qvel_addr : joint.qvel_addr + 3]
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            angular_vel = self._data.qvel[joint.qvel_addr + 3 : joint.qvel_addr + 6]
        else:
            raise SimulacBaseError(f"Joint {name!r} has no angular velocity vector")
        return (
            float(angular_vel[0]),
            float(angular_vel[1]),
            float(angular_vel[2]),
        )

    def get_joint_axis(self, name: str) -> tuple[float, float, float]:
        joint = self.__require_joint(name)
        self.__require_scalar_joint(name, joint.joint_id)
        return joint.axis

    def get_joint_limited(self, name: str) -> bool:
        joint = self.__require_joint(name)
        return bool(self._model.jnt_limited[joint.joint_id])

    def get_joint_range(self, name: str) -> tuple[float, float] | None:
        joint = self.__require_joint(name)
        if not bool(self._model.jnt_limited[joint.joint_id]):
            return None
        return (
            float(self._model.jnt_range[joint.joint_id][0]),
            float(self._model.jnt_range[joint.joint_id][1]),
        )

    def get_site_pos(self, name):
        full_name = f"{self.id}/{name}"
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, full_name)
        if site_id < 0:
            raise SimulacBaseError(f"No site {name!r} on robot {self.id!r}")
        pos = self._data.site_xpos[site_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def get_site_quat(self, name: str) -> tuple[float, float, float, float]:
        full_name = f"{self.id}/{name}"
        site_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_SITE,
            full_name,
        )
        if site_id < 0:
            raise SimulacBaseError(f"No site {name!r} on robot {self.id!r}")

        quat_wxyz = [0.0, 0.0, 0.0, 0.0]
        mujoco.mju_mat2Quat(quat_wxyz, self._data.site_xmat[site_id])

        return (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )

    def get_site_linear_vel(self, name: str) -> tuple[float, float, float]:
        site_id = self.__require_site_id(name)
        vel = self._object_velocity(mujoco.mjtObj.mjOBJ_SITE, site_id)
        return (
            float(vel[3]),
            float(vel[4]),
            float(vel[5]),
        )

    def get_site_angular_vel(self, name: str) -> tuple[float, float, float]:
        site_id = self.__require_site_id(name)
        vel = self._object_velocity(mujoco.mjtObj.mjOBJ_SITE, site_id)
        return (
            float(vel[0]),
            float(vel[1]),
            float(vel[2]),
        )

    def get_sensor_value(self, name: str) -> tuple[float, ...]:
        sensor = self.__require_sensor(name)
        values = self._data.sensordata[sensor.adr : sensor.adr + sensor.dim]
        return tuple(float(value) for value in values)

    def get_sensor_dim(self, name: str) -> int:
        return self.__require_sensor(name).dim

    def get_sensor_type(self, name: str) -> int:
        return self.__require_sensor(name).sensor_type

    def __require_sensor(self, name: str):
        sensor = self._binding.sensors.get(name)
        if sensor is None:
            known = ", ".join(sorted(self._binding.sensors)) or "<none>"
            raise SimulacBaseError(
                f"No sensor {name!r} on robot {self.id!r}. Known sensors: {known}"
            )
        return sensor

    def get_link_pos(self, name: str) -> tuple[float, float, float]:
        link = self._binding.links.get(name)
        if link is None:
            known = ", ".join(sorted(self._binding.links)) or "<none>"
            raise SimulacBaseError(
                f"No link {name!r} on robot {self.id!r}. Known links: {known}"
            )

        pos = self._data.xpos[link.body_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def get_link_quat(self, name: str) -> tuple[float, float, float, float]:
        link = self._binding.links.get(name)
        if link is None:
            known = ", ".join(sorted(self._binding.links)) or "<none>"
            raise SimulacBaseError(
                f"No link {name!r} on robot {self.id!r}. Known links: {known}"
            )

        quat_wxyz = self._data.xquat[link.body_id]
        return (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )

    def get_link_linear_vel(self, name: str) -> tuple[float, float, float]:
        link = self.__require_link(name)
        vel = self._object_velocity(mujoco.mjtObj.mjOBJ_BODY, link.body_id)
        return (
            float(vel[3]),
            float(vel[4]),
            float(vel[5]),
        )

    def get_link_angular_vel(self, name: str) -> tuple[float, float, float]:
        link = self.__require_link(name)
        vel = self._object_velocity(mujoco.mjtObj.mjOBJ_BODY, link.body_id)
        return (
            float(vel[0]),
            float(vel[1]),
            float(vel[2]),
        )

    def get_joint_force(self, name: str) -> float:
        joint = self.__require_joint(name)
        self.__require_scalar_joint(name, joint.joint_id)
        return float(self._data.qfrc_actuator[joint.qvel_addr])

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

    def set_control(self, action: list[float]) -> None:
        self._set_action(action)

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

    def __require_joint(self, name: str):
        joint = self._binding.joints.get(name)
        if joint is None:
            known = ", ".join(sorted(self._binding.joints)) or "<none>"
            raise SimulacBaseError(
                f"No joint {name!r} on robot {self.id!r}. Known joints: {known}"
            )
        return joint

    def __require_link(self, name: str):
        link = self._binding.links.get(name)
        if link is None:
            known = ", ".join(sorted(self._binding.links)) or "<none>"
            raise SimulacBaseError(
                f"No link {name!r} on robot {self.id!r}. Known links: {known}"
            )
        return link

    def __require_site_id(self, name: str) -> int:
        full_name = f"{self.id}/{name}"
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, full_name)
        if site_id < 0:
            raise SimulacBaseError(f"No site {name!r} on robot {self.id!r}")
        return site_id

    def __require_scalar_joint(self, name: str, joint_id: int) -> None:
        joint_type = int(self._model.jnt_type[joint_id])
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise SimulacBaseError(f"Joint {name!r} has no scalar position/velocity")

    def _object_velocity(self, obj_type: mujoco.mjtObj, obj_id: int):
        vel = self._data.cvel[0].copy()
        mujoco.mj_objectVelocity(self._model, self._data, obj_type, obj_id, vel, 0)
        return vel


class MujocoCameraRuntimeOps(ICameraRuntimeOps):
    def __init__(
        self,
        entity_id: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        binding: MujocoCameraBinding,
    ) -> None:
        self.id = entity_id
        self._model = model
        self._data = data
        self._binding = binding

    def get_pos(self) -> tuple[float, float, float]:
        pos = self._data.xpos[self._binding.root_body_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def get_quat(self) -> tuple[float, float, float, float]:
        quat_wxyz = self._data.xquat[self._binding.root_body_id]
        return (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )

    def change_pos(self, pos: tuple[float, float, float]) -> None:
        self._model.body_pos[self._binding.root_body_id] = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        )
        mujoco.mj_forward(self._model, self._data)

    def change_quat(self, quat: tuple[float, float, float, float]) -> None:
        self._model.body_quat[self._binding.root_body_id] = (
            float(quat[3]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
        )
        mujoco.mj_forward(self._model, self._data)

    def get_fov(self) -> float:
        return float(self._model.cam_fovy[self._binding.camera_id])

    def change_fov(self, fov: float) -> None:
        if fov <= 0:
            raise SimulacBaseError("camera fov must be positive")

        self._model.cam_fovy[self._binding.camera_id] = float(fov)
        mujoco.mj_forward(self._model, self._data)


class MujocoLightRuntimeOps(ICameraRuntimeOps):
    def get_pos(self) -> tuple[float, float, float]: ...
    def get_quat(self) -> tuple[float, float, float, float]: ...

    def change_pos(self, pos: tuple[float, float, float]) -> None: ...
    def change_quat(self, quat: tuple[float, float, float, float]) -> None: ...

    def get_color(self) -> tuple[float, float, float]: ...
    def change_color(self, color: tuple[float, float, float]) -> None: ...

    def get_intensity(self) -> float: ...
    def change_intensity(self, intensity: float) -> None: ...
