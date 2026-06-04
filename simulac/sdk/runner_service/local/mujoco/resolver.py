from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal, cast, overload

import mujoco

from simulac.base.error.error import SimulacBaseError
from simulac.base.types.geometry import Quat, Vec3
from simulac.sdk.environment_service.common.model.ref import (
    AnchorPosRef,
    AnchorRef,
    BodyPosRef,
    BoundsCenterRef,
    BoundsMaxRef,
    BoundsMinRef,
    BoundsSizeRef,
    CameraPosRef,
    ColliderCenterRef,
    ColliderRef,
    EntityPosRef,
    EntityQuatRef,
    EntityRef,
    EntityRotRef,
    JointAxisRef,
    JointRef,
    LightPosRef,
    PointRefBase,
    PointRefType,
    RefBase,
    SupportPointRef,
    SurfaceCenterRef,
    SurfaceNormalRef,
    SurfaceSampleRef,
    WorldPointRef,
)
from simulac.sdk.environment_service.common.randomize import Randomizable
from simulac.sdk.runner_service.common.sampler import ResetSampler
from simulac.sdk.runner_service.local.mujoco.binding import (
    MujocoCameraBinding,
    MujocoLightBinding,
    MujocoRobotBinding,
    MujocoStuffBinding,
)

from .runtime import _wxyz_to_xyzw

_AXIS: dict[str, tuple[float, float, float]] = {
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "front": (0.0, 1.0, 0.0),
    "back": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


@dataclass(frozen=True, slots=True)
class _ResolvedJointBase:
    entity_id: str
    name: str
    joint_id: int
    qpos_addr: int
    qvel_addr: int


@dataclass(frozen=True, slots=True)
class _ResolvedHingeJoint(_ResolvedJointBase):
    type: Literal["hinge"]
    pos: float
    vel: float
    axis: Vec3
    limited: bool
    range: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class _ResolvedSlideJoint(_ResolvedJointBase):
    type: Literal["slide"]
    pos: float
    vel: float
    axis: Vec3
    limited: bool
    range: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class _ResolvedBallJoint(_ResolvedJointBase):
    type: Literal["ball"]
    quat: Quat
    angular_vel: Vec3


@dataclass(frozen=True, slots=True)
class _ResolvedFreeJoint(_ResolvedJointBase):
    type: Literal["free"]
    pos: Vec3
    quat: Quat
    linear_vel: Vec3
    angular_vel: Vec3


type _ResolvedJoint = (
    _ResolvedHingeJoint | _ResolvedSlideJoint | _ResolvedBallJoint | _ResolvedFreeJoint
)


class MujocoRefResolver:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        stuff_bindings: dict[str, MujocoStuffBinding],
        machine_bindings: dict[str, MujocoRobotBinding],
        camera_bindings: dict[str, MujocoCameraBinding],
    ) -> None:
        self.model = model
        self.data = data
        self._stuff_bindings = stuff_bindings
        self._machine_bindings = machine_bindings
        self._camera_bindings = camera_bindings

    def resolve_frame(
        self, ref: EntityRef | AnchorPosRef | ColliderRef
    ) -> tuple[Vec3, Quat]:
        data = self.data

        if isinstance(ref, EntityRef):
            binding = self._binding(ref.entity_id)
            pos = data.xpos[binding.root_body_id]
            quat = data.xquat[binding.root_body_id]
            return (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                (float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])),
            )

        if isinstance(ref, AnchorRef):
            site_id = self._named_id(
                mujoco.mjtObj.mjOBJ_SITE,
                ref.entity_id,
                ref.name,
            )
            pos = data.site_xpos[site_id]
            xmat = data.site_xmat[site_id]
            return (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                self.__mat_to_quat_xyzw(xmat),
            )

        if isinstance(ref, ColliderRef):
            geom_id = self._named_id(
                mujoco.mjtObj.mjOBJ_GEOM,
                ref.entity_id,
                ref.name,
            )
            pos = data.geom_xpos[geom_id]
            xmat = data.geom_xmat[geom_id]
            return (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                self.__mat_to_quat_xyzw(xmat),
            )

        raise SimulacBaseError(f"Unsupported frame ref: {ref}")

    def resolve_point(self, ref: RefBase) -> Vec3:
        if isinstance(ref, WorldPointRef):
            if not isinstance(ref.pos, tuple):
                raise SimulacBaseError(
                    f"Resolved ref point must be tuple {ref}/{ref.pos}"
                )
            return (float(ref.pos[0]), float(ref.pos[1]), float(ref.pos[2]))

        if isinstance(ref, EntityPosRef):
            body_id = self._binding(ref.entity_id).root_body_id
            return self.data.xpos[body_id].copy().tolist()

        if isinstance(ref, BodyPosRef):
            body_id = self._body_id(ref.entity_id, ref.name)
            return self.data.xpos[body_id].copy().tolist()

        if isinstance(ref, AnchorPosRef):
            site_id = self._named_id(
                mujoco.mjtObj.mjOBJ_SITE,
                ref.entity_id,
                ref.name,
            )
            return self.data.site_xpos[site_id].copy().tolist()

        if isinstance(ref, ColliderCenterRef):
            geom_id = self._named_id(
                mujoco.mjtObj.mjOBJ_GEOM,
                ref.entity_id,
                ref.name,
            )
            return self.data.geom_xpos[geom_id].copy().tolist()

        if isinstance(ref, BoundsCenterRef):
            lo, hi = self.resolve_bounds(ref.entity_id, ref.collider_name)
            return (
                (lo[0] + hi[0]) * 0.5,
                (lo[1] + hi[1]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
            )

        if isinstance(ref, BoundsMinRef):
            lo, _ = self.resolve_bounds(ref.entity_id, ref.collider_name)
            return lo

        if isinstance(ref, BoundsMaxRef):
            _, hi = self.resolve_bounds(ref.entity_id, ref.collider_name)
            return hi

        if isinstance(ref, (SurfaceCenterRef, SurfaceSampleRef)):
            center, _ = self._surface(ref)
            return center

        if isinstance(ref, SupportPointRef):
            return self._support(ref)

        if isinstance(ref, CameraPosRef):
            camera_name = ref.name or "__root__"
            camera_id = self._named_id(
                mujoco.mjtObj.mjOBJ_CAMERA,
                ref.entity_id,
                camera_name,
            )
            return self.data.cam_xpos[camera_id].copy().tolist()

        if isinstance(ref, LightPosRef):
            light_name = ref.name or "__root__"
            light_id = self._named_id(
                mujoco.mjtObj.mjOBJ_LIGHT,
                ref.entity_id,
                light_name,
            )
            return self.data.light_xpos[light_id].copy().tolist()

        raise SimulacBaseError(f"Unsupported point ref: {ref}")

    @overload
    def resolve_vector(self, ref: EntityQuatRef) -> Quat: ...
    @overload
    def resolve_vector(self, ref: BoundsSizeRef) -> Vec3: ...
    @overload
    def resolve_vector(self, ref: SurfaceNormalRef) -> Vec3: ...
    @overload
    def resolve_vector(self, ref: JointAxisRef) -> Vec3: ...
    def resolve_vector(
        self,
        ref: EntityQuatRef
        | BoundsSizeRef
        | SurfaceNormalRef
        | JointAxisRef
        | EntityRotRef,
    ) -> Quat | Vec3:
        if isinstance(ref, EntityQuatRef):
            body_id = self._binding(ref.entity_id).root_body_id
            quat_wxyz = self.data.xquat[body_id]
            return (
                float(quat_wxyz[1]),
                float(quat_wxyz[2]),
                float(quat_wxyz[3]),
                float(quat_wxyz[0]),
            )

        if isinstance(ref, EntityRotRef):
            raise SimulacBaseError(
                "EntityRotRef rot should be change to quat in previous state"
            )

        if isinstance(ref, BoundsSizeRef):
            lo, hi = self.resolve_bounds(ref.entity_id, ref.collider_name)
            return (
                float(hi[0]) - float(lo[0]),
                float(hi[1]) - float(lo[1]),
                float(hi[2]) - float(lo[2]),
            )

        if isinstance(ref, SurfaceNormalRef):
            _, normal = self._surface(ref)
            return (
                float(normal[0]),
                float(normal[1]),
                float(normal[2]),
            )

        if isinstance(ref, JointAxisRef):  # pyright: ignore[reportUnnecessaryIsInstance]
            joint_id = self._joint_id(ref.entity_id, ref.name)
            axis = self.model.jnt_axis[joint_id]
            return (
                float(axis[0]),
                float(axis[1]),
                float(axis[2]),
            )

        raise SimulacBaseError(f"Unsupported vector ref: {ref}")

    def resolve_joint(self, ref: JointRef) -> _ResolvedJoint:
        joint_id = self._joint_id(ref.entity_id, ref.name)
        joint_type = int(self.model.jnt_type[joint_id])

        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        qvel_addr = int(self.model.jnt_dofadr[joint_id])

        limited = bool(self.model.jnt_limited[joint_id])
        joint_range = None
        if limited:
            joint_range = (
                float(self.model.jnt_range[joint_id][0]),
                float(self.model.jnt_range[joint_id][1]),
            )

        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            axis = self.model.jnt_axis[joint_id]

            return _ResolvedHingeJoint(
                entity_id=ref.entity_id,
                name=ref.name,
                joint_id=joint_id,
                qpos_addr=qpos_addr,
                qvel_addr=qvel_addr,
                type="hinge",
                pos=float(self.data.qpos[qpos_addr]),
                vel=float(self.data.qvel[qvel_addr]),
                axis=(float(axis[0]), float(axis[1]), float(axis[2])),
                limited=limited,
                range=joint_range,
            )

        if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            axis = self.model.jnt_axis[joint_id]

            return _ResolvedSlideJoint(
                entity_id=ref.entity_id,
                name=ref.name,
                joint_id=joint_id,
                qpos_addr=qpos_addr,
                qvel_addr=qvel_addr,
                type="slide",
                pos=float(self.data.qpos[qpos_addr]),
                vel=float(self.data.qvel[qvel_addr]),
                axis=(float(axis[0]), float(axis[1]), float(axis[2])),
                limited=limited,
                range=joint_range,
            )

        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            quat_wxyz = self.data.qpos[qpos_addr : qpos_addr + 4]
            angular_vel = self.data.qvel[qvel_addr : qvel_addr + 3]

            return _ResolvedBallJoint(
                entity_id=ref.entity_id,
                name=ref.name,
                joint_id=joint_id,
                qpos_addr=qpos_addr,
                qvel_addr=qvel_addr,
                type="ball",
                quat=(
                    float(quat_wxyz[1]),
                    float(quat_wxyz[2]),
                    float(quat_wxyz[3]),
                    float(quat_wxyz[0]),
                ),
                angular_vel=(
                    float(angular_vel[0]),
                    float(angular_vel[1]),
                    float(angular_vel[2]),
                ),
            )

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            pos = self.data.qpos[qpos_addr : qpos_addr + 3]
            quat_wxyz = self.data.qpos[qpos_addr + 3 : qpos_addr + 7]
            linear_vel = self.data.qvel[qvel_addr : qvel_addr + 3]
            angular_vel = self.data.qvel[qvel_addr + 3 : qvel_addr + 6]

            return _ResolvedFreeJoint(
                entity_id=ref.entity_id,
                name=ref.name,
                joint_id=joint_id,
                qpos_addr=qpos_addr,
                qvel_addr=qvel_addr,
                type="free",
                pos=(
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                ),
                quat=(
                    float(quat_wxyz[1]),
                    float(quat_wxyz[2]),
                    float(quat_wxyz[3]),
                    float(quat_wxyz[0]),
                ),
                linear_vel=(
                    float(linear_vel[0]),
                    float(linear_vel[1]),
                    float(linear_vel[2]),
                ),
                angular_vel=(
                    float(angular_vel[0]),
                    float(angular_vel[1]),
                    float(angular_vel[2]),
                ),
            )

        raise SimulacBaseError(f"Unsupported MuJoCo joint type: {joint_type}")

    def resolve_bounds(
        self,
        entity_id: str,
        collider_name: str,
    ) -> tuple[Vec3, Vec3]:
        """Calculate AABB.
        Reference:
            https://en.wikipedia.org/wiki/Minimum_bounding_box#Axis-aligned_minimum_bounding_box
        """
        geom_id = self._named_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            entity_id,
            collider_name,
        )

        geom_type = int(self.model.geom_type[geom_id])
        center = self.data.geom_xpos[geom_id]
        xmat = self.data.geom_xmat[geom_id]
        size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half_extents = [float(size[0]), float(size[1]), float(size[2])]
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(size[0])
            half_extents = [r, r, r]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r = float(size[0])
            h = float(size[1])
            half_extents = [r, r, h]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r = float(size[0])
            h = float(size[1])
            half_extents = [r, r, h + r]
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            half_extents = [float(size[0]), float(size[1]), float(size[2])]
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                raise SimulacBaseError(f"No mesh data for geom {geom_id}")

            vert_adr = int(self.model.mesh_vertadr[mesh_id])
            vert_num = int(self.model.mesh_vertnum[mesh_id])
            verts = self.model.mesh_vert[vert_adr : vert_adr + vert_num]

            local_min = verts.min(axis=0)
            local_max = verts.max(axis=0)

            local_center = [
                (float(local_min[0]) + float(local_max[0])) * 0.5,
                (float(local_min[1]) + float(local_max[1])) * 0.5,
                (float(local_min[2]) + float(local_max[2])) * 0.5,
            ]
            half_extents = [
                (float(local_max[0]) - float(local_min[0])) * 0.5,
                (float(local_max[1]) - float(local_min[1])) * 0.5,
                (float(local_max[2]) - float(local_min[2])) * 0.5,
            ]
            center = [
                float(center[0])
                + float(xmat[0]) * local_center[0]
                + float(xmat[1]) * local_center[1]
                + float(xmat[2]) * local_center[2],
                float(center[1])
                + float(xmat[3]) * local_center[0]
                + float(xmat[4]) * local_center[1]
                + float(xmat[5]) * local_center[2],
                float(center[2])
                + float(xmat[6]) * local_center[0]
                + float(xmat[7]) * local_center[1]
                + float(xmat[8]) * local_center[2],
            ]

        else:
            raise SimulacBaseError(
                f"Bounds resolution is unsupported for geom type {geom_type}"
            )

        axes = [
            [float(xmat[0]), float(xmat[3]), float(xmat[6])],
            [float(xmat[1]), float(xmat[4]), float(xmat[7])],
            [float(xmat[2]), float(xmat[5]), float(xmat[8])],
        ]

        world_half = [
            abs(axes[0][0]) * half_extents[0]
            + abs(axes[1][0]) * half_extents[1]
            + abs(axes[2][0]) * half_extents[2],
            abs(axes[0][1]) * half_extents[0]
            + abs(axes[1][1]) * half_extents[1]
            + abs(axes[2][1]) * half_extents[2],
            abs(axes[0][2]) * half_extents[0]
            + abs(axes[1][2]) * half_extents[1]
            + abs(axes[2][2]) * half_extents[2],
        ]

        return (
            (
                float(center[0]) - world_half[0],
                float(center[1]) - world_half[1],
                float(center[2]) - world_half[2],
            ),
            (
                float(center[0]) + world_half[0],
                float(center[1]) + world_half[1],
                float(center[2]) + world_half[2],
            ),
        )

    def _binding(self, entity_id: str) -> MujocoStuffBinding | MujocoRobotBinding:
        binding = self._stuff_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self._machine_bindings.get(entity_id)
        if binding is not None:
            return binding

        raise SimulacBaseError(f"No MuJoCo binding for entity {entity_id!r}")

    def _body_id(self, entity_id: str, name: str | None) -> int:
        binding = self._binding(entity_id)

        if name is None:
            return binding.root_body_id

        if isinstance(binding, MujocoRobotBinding):
            link = binding.links.get(name)
            if link is not None:
                return link.body_id

        return self._named_id(mujoco.mjtObj.mjOBJ_BODY, entity_id, name)

    def _joint_id(self, entity_id: str, name: str) -> int:
        binding = self._binding(entity_id)

        if isinstance(binding, MujocoRobotBinding):
            joint = binding.joints.get(name)
            if joint is not None:
                return joint.joint_id

        return self._named_id(mujoco.mjtObj.mjOBJ_JOINT, entity_id, name)

    def _named_id(self, obj_type: mujoco.mjtObj, entity_id: str, name: str) -> int:
        full_name = f"{entity_id}/{name}"
        idx = mujoco.mj_name2id(self.model, obj_type, full_name)
        if idx < 0:
            raise SimulacBaseError(f"No MuJoCo object named {full_name}")
        return idx

    def _surface(
        self,
        ref: SurfaceCenterRef | SurfaceSampleRef | SurfaceNormalRef,
    ) -> tuple[Vec3, Vec3]:
        geom_id = self._named_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            ref.entity_id,
            ref.collider_name,
        )

        side = ref.side
        local = _AXIS[side]
        xmat = self.data.geom_xmat[geom_id]

        normal = self.__rot_local_to_world(xmat, local)

        axis_idx = max(range(3), key=lambda idx: abs(local[idx]))
        geom_pos = self.data.geom_xpos[geom_id]
        half_extent = float(self.model.geom_size[geom_id][axis_idx])

        center = (
            float(geom_pos[0]) + normal[0] * half_extent,
            float(geom_pos[1]) + normal[1] * half_extent,
            float(geom_pos[2]) + normal[2] * half_extent,
        )

        if isinstance(ref, SurfaceSampleRef):
            margin = self.__require_concrete_float(ref.margin)
            size = self.model.geom_size[geom_id]
            tangent_axes = [idx for idx in range(3) if idx != axis_idx]
            local_offset = [0.0, 0.0, 0.0]

            for axis_order, idx in enumerate(tangent_axes):
                explicit = ref.x if axis_order == 0 else ref.y
                span = max(float(size[idx]) - margin, 0.0)

                if explicit is None:
                    local_offset[idx] = random.uniform(-span, span)
                else:
                    value = self.__require_concrete_float(explicit)
                    local_offset[idx] = max(-span, min(span, value))

            offset = self.__require_concrete_vec3(ref.offset)

            if ref.offset_frame == "target":
                local_offset[0] += float(offset[0])
                local_offset[1] += float(offset[1])
                local_offset[2] += float(offset[2])

                world_offset = self.__rot_local_to_world(
                    xmat,
                    (
                        float(local_offset[0]),
                        float(local_offset[1]),
                        float(local_offset[2]),
                    ),
                )
            elif ref.offset_frame == "world":
                world_sample_offset = self.__rot_local_to_world(
                    xmat,
                    (
                        float(local_offset[0]),
                        float(local_offset[1]),
                        float(local_offset[2]),
                    ),
                )
                world_offset = (
                    world_sample_offset[0] + float(offset[0]),
                    world_sample_offset[1] + float(offset[1]),
                    world_sample_offset[2] + float(offset[2]),
                )
            else:
                raise SimulacBaseError(f"Unsupported offset frame: {ref.offset_frame}")

            center = (
                center[0] + world_offset[0],
                center[1] + world_offset[1],
                center[2] + world_offset[2],
            )

        return center, normal

    def _support(self, ref: SupportPointRef) -> Vec3:
        geom_id = self._named_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            ref.entity_id,
            ref.collider_name,
        )
        direction = self.__require_concrete_vec3(ref.direction)

        xmat = self.data.geom_xmat[geom_id]
        if ref.frame == "local":
            direction = self.__rot_local_to_world(xmat, direction)

        norm = max(
            sqrt(
                direction[0] * direction[0]
                + direction[1] * direction[1]
                + direction[2] * direction[2]
            ),
            1e-9,
        )

        direction = [
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
        ]

        radius = max(float(size) for size in self.model.geom_size[geom_id])
        geom_pos = self.data.geom_xpos[geom_id]

        return (
            float(geom_pos[0]) + direction[0] * radius,
            float(geom_pos[1]) + direction[1] * radius,
            float(geom_pos[2]) + direction[2] * radius,
        )

    def __joint_dims(self, joint_type: int) -> tuple[int, int]:
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        return 1, 1

    type Matrix = tuple[float, float, float, float, float, float, float, float, float]
    type Vec3 = tuple[float, float, float]

    def __rot_local_to_world(self, xmat: Matrix, direction: Vec3) -> Vec3:
        """change `local frame` to `world frame`
        world_direction = mat @ direction
        world_direction = [
            R00 * x + R01 * y + R02 * z,
            R10 * x + R11 * y + R12 * z,
            R20 * x + R21 * y + R22 * z,
        ]
        """
        world_direction = (
            float(xmat[0]) * direction[0]
            + float(xmat[1]) * direction[1]
            + float(xmat[2]) * direction[2],
            float(xmat[3]) * direction[0]
            + float(xmat[4]) * direction[1]
            + float(xmat[5]) * direction[2],
            float(xmat[6]) * direction[0]
            + float(xmat[7]) * direction[1]
            + float(xmat[8]) * direction[2],
        )
        return world_direction

    def __require_concrete_float(self, value: Randomizable[float]) -> float:
        if not isinstance(value, float):
            raise SimulacBaseError("RandomizableFloat must be sampled before resolve")
        return value

    def __require_concrete_vec3(
        self, value: Randomizable[tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        if isinstance(value, tuple) and len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))

        if isinstance(value, list) and len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))

        raise SimulacBaseError("RandomizableVec3 must be sampled before resolve")

    def __mat_to_quat_xyzw(
        self, xmat: tuple[float, float, float, float, float, float, float, float, float]
    ) -> Quat:
        # TODO: @gangjeuk
        # Refactoring this function
        # Dup of mujoco_adapter::_mat_to_quat_xyzw_from_values
        m00, m01, m02 = float(xmat[0]), float(xmat[1]), float(xmat[2])
        m10, m11, m12 = float(xmat[3]), float(xmat[4]), float(xmat[5])
        m20, m21, m22 = float(xmat[6]), float(xmat[7]), float(xmat[8])

        trace = m00 + m11 + m22

        if trace > 0.0:
            s = (trace + 1.0) ** 0.5 * 2.0
            qw = 0.25 * s
            qx = (m21 - m12) / s
            qy = (m02 - m20) / s
            qz = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s

        norm = max((qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5, 1e-9)

        return (
            qx / norm,
            qy / norm,
            qz / norm,
            qw / norm,
        )


class MujocoPlacementResolver:
    def __init__(
        self,
        data: mujoco.MjData,
        resolver: MujocoRefResolver,
        *,
        stuff_bindings: dict[str, MujocoStuffBinding],
        machine_bindings: dict[str, MujocoRobotBinding],
        camera_bindings: dict[str, MujocoCameraBinding],
        light_bindings: dict[str, MujocoLightBinding],
    ) -> None:
        self.data = data
        self.resolver = resolver
        self._stuff_bindings = stuff_bindings
        self._machine_bindings = machine_bindings
        self._camera_bindings = camera_bindings
        self._light_bindings = light_bindings

    def resolve_entity_pos(
        self,
        entity_id: str,
        pos: Any,
    ) -> Vec3:

        if isinstance(pos, SurfaceSampleRef):
            return self.resolve_surface_sample_pos(
                entity_id=entity_id,
                ref=pos,
            )

        if isinstance(pos, PointRefBase):
            return self.__vec3(self.resolver.resolve_point(pos))

        return self.__vec3(pos)

    def resolve_surface_sample_pos(
        self,
        *,
        entity_id: str,
        ref: SurfaceSampleRef,
    ) -> Vec3:
        target = self.__vec3(self.resolver.resolve_point(ref))

        if ref.using is None:
            return target

        source_ref = self.__source_ref(entity_id, ref.using)
        source = self.__vec3(self.resolver.resolve_point(source_ref))
        root = self.__root_pos(entity_id)

        return (
            root[0] + target[0] - source[0],
            root[1] + target[1] - source[1],
            root[2] + target[2] - source[2],
        )

    def __source_ref(
        self,
        entity_id: str,
        using: str | PointRefType,
    ) -> PointRefType:
        if isinstance(using, str):
            return AnchorPosRef(entity_id, using)

        if isinstance(using, PointRefBase):  # pyright: ignore[reportUnnecessaryIsInstance]
            return using

        raise SimulacBaseError(f"Invalid surface sample using ref: {using!r}")

    def __root_pos(self, entity_id: str) -> Vec3:
        binding = self.__binding(entity_id)

        root_body_id = binding.root_body_id
        if root_body_id is None:
            raise SimulacBaseError(
                f"Entity {entity_id!r} does not have a root body for placement"
            )

        pos = self.data.xpos[root_body_id]
        return (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        )

    def __binding(
        self,
        entity_id: str,
    ) -> (
        MujocoStuffBinding
        | MujocoRobotBinding
        | MujocoCameraBinding
        | MujocoLightBinding
    ):
        binding = self._stuff_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self._machine_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self._camera_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self._light_bindings.get(entity_id)
        if binding is not None:
            return binding

        raise SimulacBaseError(f"No MuJoCo binding for entity {entity_id!r}")

    def __vec3(self, value: Any) -> Vec3:
        try:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]),
            )
        except (TypeError, IndexError, ValueError) as exc:
            raise SimulacBaseError(
                f"Expected Vec3-compatible value: {value!r}"
            ) from exc
