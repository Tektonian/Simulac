from __future__ import annotations

import random
from abc import ABCMeta
from dataclasses import dataclass, field
from math import sqrt
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    MutableMapping,
    NotRequired,
    TypedDict,
)

import mujoco
import mujoco.viewer

from simulac.base.error.error import SimulacBaseError
from simulac.base.types.geometry import Quat, Vec3
from simulac.base.utils.rotation import euler_to_quat
from simulac.sdk.environment_service.common.model.ref import (
    AnchorPosRef,
    AnchorRef,
    AttachOp,
    BuildOpBase,
    CameraRef,
    ColliderCenterRef,
    ColliderRef,
    FollowOp,
    JointAxisRef,
    JointRef,
    LookAtOp,
    PlaceOp,
    PointRefBase,
    RefBase,
    SetColliderFrictionOp,
    SetJointDampingOp,
    SetJointFrictionOp,
    SetJointPosOp,
    SupportPointRef,
    SurfaceCenterRef,
    SurfaceNormalRef,
    SurfaceSampleRef,
    WorldPointRef,
)
from simulac.sdk.environment_service.common.randomize import (
    BboxConstraintSpec,
    ChoiceRandomSpec,
    DistanceConstraintSpec,
    EntryRandomSpec,
    NonpenetrationConstraintSpec,
    NormalRandomSpec,
    UniformRandomSpec,
)
from simulac.sdk.runner_service.common.model.runtime import (
    CameraRuntime,
    RobotRuntime,
    StuffRuntime,
)
from simulac.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterState,
)
from simulac.sdk.runner_service.common.runner import IRunner, IRunnerFactory
from simulac.sdk.runner_service.common.runner_service import IRunnerManagementService
from simulac.sdk.runner_service.common.sampler import ResetSampler
from simulac.sdk.runner_service.local.mujoco.binding import (
    MujocoActuatorBinding,
    MujocoCameraBinding,
    MujocoJointBinding,
    MujocoLinkBinding,
    MujocoRobotBinding,
    MujocoStuffBinding,
)
from simulac.sdk.runner_service.local.mujoco.runtime import (
    MujocoCameraRuntimeOps,
    MujocoRobotRuntimeOps,
    MujocoStuffRuntimeOps,
)

from .mujoco.resolver import MujocoRefResolver
from .mujoco.constraint import MujocoConstraintEvaluator

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.environment import IEnvironment
    from simulac.sdk.environment_service.common.environment_service import (
        IEnvironmentManagementService,
    )
    from simulac.sdk.environment_service.common.model.entity import (
        EnvironmentCameraEntity,
        EnvironmentMachineEntity,
        EnvironmentStuffEntity,
    )
    from simulac.sdk.environment_service.common.randomize import (
        RandomizableVec3,
        RandomSpec,
    )
    from simulac.sdk.log_service.common.log_service import ILogService


MUJOCO_SCENE = """
<mujoco model="scene">
  <statistic center="0.3 0 0.4" extent="1"/>

  <option timestep="0.005" iterations="5" ls_iterations="8" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <custom>
    <numeric data="12" name="max_contact_points"/>
  </custom>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
    <scale contactwidth="0.075" contactheight="0.025" forcewidth="0.05" com="0.05" framewidth="0.01" framelength="0.2"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1"/>
  </worldbody>
</mujoco>
"""

type _ConstraintSpec = (
    BboxConstraintSpec | DistanceConstraintSpec | NonpenetrationConstraintSpec
)

type _SampledPoint = Vec3 | RefBase
type _SampledRot = Vec3 | RefBase
type _SampledFloat = float
type _SampledSize = Vec3


class _CandidateConstraints(TypedDict, total=False):
    pos: list[_ConstraintSpec]
    rot: list[_ConstraintSpec]


class _EntityCandidate(TypedDict):
    pos: _SampledPoint
    rot: _SampledRot
    constraints: _CandidateConstraints

    mass: NotRequired[_SampledFloat]
    density: NotRequired[_SampledFloat]
    friction: NotRequired[_SampledFloat]
    size: NotRequired[_SampledSize]


type _ResetCandidate = dict[str, _EntityCandidate]


def _subtree_body_ids(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    body_ids: list[int] = []
    for bid in range(model.nbody):
        cur = bid
        while cur != 0:
            if cur == root_body_id:
                body_ids.append(bid)
                break
            cur = int(model.body_parentid[cur])
    return body_ids


class MujocoRunner(IRunner):
    def __init__(
        self,
        runner_id: str,
        env: IEnvironment,
        mj_model: mujoco.MjModel,
        stuff_entities: dict[str, EnvironmentStuffEntity],
        machine_entities: dict[str, EnvironmentMachineEntity],
        camera_entities: dict[str, EnvironmentCameraEntity],
        stuff_bindings: dict[str, MujocoStuffBinding],
        camera_bindins: dict[str, MujocoCameraBinding],
        machine_bindings: dict[str, MujocoRobotBinding],
        on_after_call_step: Callable[[str], None],
    ) -> None:
        self.runner_type = "mujoco"
        self.runner_id = runner_id
        self.env = env
        self.mj_model = mj_model
        self._stuff_entities = stuff_entities
        self._machine_entities = machine_entities
        self._camera_entities = camera_entities
        self._stuff_bindings = stuff_bindings
        self._machine_bindings = machine_bindings
        self._camera_bindings = camera_bindins
        self._runtimes = dict[str, StuffRuntime | RobotRuntime | CameraRuntime]()
        self._follow_ops: list[FollowOp] = []
        self.state = {}
        self.on_after_call_step = on_after_call_step
        self._data: mujoco.MjData | None = None
        self.resolver: MujocoRefResolver | None = None

        self.__MAX_RESET_RETRY = 100
        self.__reset_passed = False

    def initialize(self) -> None:
        self._data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self._data)

    def _require_data(self) -> mujoco.MjData:
        if self._data is None:
            raise SimulacBaseError("Runner must be initialized")
        return self._data

    def step(self, action: list[float]) -> None:
        data = self._require_data()
        # TODO: @gangjeuk
        # seperate action spaces by each `Robot` instance
        if len(action) != self.mj_model.nu:
            raise SimulacBaseError(
                f"Action size mismatch: expected {self.mj_model.nu}, got {len(action)}"
            )
        data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, data)
        self._apply_follow_ops(self.resolver)
        self.on_after_call_step(self.runner_id)

    def tick(self) -> None:
        mujoco.mj_step(self.mj_model, self._require_data())
        self._apply_follow_ops(self.resolver)

    # FIXME: debug purpose for now. Should return state info mapped with self._env
    def get_state(self) -> None:
        for i in range(self.mj_model.nbody):
            print(self._data.body(i))
        breakpoint()

    def get_runtime_object(self, entity_id: str):
        ret = self._runtimes.get(entity_id, None)
        if ret is not None:
            return ret

        raise SimulacBaseError(f"There is no runtime object id '{entity_id}'")

    def snapshot(self): ...
    def _native_snapshot(self, camera_id: str, *, width: int = 640, height: int = 480):
        data = self._require_data()
        binding = self._camera_bindings.get(camera_id)
        if binding is None:
            raise SimulacBaseError(f"No camera named {camera_id!r}")

        entity = self._camera_entities[camera_id]

        # self._apply_follow_ops()

        renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
        try:
            if entity.spec.type == "depth":
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=binding.camera_id)
                return renderer.render().copy()

            if entity.spec.type == "segmentation":
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=binding.camera_id)
                return renderer.render().copy()

            renderer.update_scene(data, camera=binding.camera_id)
            return renderer.render().copy()
        finally:
            renderer.close()

    def set_state(self) -> None: ...
    def clone_state(self) -> None: ...
    def reset(self, seed: int | None = 0) -> None:
        data = self._require_data()
        sampler = ResetSampler(seed)

        self._clean_runtimes()

        retry_count = 0
        while self.__reset_passed or retry_count <= self.__MAX_RESET_RETRY:
            candidate = self._sampling_candidate(sampler)

            mujoco.mj_resetData(self.mj_model, data)

            self._apply_candidate(candidate, sampler)

            mujoco.mj_setConst(self.mj_model, data)
            mujoco.mj_forward(self.mj_model, data)

            if not self._constraints_pass():
                retry_count += 1
                continue

            self._create_runtimes()
            self.__reset_passed = True
            return
        raise SimulacBaseError("Failed to sample valid reset state")

    def _debug_render(self):
        return mujoco.viewer.launch_passive(self.mj_model, self._data)

    def _sampling_candidate(self, sampler: ResetSampler) -> dict[str, dict[str, Any]]:
        candidate: _ResetCandidate = {}
        for eid, entity in self._stuff_entities.items():
            candidate[eid] = {
                "pos": sampler.sample(entity.pos),
                "rot": sampler.sample(entity.rot),
                "constraints": {
                    "pos": sampler.constraints(entity.pos),
                    "rot": sampler.constraints(entity.rot),
                },
            }
            for name in ("mass", "density", "friction", "size"):
                if hasattr(entity, name):
                    value = getattr(entity, name)
                    if value is not None:
                        candidate[eid][name] = sampler.sample(value)
        for eid, entity in self._machine_entities.items():
            candidate[eid] = {
                "pos": sampler.sample(entity.pos),
                "rot": sampler.sample(entity.rot),
                "constraints": {
                    "pos": sampler.constraints(entity.pos),
                    "rot": sampler.constraints(entity.rot),
                },
            }

        for eid, entity in self._camera_entities.items():
            candidate[eid] = {
                "pos": sampler.sample(entity.pos),
                "rot": sampler.sample(entity.rot),
                "constraints": {
                    "pos": sampler.constraints(entity.pos),
                    "rot": sampler.constraints(entity.rot),
                },
            }
        return candidate

    def _clean_runtimes(self) -> None:
        self._runtimes: dict[str, StuffRuntime | RobotRuntime] = dict()

    def _create_runtimes(self) -> None:
        for eid, binding in self._stuff_bindings.items():
            ops = MujocoStuffRuntimeOps(
                eid, self.mj_model, self._require_data(), binding
            )

            stuff_runtime = StuffRuntime(eid, ops)
            self._runtimes[eid] = stuff_runtime
        for eid, binding in self._machine_bindings.items():
            ops = MujocoRobotRuntimeOps(
                eid,
                self.mj_model,
                self._require_data(),
                binding,
                on_after_step=lambda: self.on_after_call_step(self.runner_id),
            )
            robot_runtime = RobotRuntime(eid, ops)
            self._runtimes[eid] = robot_runtime
        for eid, binding in self._camera_bindings.items():
            ops = MujocoCameraRuntimeOps(
                eid, self.mj_model, self._require_data(), binding
            )
            camera_runtime = CameraRuntime(eid, ops)
            self._runtimes[eid] = camera_runtime

    def _apply_candidate(
        self, candidate: dict[str, dict[str, Any]], sampler: ResetSampler
    ) -> None:
        data = self._require_data()
        self.resolver = MujocoRefResolver(
            self.mj_model,
            data,
            stuff_bindings=self._stuff_bindings,
            machine_bindings=self._machine_bindings,
            camera_bindings=self._camera_bindings,
        )

        for eid, values in candidate.items():
            binding = self._entity_binding(eid)
            pos = values.get("pos")
            if isinstance(pos, RefBase):
                pos = self.resolver.resolve_point(sampler.sample(pos))

            rot = values.get("rot")
            quat = None
            if rot is not None and not isinstance(rot, RefBase):
                quat = euler_to_quat(*rot)

            if eid in self._camera_bindings:
                self._apply_camera_pose(self._camera_bindings[eid], pos, quat)
                continue

            self._apply_root_pose(binding, pos, quat)

            friction = values.get("friction")
            if (
                eid in self._stuff_bindings
                and friction is not None
                and not isinstance(friction, RefBase)
            ):
                self._apply_stuff_friction(self._stuff_bindings[eid], friction)

            mass = values.get("mass")
            if (
                eid in self._stuff_bindings
                and mass is not None
                and not isinstance(mass, RefBase)
            ):
                self._apply_stuff_mass(self._stuff_bindings[eid], mass)

        mujoco.mj_forward(self.mj_model, data)
        for eid, values in candidate.items():
            ops = [
                placeop
                for placeop in self.env.relations
                if placeop.entity.entity_id == eid
            ]
            for op in ops:
                self._apply_build_op(eid, op, self.resolver, sampler)
                mujoco.mj_forward(self.mj_model, data)

    def _entity_binding(
        self,
        entity_id: str,
    ) -> MujocoStuffBinding | MujocoRobotBinding | MujocoCameraBinding:
        binding = self._stuff_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self._machine_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self._camera_bindings.get(entity_id)
        if binding is not None:
            return binding
        raise SimulacBaseError(f"No MuJoCo binding for entity {entity_id!r}")

    def _camera_binding(self, entity_id: str) -> MujocoCameraBinding:
        binding = self._camera_bindings.get(entity_id)
        if binding is None:
            raise SimulacBaseError(f"No MuJoCo camera binding for entity {entity_id!r}")
        return binding

    def _apply_attach_op(
        self,
        op: AttachOp,
        resolver: MujocoRefResolver,
        sampler: ResetSampler,
    ) -> None:
        entity_id = op.entity.entity_id

        if entity_id not in self._camera_bindings:
            raise SimulacBaseError("AttachOp currently supports camera entities first")

        binding = self._camera_bindings[entity_id]

        parent_pos, parent_quat = resolver.resolve_frame(op.parent)

        offset: Vec3 = sampler.sample(op.offset)

        if op.offset_frame == "target":
            x, y, z, w = parent_quat
            vx, vy, vz = offset
            # q * v * q^-1
            # https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
            tx = 2.0 * (y * vz - z * vy)
            ty = 2.0 * (z * vx - x * vz)
            tz = 2.0 * (x * vy - y * vx)
            offset_world = (
                vx + w * tx + (y * tz - z * ty),
                vy + w * ty + (z * tx - x * tz),
                vz + w * tz + (x * ty - y * tx),
            )
        elif op.offset_frame == "world":
            offset_world = offset
        else:
            raise SimulacBaseError(f"Unsupported offset frame: {op.offset_frame}")
        rot: Quat = sampler.sample(op.rot)
        local_quat = euler_to_quat(float(rot[0]), float(rot[1]), float(rot[2]))

        # https://github.com/google-deepmind/mujoco/blob/5ee8bd7b9c3147f1094816882903e741e53c26bf/src/engine/engine_util_spatial.c#L66
        ax, ay, az, aw = parent_quat
        bx, by, bz, bw = local_quat
        camera_quat: Quat = (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

        camera_pos: Vec3 = (
            parent_pos[0] + offset_world[0],
            parent_pos[1] + offset_world[1],
            parent_pos[2] + offset_world[2],
        )

        self._apply_camera_pose(binding, camera_pos, camera_quat)

    def _apply_look_at_op(
        self,
        op: LookAtOp,
        resolver: MujocoRefResolver,
        sampler: ResetSampler,
    ) -> None:
        entity_id = op.entity.entity_id

        if entity_id not in self._camera_bindings:
            raise SimulacBaseError("LookAtOp currently supports camera entities")

        binding = self._camera_bindings[entity_id]

        target = resolver.resolve_point(sampler.sample(op.target))
        offset: Vec3 = sampler.sample(op.offset)

        current_pos = self._require_data().xpos[binding.root_body_id]
        camera_pos = (
            float(current_pos[0]) + float(offset[0]),
            float(current_pos[1]) + float(offset[1]),
            float(current_pos[2]) + float(offset[2]),
        )

        quat = self._look_at_quat(
            eye=camera_pos,
            target=(float(target[0]), float(target[1]), float(target[2])),
            up=sampler.sample(op.up),
        )

        self._apply_camera_pose(binding, camera_pos, quat)

    def _apply_follow_op(self, op: FollowOp, resolver: MujocoRefResolver) -> None:
        entity_id = op.entity.entity_id

        if entity_id not in self._camera_bindings:
            raise SimulacBaseError("FollowOp currently supports camera entities")

        binding = self._camera_bindings[entity_id]

        target_pos, target_quat = resolver.resolve_frame(op.target)

        offset = op.offset
        offset_vec = (
            float(offset[0]),
            float(offset[1]),
            float(offset[2]),
        )

        if op.frame == "local":
            x, y, z, w = target_quat
            vx, vy, vz = offset_vec
            # q * v * q^-1
            tx = 2.0 * (y * vz - z * vy)
            ty = 2.0 * (z * vx - x * vz)
            tz = 2.0 * (x * vy - y * vx)
            offset_world = (
                vx + w * tx + (y * tz - z * ty),
                vy + w * ty + (z * tx - x * tz),
                vz + w * tz + (x * ty - y * tx),
            )
        elif op.frame == "world":
            offset_world = offset_vec
        else:
            raise SimulacBaseError(f"Unsupported follow frame: {op.frame}")

        camera_pos = (
            target_pos[0] + offset_world[0],
            target_pos[1] + offset_world[1],
            target_pos[2] + offset_world[2],
        )

        self._apply_camera_pose(binding, camera_pos, target_quat)

    def _apply_follow_ops(self, resolver: MujocoRefResolver) -> None:
        if not self._follow_ops:
            return
        for op in self._follow_ops:
            self._apply_follow_op(op, resolver)

        mujoco.mj_forward(self.mj_model, self._require_data())

    def _look_at_quat(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float],
    ) -> tuple[float, float, float, float]:
        # https://graphicscompendium.com/opengl/18-lookat-matrix

        # f of Forward.
        # Camera local -Z points forward.
        fx = target[0] - eye[0]
        fy = target[1] - eye[1]
        fz = target[2] - eye[2]

        flen = max((fx * fx + fy * fy + fz * fz) ** 0.5, 1e-9)
        fx, fy, fz = fx / flen, fy / flen, fz / flen

        ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
        ulen = max((ux * ux + uy * uy + uz * uz) ** 0.5, 1e-9)
        ux, uy, uz = ux / ulen, uy / ulen, uz / ulen

        # r of Right
        # right = forward x up
        rx = fy * uz - fz * uy
        ry = fz * ux - fx * uz
        rz = fx * uy - fy * ux

        rlen = max((rx * rx + ry * ry + rz * rz) ** 0.5, 1e-9)
        rx, ry, rz = rx / rlen, ry / rlen, rz / rlen

        # u of Up
        # corrected up = right x forward
        ux = ry * fz - rz * fy
        uy = rz * fx - rx * fz
        uz = rx * fy - ry * fx

        # columns: local X=right, local Y=up, local Z=-forward
        xmat = [
            rx,
            ux,
            -fx,
            ry,
            uy,
            -fy,
            rz,
            uz,
            -fz,
        ]
        return self._mat_to_quat_xyzw_from_values(xmat)

    def _mat_to_quat_xyzw_from_values(
        self,
        xmat: list[float] | tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        # https://github.com/google-deepmind/mujoco/blob/5ee8bd7b9c3147f1094816882903e741e53c26bf/src/engine/engine_util_spatial.c#L187
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

    def _apply_camera_pose(
        self,
        binding: MujocoCameraBinding,
        pos: tuple[float, float, float] | None,
        quat: tuple[float, float, float, float] | None,
    ) -> None:
        if pos is not None:
            self.mj_model.body_pos[binding.root_body_id] = (
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
            )

        if quat is not None:
            self.mj_model.body_quat[binding.root_body_id] = (
                float(quat[3]),
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
            )

    def _apply_root_pose(
        self,
        binding: MujocoStuffBinding | MujocoRobotBinding,
        pos: Any,
        quat: tuple[float, float, float, float] | None,
    ) -> None:
        if pos is not None:
            root_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            if binding.root_freejoint_id >= 0:
                qadr = int(self.mj_model.jnt_qposadr[binding.root_freejoint_id])
                self._require_data().qpos[qadr : qadr + 3] = root_pos
            else:
                self.mj_model.body_pos[binding.root_body_id] = root_pos

        if quat is not None:
            quat_wxyz = [
                float(quat[3]),
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
            ]
            if binding.root_freejoint_id >= 0:
                qadr = int(self.mj_model.jnt_qposadr[binding.root_freejoint_id])
                self._require_data().qpos[qadr + 3 : qadr + 7] = quat_wxyz
            else:
                self.mj_model.body_quat[binding.root_body_id] = quat_wxyz

    def _apply_stuff_friction(
        self,
        binding: MujocoStuffBinding,
        friction: float,
    ) -> None:
        if friction < 0:
            raise SimulacBaseError("friction must be non-negative")

        for geom_id in binding.geom_ids:
            self.mj_model.geom_friction[geom_id][0] = float(friction)

    def _apply_stuff_mass(
        self,
        binding: MujocoStuffBinding,
        mass: float,
    ) -> None:
        if mass <= 0:
            raise SimulacBaseError("mass must be positive")

        current_mass = sum(
            float(self.mj_model.body_mass[body_id]) for body_id in binding.body_ids
        )
        if current_mass <= 0:
            raise SimulacBaseError("current mass must be positive")

        ratio = float(mass) / current_mass
        for body_id in binding.body_ids:
            self.mj_model.body_mass[body_id] = (
                float(self.mj_model.body_mass[body_id]) * ratio
            )
            self.mj_model.body_inertia[body_id] = [
                float(self.mj_model.body_inertia[body_id][0]) * ratio,
                float(self.mj_model.body_inertia[body_id][1]) * ratio,
                float(self.mj_model.body_inertia[body_id][2]) * ratio,
            ]

    def _apply_build_op(
        self,
        eid: str,
        op: BuildOpBase,
        resolver: MujocoRefResolver,
        sampler: ResetSampler,
    ) -> None:
        data = self._require_data()
        if isinstance(op, PlaceOp):
            entity_id = op.entity.entity_id
            binding = self._entity_binding(entity_id)

            target_ref = sampler.sample(op.target)
            target_point = resolver.resolve_point(target_ref)

            if op.source is None:
                source_pos = data.xpos[binding.root_body_id]
                source_point = [
                    float(source_pos[0]),
                    float(source_pos[1]),
                    float(source_pos[2]),
                ]
            else:
                source_ref = sampler.sample(op.source)
                source_point = resolver.resolve_point(source_ref)

            delta = [
                float(target_point[0]) - float(source_point[0]),
                float(target_point[1]) - float(source_point[1]),
                float(target_point[2]) - float(source_point[2]),
            ]

            if binding.root_freejoint_id >= 0:
                qadr = int(self.mj_model.jnt_qposadr[binding.root_freejoint_id])
                data.qpos[qadr : qadr + 3] = [
                    float(data.qpos[qadr]) + delta[0],
                    float(data.qpos[qadr + 1]) + delta[1],
                    float(data.qpos[qadr + 2]) + delta[2],
                ]
            else:
                root_pos = self.mj_model.body_pos[binding.root_body_id]
                self.mj_model.body_pos[binding.root_body_id] = [
                    float(root_pos[0]) + delta[0],
                    float(root_pos[1]) + delta[1],
                    float(root_pos[2]) + delta[2],
                ]
            return

        if isinstance(op, SetColliderFrictionOp):
            target = op.target
            friction = float(sampler.sample(op.friction))

            if friction < 0:
                raise SimulacBaseError("fiction must be non-negative")

            geom_id = self._named_id(
                mujoco.mjtObj.mjOBJ_GEOM, target.entity_id, target.name
            )

            self.mj_model.geom_friction[geom_id][0] = friction
            return

        if isinstance(op, SetJointPosOp):
            target = op.target
            pos = float(sampler.sample(op.pos))

            joint_id = self._named_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                target.entity_id,
                target.name,
            )

            joint_type = int(self.mj_model.jnt_type[joint_id])
            if joint_type not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise SimulacBaseError(
                    "SetJointPosOp only supports hinge and slide joints"
                )

            qadr = int(self.mj_model.jnt_qposadr[joint_id])
            data.qpos[qadr] = pos
            return

        if isinstance(op, SetJointFrictionOp):
            target = op.target
            friction = float(sampler.sample(op.friction))

            if friction < 0:
                raise SimulacBaseError("joint friction must be non-negative")

            joint_id = self._named_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                target.entity_id,
                target.name,
            )

            joint_type = int(self.mj_model.jnt_type[joint_id])
            _, qvel_dim = (1, 1)
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                qvel_dim = 6

            if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                qvel_dim = 3

            dadr = int(self.mj_model.jnt_dofadr[joint_id])
            self.mj_model.dof_frictionloss[dadr : dadr + qvel_dim] = [
                friction
            ] * qvel_dim
            return

        if isinstance(op, SetJointDampingOp):
            target = op.target
            damping = float(sampler.sample(op.damping))

            if damping < 0:
                raise SimulacBaseError("joint damping must be non-negative")

            joint_id = self._named_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                target.entity_id,
                target.name,
            )

            joint_type = int(self.mj_model.jnt_type[joint_id])
            _, qvel_dim = (1, 1)
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                qvel_dim = 6

            if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                qvel_dim = 3

            dadr = int(self.mj_model.jnt_dofadr[joint_id])
            self.mj_model.dof_damping[dadr : dadr + qvel_dim] = [damping] * qvel_dim
            return

        if isinstance(op, AttachOp):
            self._apply_attach_op(op, resolver, sampler)
            return

        if isinstance(op, LookAtOp):
            self._apply_look_at_op(op, resolver, sampler)
            return

        if isinstance(op, FollowOp):
            self._follow_ops.append(op)
            self._apply_follow_op(op, resolver=resolver)
            return

        raise SimulacBaseError(f"Unsupported build op: {type(op).__name__}")

    def _named_id(
        self,
        obj_type: mujoco.mjtObj,
        entity_id: str,
        name: str,
    ) -> int:
        full_name = f"{entity_id}/{name}"
        obj_id = mujoco.mj_name2id(self.mj_model, obj_type, full_name)

        if obj_id < 0:
            raise SimulacBaseError(f"No MuJoCo object named {full_name!r}")

        return obj_id

    def _constraints_pass(self) -> bool:
        evaluator = MujocoConstraintEvaluator(
            model=self.mj_model,
            data=self._require_data(),
            resolver=self.resolver,
            bindings={
                **self._stuff_bindings,
                **self._machine_bindings,
                **self._camera_bindings,
            },
        )
        return evaluator.passed(self.env.constraints)


class MujocoAdapter(IPhysicsEngineAdapter):
    """_summary_
        ![test](https://picsum.photos/200/300)

    Args:
        PhysicsEngineAdapter (_type_): _description_
    """

    def __init__(
        self,
        env_id: str,
        LogService: ILogService,
        RunnerManagementService: IRunnerManagementService,
        EnvironmentManagementService: IEnvironmentManagementService,
    ) -> None:

        self.env_id = env_id
        self.LogService = LogService
        self.RunnerManagementService = RunnerManagementService
        self.EnvironmentManagementService = EnvironmentManagementService

        self._runner_count = 0
        self._step_count = 0
        self._step_count_map: MutableMapping[str, int] = dict()

        self.root_spec = mujoco.MjSpec.from_string(MUJOCO_SCENE)
        self.root_frame = self.root_spec.worldbody.add_frame()

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._stuff_bindings: dict[str, MujocoStuffBinding] = {}
        self._machine_bindings: dict[str, MujocoRobotBinding] = {}
        self._camera_bindings: dict[str, MujocoCameraBinding] = {}

        env_ret = self.EnvironmentManagementService.get_environment(self.env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]
        self.env = env

        self._stuff_entities = {e.id: e for e in env.stuffs if e.id is not None}
        self._machine_entities = {e.id: e for e in env.machines if e.id is not None}
        self._camera_entities = {e.id: e for e in env.cameras if e.id is not None}
        for stuff in env.stuffs:
            self.__add_stuff(stuff)
        for machine in env.machines:
            self.__add_machine(machine)
        for camera in env.cameras:
            self.__add_camera(camera)
        """
        TODO: @gangjeuk
            1. implement mujoco parallel
            2. [o] - implement robos initialization code (27/04/2026)
            
        """
        self.model = self.root_spec.compile()
        for stuff in env.stuffs:
            self._stuff_bindings[stuff.id] = self.__build_stuff_binding(
                stuff.id,
            )
        for machine in env.machines:
            self._machine_bindings[machine.id] = self.__build_machine_binding(
                machine.id,
            )
        for camera in env.cameras:
            self._camera_bindings[camera.id] = self.__build_camera_binding(camera.id)

    def create_runner(self) -> IRunner:
        if self._step_count != 0:
            raise SimulacBaseError(
                "Cannot create new runner after calling step() function"
            )

        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        def on_after_runner_step(runner_id: str):
            self._step_count_map[runner_id] += 1
            self._step_count += 1

        new_runner_id = f"run_{self._runner_count}"

        runner = MujocoRunner(
            new_runner_id,
            self.env,
            mj_model=self.model,
            stuff_entities=self._stuff_entities,
            camera_entities=self._camera_entities,
            machine_entities=self._machine_entities,
            stuff_bindings=self._stuff_bindings,
            machine_bindings=self._machine_bindings,
            camera_bindins=self._camera_bindings,
            on_after_call_step=on_after_runner_step,
        )

        self.LogService.debug(
            f"new mujoco runner created env_id: {self.env_id} runner_id: {new_runner_id}"
        )
        self._step_count_map[new_runner_id] = 0
        self._runner_count += 1

        return runner

    def get_state(self) -> IPhysicsEngineAdapterState:
        return IPhysicsEngineAdapterState(
            self.env_id, self._runner_count, self._step_count_map
        )

    def __prepare_child_root(
        self, child: mujoco.MjSpec, entity_id: str, add_freejoint: bool
    ):
        bodies: list[mujoco.MjsBody] = child.bodies
        roots = [body for body in bodies if body.parent == child.worldbody]
        if len(roots) != 1:
            raise SimulacBaseError(
                f"MJCF asset for {entity_id!r} must have one root body"
            )

        root = roots[0]
        root.name = "__root__"

        if add_freejoint:
            root.add_freejoint(name="root_freejoint")

    def __add_stuff(self, stuff: EnvironmentStuffEntity):
        # TODO: URDF file is not handled here. Need handling code
        child = mujoco.MjSpec.from_file(stuff.asset_uri)
        self.__prepare_child_root(child, stuff.id, add_freejoint=False)

        self.root_spec.attach(
            child, frame=self.root_frame, prefix=f"{stuff.id}/", suffix=""
        )

    def __add_machine(self, machine: EnvironmentMachineEntity):
        child = mujoco.MjSpec.from_file(machine.asset_uri)
        # TODO: @gangjeuk
        # handle machine.pos, machine.quat
        self.__prepare_child_root(child, machine.id, add_freejoint=False)
        self.root_spec.attach(
            child, frame=self.root_frame, prefix=f"{machine.id}/", suffix=""
        )

    def __add_camera(self, camera: EnvironmentCameraEntity) -> None:
        """add camera as mujoco.mjSpec

        If works like adding entity below
        ```xml
        <body name="front_rgb/__root__" pos="0 0 0">
            <camera name="front_rgb/__root__camera" pos="0 0 0" quat="1 0 0 0" fovy="45"/>
        </body>
        ```
        """
        if camera.id is None:
            raise SimulacBaseError("Camera entity id is required")

        body = self.root_spec.worldbody.add_body(
            name=f"{camera.id}/__root__",
            pos=(0.0, 0.0, 0.0),
        )

        body.add_camera(
            name=f"{camera.id}/__root__camera",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            fovy=float(camera.spec.fov),
        )

    def __actuator_target(
        self,
        actuator_id: int,
    ) -> tuple[
        Literal["joint", "tendon", "site", "body", "unknown"], int | None, str | None
    ]:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        model = self.model
        trn_type = int(model.actuator_trntype[actuator_id])
        target_id = int(model.actuator_trnid[actuator_id][0])

        if target_id < 0:
            return "unknown", None, None

        if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
            return (
                "joint",
                target_id,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, target_id),
            )

        if trn_type == mujoco.mjtTrn.mjTRN_TENDON:
            return (
                "tendon",
                target_id,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, target_id),
            )

        if trn_type == mujoco.mjtTrn.mjTRN_SITE:
            return (
                "site",
                target_id,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, target_id),
            )

        if trn_type == mujoco.mjtTrn.mjTRN_BODY:
            return (
                "body",
                target_id,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, target_id),
            )

        return "unknown", target_id, None

    def __build_machine_binding(self, entity_id: str) -> MujocoRobotBinding:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        model = self.model
        root_name = f"{entity_id}/__root__"
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)

        if root_body_id < 0:
            raise SimulacBaseError(f"No MuJoCo root body for machine {entity_id!r}")

        body_ids = _subtree_body_ids(model, root_body_id)
        geom_ids = [
            gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in body_ids
        ]
        joint_ids = [
            jid for jid in range(model.njnt) if int(model.jnt_bodyid[jid]) in body_ids
        ]
        actuator_ids: list[int] = []
        for aid in range(model.nu):
            target_type, target_id, _ = self.__actuator_target(aid)
            if target_type == "joint" and target_id in joint_ids:
                actuator_ids.append(aid)
            elif target_type == "body" and target_id in body_ids:
                actuator_ids.append(aid)
            elif target_type in {"site", "tendon"}:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                if name and name.startswith(f"{entity_id}/"):
                    actuator_ids.append(aid)

        root_freejoint_id = -1

        for jid in joint_ids:
            if (
                int(model.jnt_bodyid[jid]) == root_body_id
                and int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_FREE
            ):
                root_freejoint_id = jid
                break

        links: dict[str, MujocoLinkBinding] = {}
        for body_id in body_ids:
            body_full_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"body_{body_id}"
            )
            body_name = (
                body_full_name.split("/", 1)[1]
                if body_full_name.startswith(f"{entity_id}/")
                else body_full_name
            )
            body_geom_ids = [
                gid for gid in geom_ids if int(model.geom_bodyid[gid]) == body_id
            ]
            body_joint_ids = [
                jid for jid in joint_ids if int(model.jnt_bodyid[jid]) == body_id
            ]
            child_body_ids = [
                bid for bid in body_ids if int(model.body_parentid[bid]) == body_id
            ]

            links[body_name] = MujocoLinkBinding(
                entity_id=entity_id,
                full_name=body_full_name,
                name=body_name,
                body_id=body_id,
                parent_body_id=int(model.body_parentid[body_id]),
                child_body_ids=child_body_ids,
                geom_ids=body_geom_ids,
                joint_ids=body_joint_ids,
                mocap_id=int(model.body_mocapid[body_id]),
            )

        joints: dict[str, MujocoJointBinding] = {}
        for joint_id in joint_ids:
            joint_full_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                or f"joint_{joint_id}"
            )
            joint_name = (
                joint_full_name.split("/", 1)[1]
                if joint_full_name.startswith(f"{entity_id}/")
                else joint_full_name
            )

            joint_type = int(model.jnt_type[joint_id])
            qpos_dim, qvel_dim = (1, 1)
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                qpos_dim, qvel_dim = (7, 6)
            if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                qpos_dim, qvel_dim = (4, 3)

            limited = bool(model.jnt_limited[joint_id])
            joint_range: tuple[float, float] | None = None
            if limited:
                joint_range = (
                    float(model.jnt_range[joint_id][0]),
                    float(model.jnt_range[joint_id][1]),
                )

            joints[joint_name] = MujocoJointBinding(
                entity_id=entity_id,
                full_name=joint_full_name,
                name=joint_name,
                joint_id=joint_id,
                body_id=int(model.jnt_bodyid[joint_id]),
                joint_type=joint_type,
                qpos_addr=int(model.jnt_qposadr[joint_id]),
                qvel_addr=int(model.jnt_dofadr[joint_id]),
                qpos_dim=qpos_dim,
                qvel_dim=qvel_dim,
                axis=(
                    float(model.jnt_axis[joint_id][0]),
                    float(model.jnt_axis[joint_id][1]),
                    float(model.jnt_axis[joint_id][2]),
                ),
                range=joint_range,
                limited=limited,
            )

        actuators: dict[str, MujocoActuatorBinding] = {}
        for actuator_id in actuator_ids:
            actuator_full_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                or f"actuator_{actuator_id}"
            )
            actuator_name = (
                actuator_full_name.split("/", 1)[1]
                if actuator_full_name.startswith(f"{entity_id}/")
                else actuator_full_name
            )

            target_type, target_id, target_name = self.__actuator_target(actuator_id)

            ctrl_range = None
            if bool(model.actuator_ctrllimited[actuator_id]):
                ctrl_range = (
                    float(model.actuator_ctrlrange[actuator_id][0]),
                    float(model.actuator_ctrlrange[actuator_id][1]),
                )

            force_range = None
            if bool(model.actuator_forcelimited[actuator_id]):
                force_range = (
                    float(model.actuator_forcerange[actuator_id][0]),
                    float(model.actuator_forcerange[actuator_id][1]),
                )

            act_range = None
            if bool(model.actuator_actlimited[actuator_id]):
                act_range = (
                    float(model.actuator_actrange[actuator_id][0]),
                    float(model.actuator_actrange[actuator_id][1]),
                )

            actuators[actuator_name] = MujocoActuatorBinding(
                entity_id=entity_id,
                name=actuator_name,
                full_name=actuator_full_name,
                actuator_id=actuator_id,
                target_type=target_type,
                target_id=target_id,
                target_name=target_name,
                ctrl_range=ctrl_range,
                force_range=force_range,
                act_range=act_range,
                group=int(model.actuator_group[actuator_id]),
            )

            if target_type == "joint" and target_id is not None:
                for joint_binding in joints.values():
                    if joint_binding.joint_id == target_id:
                        joint_binding.actuator_ids.append(actuator_id)
                        break
        return MujocoRobotBinding(
            entity_id=entity_id,
            name=entity_id,
            full_name=entity_id,
            root_body_id=root_body_id,
            root_body_name="__root__",
            root_body_full_name=root_name,
            body_ids=body_ids,
            geom_ids=geom_ids,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
            links=links,
            joints=joints,
            actuators=actuators,
            root_freejoint_id=root_freejoint_id,
            mocap_id=int(model.body_mocapid[root_body_id]),
        )

    def __build_stuff_binding(
        self,
        entity_id: str,
    ) -> MujocoStuffBinding:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        model = self.model

        root_name = f"{entity_id}/__root__"
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)

        if root_body_id < 0:
            raise SimulacBaseError(f"No MuJoCo root body for entity {entity_id!r}")

        body_ids = _subtree_body_ids(model, root_body_id)
        geom_ids = [
            gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in body_ids
        ]
        joint_ids = [
            jid for jid in range(model.njnt) if int(model.jnt_bodyid[jid]) in body_ids
        ]
        actuator_ids: list[int] = []
        for aid in range(model.nu):
            target_type, target_id, _ = self.__actuator_target(aid)
            if target_type == "joint" and target_id in joint_ids:
                actuator_ids.append(aid)
            elif target_type == "body" and target_id in body_ids:
                actuator_ids.append(aid)
            elif target_type in {"site", "tendon"}:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                if name and name.startswith(f"{entity_id}/"):
                    actuator_ids.append(aid)

        root_freejoint_id = -1
        for jid in joint_ids:
            if (
                int(model.jnt_bodyid[jid]) == root_body_id
                and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
            ):
                root_freejoint_id = jid
                break

        return MujocoStuffBinding(
            entity_id=entity_id,
            root_body_id=root_body_id,
            body_ids=body_ids,
            geom_ids=geom_ids,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
            root_freejoint_id=root_freejoint_id,
            mocap_id=int(self.model.body_mocapid[root_body_id]),
        )

    def __build_camera_binding(self, entity_id: str) -> MujocoCameraBinding:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        root_body_full_name = f"{entity_id}/__root__"
        camera_full_name = f"{entity_id}/__root__camera"

        root_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            root_body_full_name,
        )
        if root_body_id < 0:
            raise SimulacBaseError(f"No MuJoCo camera root body for {entity_id!r}")

        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_full_name,
        )
        if camera_id < 0:
            raise SimulacBaseError(f"No MuJoCo camera for {entity_id!r}")

        return MujocoCameraBinding(
            entity_id=entity_id,
            root_body_id=root_body_id,
            root_body_name="__root__",
            root_body_full_name=root_body_full_name,
            camera_id=camera_id,
            camera_name="__root__camera",
            camera_full_name=camera_full_name,
            mocap_id=int(self.model.body_mocapid[root_body_id]),
        )
