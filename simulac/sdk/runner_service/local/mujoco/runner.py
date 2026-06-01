from __future__ import annotations

from array import array
from math import sqrt
from typing import TYPE_CHECKING, Any, Callable, Literal, NotRequired, TypedDict

import mujoco
import mujoco.viewer

from simulac.base.error.error import SimulacBaseError
from simulac.base.types.geometry import Quat, Vec3
from simulac.base.utils.rotation import euler_to_quat
from simulac.sdk.environment_service.common.model.ref import (
    AttachOp,
    BuildOpBase,
    FollowOp,
    LookAtOp,
    PlaceOp,
    PointRefBase,
    RefBase,
    SetColliderFrictionOp,
    SetJointDampingOp,
    SetJointFrictionOp,
    SetJointPosOp,
    SurfaceSampleRef,
)
from simulac.sdk.environment_service.common.randomize import (
    BboxConstraintSpec,
    DistanceConstraintSpec,
    NonpenetrationConstraintSpec,
)
from simulac.sdk.runner_service.common.model.runtime import (
    CameraRuntime,
    RobotRuntime,
    RuntimeState,
    StuffRuntime,
)
from simulac.sdk.runner_service.common.runner import IRunner
from simulac.sdk.runner_service.common.sampler import ResetSampler
from simulac.sdk.runner_service.local.mujoco.binding import (
    MujocoCameraBinding,
    MujocoRobotBinding,
    MujocoStuffBinding,
)
from simulac.sdk.runner_service.local.mujoco.constraint import (
    MujocoConstraintEvaluation,
    MujocoConstraintEvaluator,
)
from simulac.sdk.runner_service.local.mujoco.context import MujocoNativeContext
from simulac.sdk.runner_service.local.mujoco.resolver import (
    MujocoPlacementResolver,
    MujocoRefResolver,
)
from simulac.sdk.runner_service.local.mujoco.runtime import (
    MujocoCameraRuntimeOps,
    MujocoRobotRuntimeOps,
    MujocoRuntimeStateOps,
    MujocoStuffRuntimeOps,
)

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.environment import IEnvironment
    from simulac.sdk.environment_service.common.model.entity import (
        EnvironmentCameraEntity,
        EnvironmentMachineEntity,
        EnvironmentStuffEntity,
    )
    from simulac.sdk.log_service.common.log_service import ILogService

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
    friction: NotRequired[_SampledFloat]
    size: NotRequired[_SampledSize]


type _ResetCandidate = dict[str, _EntityCandidate]


class MujocoRunner(IRunner):
    """NOTE, FIXME: @gangjeuk
    Now usage pattern of LogService in mujoco_adapter.py and all files in /mujoco is anti-pattern.
    Initialization of *Service MUST be performed by InstantiateService.
    Fix it later!!!
    """

    def __init__(
        self,
        LogService: ILogService,
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
        self.LogService = LogService

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
        self._step_count = 0
        self.resolver: MujocoRefResolver | None = None
        self.placement_resolver: MujocoPlacementResolver | None = None

        # Retry
        self.__MAX_RESET_RETRY = 1000
        self.__reset_passed = False

        # Constriant debugging
        self._reset_failure_count: dict[
            tuple[str, str, tuple[str, ...], tuple[str, ...]],
            int,
        ] = {}
        self._reset_failure_examples: dict[
            tuple[str, str, tuple[str, ...], tuple[str, ...]],
            str,
        ] = {}

    def initialize(self) -> None:
        self._data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self._data)

    def _require_data(self) -> mujoco.MjData:
        if self._data is None:
            raise SimulacBaseError("Runner must be initialized")
        return self._data

    def _runtime_state(self) -> RuntimeState:
        return RuntimeState(
            MujocoRuntimeStateOps(
                self.mj_model,
                self._require_data(),
                step_count=lambda: self._step_count,
                stuff_bindings=self._stuff_bindings,
                machine_bindings=self._machine_bindings,
            )
        )

    def step(self, action: list[float]) -> RuntimeState:
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
        self._step_count += 1
        self.on_after_call_step(self.runner_id)
        return self._runtime_state()

    def tick(self) -> RuntimeState:
        mujoco.mj_step(self.mj_model, self._require_data())
        self._apply_follow_ops(self.resolver)
        self._step_count += 1
        self.on_after_call_step(self.runner_id)
        return self._runtime_state()

    def get_state(self) -> RuntimeState:
        return self._runtime_state()

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
    def reset(self, seed: int | None = 0) -> RuntimeState:
        data = self._require_data()
        sampler = ResetSampler(seed)

        self._clean_runtimes()
        self._step_count = 0

        retry_count = 0

        max_retry = None if self.__reset_passed else self.__MAX_RESET_RETRY
        while max_retry is None or retry_count <= self.__MAX_RESET_RETRY:
            candidate = self._sampling_candidate(sampler)
            """NOTE: @gangjeuk
            Need refactoring. Change code like below
            ```python
            mujoco.mj_resetData(self.mj_model, data)

            self._apply_candidate_model_changes(candidate, sampler)
            mujoco.mj_setConst(self.mj_model, data)

            self._apply_candidate_state_changes(candidate, sampler)
            mujoco.mj_forward(self.mj_model, data)
            ```
            """

            mujoco.mj_resetData(self.mj_model, data)

            mujoco.mj_setConst(self.mj_model, data)
            self._apply_candidate(candidate, sampler)

            mujoco.mj_forward(self.mj_model, data)

            evaluation = self._evaluate_constraints()

            if not evaluation.passed:
                failure_logs: list[str] = []
                for failure in evaluation.failures:
                    failure_key = failure.key()
                    failure_text = failure.format()

                    self._reset_failure_count[failure_key] = (
                        self._reset_failure_count.get(failure_key, 0) + 1
                    )
                    self._reset_failure_examples[failure_key] = failure_text

                    self.LogService.debug(failure_text)
                    failure_logs.append(failure_text)

                if retry_count >= 100 and retry_count % 100 == 0:
                    suspicious_failures = sorted(
                        self._reset_failure_count.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    suspicious_lines = [
                        "---------- Reset sampling warning ----------",
                        (
                            f"Reset count is too high: runner={self.runner_id!r}, "
                            f"environment={self.env.id!r}, retry_count={retry_count}."
                        ),
                        (
                            "The environment has failed to sample a valid reset state many times. "
                            "This usually means one or more constraints are too strict, "
                            "objects are initialized in penetration, or the placement randomization "
                            "does not cover a feasible region."
                        ),
                        "---------- Suspicious constraints ----------",
                        "Most frequent constraint failures:",
                    ]
                    for failure_key, count in suspicious_failures[:10]:
                        example = self._reset_failure_examples[failure_key]
                        suspicious_lines.append(f"  - count={count}: {example}")

                    suspicious_lines.append("Current retry failures:")
                    suspicious_lines.extend(
                        f"  - {failure_text}" for failure_text in failure_logs
                    )

                    self.LogService.warn("\n".join(suspicious_lines))

                retry_count += 1
                continue

            self._create_runtimes()
            self.__reset_passed = True
            return self._runtime_state()
        raise SimulacBaseError("Failed to sample valid reset state")

    def sync(self) -> RuntimeState:
        mujoco.mj_forward(self.mj_model, self._require_data())
        self._apply_follow_ops(self.resolver)
        return self._runtime_state()

    def _debug_render(self):
        return mujoco.viewer.launch_passive(self.mj_model, self._data)

    def context(self, engine: Literal["mujoco"] | None) -> MujocoNativeContext:
        """Get native mujoco context for direct access to MjData and MjModel

        Args:
            engine (Literal[&quot;mujoco&quot;] | None): Engine type. Just ignore it. It's just for typing

        Returns:
            MujocoNativeContext: _description_
        """
        return MujocoNativeContext(
            model=self.mj_model,
            data=self._require_data(),
            stuff_bindings=self._stuff_bindings,
            machine_bindings=self._machine_bindings,
            camera_bindings=self._camera_bindings,
        )

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
            for name in ("mass", "friction", "size"):
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
        self,
        candidate: dict[str, dict[str, Any]],
        sampler: ResetSampler,
    ) -> None:
        data = self._require_data()

        self.resolver = MujocoRefResolver(
            self.mj_model,
            data,
            stuff_bindings=self._stuff_bindings,
            machine_bindings=self._machine_bindings,
            camera_bindings=self._camera_bindings,
        )

        self.placement_resolver = MujocoPlacementResolver(
            data=data,
            resolver=self.resolver,
            stuff_bindings=self._stuff_bindings,
            machine_bindings=self._machine_bindings,
            camera_bindings=self._camera_bindings,
        )

        placement_entities: list[str] = []

        for eid, values in candidate.items():
            binding = self._entity_binding(eid)

            pos = values.get("pos")
            rot = values.get("rot")

            quat = None
            if rot is not None and not isinstance(rot, RefBase):
                quat = euler_to_quat(*rot)

            if isinstance(pos, SurfaceSampleRef):
                placement_entities.append(eid)
                base_pos = None
            elif isinstance(pos, PointRefBase):
                base_pos = self.resolver.resolve_point(pos)
            else:
                base_pos = pos

            if eid in self._camera_bindings:
                self._apply_camera_pose(self._camera_bindings[eid], base_pos, quat)
            else:
                self._apply_root_pose(binding, base_pos, quat)

        mujoco.mj_forward(self.mj_model, data)

        for eid in placement_entities:
            values = candidate[eid]
            binding = self._entity_binding(eid)

            pos = self.placement_resolver.resolve_entity_pos(
                entity_id=eid,
                pos=values["pos"],
            )

            if eid in self._camera_bindings:
                self._apply_camera_pose(self._camera_bindings[eid], pos, None)
            else:
                self._apply_root_pose(binding, pos, None)

        mujoco.mj_forward(self.mj_model, data)

        for op in self.env.relations:
            self._apply_build_op(op, self.resolver, sampler)

        mujoco.mj_forward(self.mj_model, data)

        for eid, values in candidate.items():
            if eid not in self._stuff_bindings:
                continue

            binding = self._stuff_bindings[eid]

            friction = values.get("friction")
            if friction is not None:
                self._apply_stuff_friction(binding, friction)

            mass = values.get("mass")
            if mass is not None:
                self._apply_stuff_mass(binding, mass)

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

    def _evaluate_constraints(self) -> MujocoConstraintEvaluation:
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
        return evaluator.evaluate(self.env.constraints)
