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
from simulac.base.types.geometry import Vec3
from simulac.base.utils.rotation import euler_to_quat
from simulac.sdk.environment_service.common.model.ref import (
    AnchorPosRef,
    AnchorRef,
    BuildOpBase,
    ColliderCenterRef,
    ColliderRef,
    JointAxisRef,
    JointRef,
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
from simulac.sdk.runner_service.common.model.runtime import StuffRuntime
from simulac.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterState,
)
from simulac.sdk.runner_service.common.runner import IRunner, IRunnerFactory
from simulac.sdk.runner_service.common.runner_service import IRunnerManagementService
from simulac.sdk.runner_service.common.sampler import ResetSampler
from simulac.sdk.runner_service.local.mujoco.binding import (
    MujocoActuatorBinding,
    MujocoJointBinding,
    MujocoLinkBinding,
    MujocoRobotBinding,
    MujocoStuffBinding,
)
from simulac.sdk.runner_service.local.mujoco.runtime import (
    MujocoRobotRuntimeOps,
    MujocoStuffRuntimeOps,
)

from .mujoco.resolver import MujocoRefResolver

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.environment import IEnvironment
    from simulac.sdk.environment_service.common.environment_service import (
        IEnvironmentManagementService,
    )
    from simulac.sdk.environment_service.common.model.entity import (
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
        stuff_bindings: dict[str, MujocoStuffBinding],
        machine_bindings: dict[str, MujocoRobotBinding],
        on_after_call_step: Callable[[str], None],
    ) -> None:
        self.runner_type = "mujoco"
        self.runner_id = runner_id
        self.env = env
        self.mj_model = mj_model
        self._stuff_entities = stuff_entities
        self._machine_entities = machine_entities
        self._stuff_bindings = stuff_bindings
        self._machine_bindings = machine_bindings
        self._stuff_runtimes = dict[str, StuffRuntime]()
        self.state = {}
        self.on_after_call_step = on_after_call_step
        self._data: mujoco.MjData | None = None

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
        self.on_after_call_step(self.runner_id)

    def tick(self) -> None:
        mujoco.mj_step(self.mj_model, self._require_data())

    # FIXME: debug purpose for now. Should return state info mapped with self._env
    def get_state(self) -> None:
        for i in range(self.mj_model.nbody):
            print(self._data.body(i))
        breakpoint()

    def get_runtime_object(self, entity_id: str):
        # breakpoint()
        return self._stuff_runtimes.get(entity_id)

    def set_state(self) -> None: ...
    def clone_state(self) -> None: ...
    def render(self) -> None: ...
    def reset(self, seed: int | None = 0) -> None:
        data = self._require_data()
        sampler = ResetSampler(seed)

        self._clean_runtime_stuff()

        retry_count = 0
        while self.__reset_passed or retry_count <= self.__MAX_RESET_RETRY:
            candidate = self._sampling_candidate(sampler)

            mujoco.mj_resetData(self.mj_model, data)

            self._apply_candidate(candidate, sampler)

            mujoco.mj_setConst(self.mj_model, data)
            mujoco.mj_forward(self.mj_model, data)

            if not self._constraints_pass(candidate):
                continue
            self._create_runtime_stuff()
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
        return candidate

    def _clean_runtime_stuff(self) -> None:
        self._stuff_runtimes: dict[str, StuffRuntime] = dict()

    def _create_runtime_stuff(self) -> None:
        for eid, binding in self._stuff_bindings.items():
            ops = MujocoStuffRuntimeOps(
                eid, self.mj_model, self._require_data(), binding
            )

            stuff_runtime = StuffRuntime(eid, ops)
            self._stuff_runtimes[eid] = stuff_runtime

    def _apply_candidate(
        self, candidate: dict[str, dict[str, Any]], sampler: ResetSampler
    ) -> None:
        data = self._require_data()
        resolver = MujocoRefResolver(
            self.mj_model,
            data,
            stuff_bindings=self._stuff_bindings,
            machine_bindings=self._machine_bindings,
        )

        for eid, values in candidate.items():
            binding = self._entity_binding(eid)
            pos = values.get("pos")
            if isinstance(pos, RefBase):
                pos = resolver.resolve_point(sampler.sample(pos))

            rot = values.get("rot")
            quat = None
            if rot is not None and not isinstance(rot, RefBase):
                quat = euler_to_quat(*rot)

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
                self._apply_build_op(eid, op, resolver, sampler)
                mujoco.mj_forward(self.mj_model, data)

    def _entity_binding(
        self,
        entity_id: str,
    ) -> MujocoStuffBinding | MujocoRobotBinding:
        binding = self._stuff_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self._machine_bindings.get(entity_id)
        if binding is not None:
            return binding

        raise SimulacBaseError(f"No MuJoCo binding for entity {entity_id!r}")

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

    def _constraints_pass(self, candidate: dict[str, dict[str, Any]]) -> bool:
        for eid, values in candidate.items():
            for c in values.get("constraints", {}).get("pos", []):
                if not self._constraint_pass(eid, c):
                    return False
        return True

    def _constraint_pass(self, eid: str, constraint: _ConstraintSpec) -> bool:
        typ = constraint["type"]

        if typ == "bbox":
            return self._bbox_constraint_pass(eid, constraint)
        if typ == "distance":
            return self._distance_constraint_pass(constraint)
        if typ == "nonpenetration":
            return self._nonpenetration_constraint_pass(constraint)

        raise SimulacBaseError(f"Unsupported constraint: {typ}")

    def _bbox_constraint_pass(self, eid: str, constraint: BboxConstraintSpec):
        binding = self._entity_binding(eid)
        pos = self._require_data().xpos[binding.root_body_id]

        lo = constraint["min"]
        hi = constraint["max"]

        inside = all(float(lo[i]) <= float(pos[i]) <= float(hi[i]) for i in range(3))

        mode = constraint.get("mode", "inside")
        if mode == "inside":
            return inside

        if mode == "outside":
            return not inside

        raise SimulacBaseError(f"Unsupported bbox constraint mode: {mode}")

    def _distance_constraint_pass(self, constraint: DistanceConstraintSpec):
        a, b = constraint["between"]

        a_binding = self._entity_binding(a)
        b_binding = self._entity_binding(b)

        data = self._require_data()
        pa = data.xpos[a_binding.root_body_id]
        pb = data.xpos[b_binding.root_body_id]

        dx = float(pa[0]) - float(pb[0])
        dy = float(pa[1]) - float(pb[1])
        dz = float(pa[2]) - float(pb[2])

        distance = sqrt(dx * dx + dy * dy + dz * dz)

        return float(constraint["min"]) <= distance <= float(constraint["max"])

    def _nonpenetration_constraint_pass(
        self, constraint: NonpenetrationConstraintSpec
    ) -> bool:
        between = constraint["between"]
        if len(between) < 2:
            raise SimulacBaseError(
                "nonpenetration constraint requires at least two entities"
            )

        for idx, a in enumerate(between):
            for b in between[idx + 1 :]:
                if not self._nonpenetration_pair_pass(a, b):
                    return False

        return True

    def _nonpenetration_pair_pass(self, a: str, b: str) -> bool:
        data = self._require_data()
        a_geoms = set(self._entity_binding(a).geom_ids)
        b_geoms = set(self._entity_binding(b).geom_ids)
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist >= -1e-5:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in a_geoms and g2 in b_geoms) or (g2 in a_geoms and g1 in b_geoms):
                return False
        return True

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

        env_ret = self.EnvironmentManagementService.get_environment(self.env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]
        self.env = env

        self._stuff_entities = {e.id: e for e in env.stuffs if e.id is not None}
        self._machine_entities = {e.id: e for e in env.machines if e.id is not None}

        for stuff in env.stuffs:
            self.__add_stuff(stuff)
        for machine in env.machines:
            self.__add_machine(machine)
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
            machine_entities=self._machine_entities,
            stuff_bindings=self._stuff_bindings,
            machine_bindings=self._machine_bindings,
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
