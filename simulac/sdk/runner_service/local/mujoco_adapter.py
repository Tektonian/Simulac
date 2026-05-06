from __future__ import annotations

import random
from abc import ABCMeta
from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING, Any, Callable, Literal, MutableMapping, TypedDict

import mujoco
import mujoco.viewer

from simulac.base.error.error import SimulacBaseError
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
    ChoiceRandomSpec,
    EntryRandomSpec,
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
from simulac.sdk.runner_service.local.mujoco.binding import MujocoEntityBinding
from simulac.sdk.runner_service.local.mujoco.runtime import MujocoStuffRuntimeOps

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


_AXIS: dict[str, tuple[float, float, float]] = {
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "front": (0.0, 1.0, 0.0),
    "back": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


class ResetSampler:
    def __init__(self, seed: int | None) -> None:
        self.rng = random.Random(seed)

    def sample(self, value: RefBase | RandomSpec[Any]) -> Any:
        if isinstance(value, RefBase):
            return value
        elif isinstance(value, tuple):
            return value
        print(value)
        typ = value["type"]
        if typ == "uniform":
            sampled = self._uniform(value["min"], value["max"])
            print(sampled)
            return sampled
        if typ == "normal":
            sampled = self._normal(value["mean"], value["std"])
            if "clip_min" in value:
                sampled = self._max_like(sampled, value["clip_min"])
            if "clip_max" in value:
                sampled = self._min_like(sampled, value["clip_max"])
            print(sampled)
            return sampled
        if typ == "choice":
            return self.rng.choice(value["values"])

        raise SimulacBaseError(f"Unsupported random spec: {value}")

    def _is_random_spec(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        rand_type = value.get("type")
        if rand_type == "uniform" or rand_type == "normal" or rand_type == "choice":
            return True
        return False

    def constraints(self, value: Any) -> list[dict[str, Any]]:
        return list(value.get("constraints", [])) if self._is_random_spec(value) else []

    type RandomInputType = float | list[float] | tuple[float]

    def _uniform(
        self,
        lo: RandomInputType,
        hi: RandomInputType,
    ):
        if isinstance(lo, tuple):
            return tuple(self.rng.uniform(a, b) for a, b in zip(lo, hi))
        return self.rng.uniform(lo, hi)

    def _normal(self, mean: RandomInputType, std: RandomInputType):
        if isinstance(mean, tuple):
            return tuple(self.rng.gauss(m, s) for m, s in zip(mean, std))
        return self.rng.gauss(mean, std)

    def _min_like(self, value: RandomInputType, limit: RandomInputType):
        if isinstance(value, tuple):
            return tuple(min(v, l) for v, l in zip(value, limit))
        return min(value, limit)

    def _max_like(self, value: RandomInputType, limit: RandomInputType):
        if isinstance(value, tuple):
            return tuple(max(v, l) for v, l in zip(value, limit))
        return max(value, limit)


class MujocoRefResolver:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

    def _id(self, obj_type: mujoco.mjtObj, entity_id: str, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, f"{entity_id}/{name}")
        if idx < 0:
            raise SimulacBaseError(f"No MuJoCo object named {entity_id}/{name}")
        return idx

    def resolve_point(self, ref: RefBase) -> list[float]:
        if isinstance(ref, WorldPointRef):
            if not isinstance(ref.pos, tuple):
                raise SimulacBaseError(
                    f"Resolved ref point must be tuple {ref}/{ref.pos}"
                )
            pos = ref.pos
            return [float(pos[0]), float(pos[1]), float(pos[2])]

        if isinstance(ref, (AnchorRef, AnchorPosRef)):
            sid = self._id(mujoco.mjtObj.mjOBJ_SITE, ref.entity_id, ref.name)
            return self.data.site_xpos[sid].copy().tolist()

        if isinstance(ref, (ColliderRef, ColliderCenterRef)):
            gid = self._id(mujoco.mjtObj.mjOBJ_GEOM, ref.entity_id, ref.name)
            return self.data.geom_xpos[gid].copy().tolist()

        raise SimulacBaseError(f"Unsupported point ref: {ref}")

    def _joint_dims(self, joint_type: int) -> tuple[int, int]:
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        return 1, 1


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
        entities: dict[str, EnvironmentMachineEntity | EnvironmentStuffEntity],
        bindings: dict[str, MujocoEntityBinding],
        on_after_call_step: Callable[[str], None],
    ) -> None:
        self.runner_type = "mujoco"
        self.runner_id = runner_id
        self.env = env
        self.mj_model = mj_model
        self._entities = entities
        self._bindings = bindings
        self._runtimes = dict[str, StuffRuntime]()
        self.state = {}
        self.on_after_call_step = on_after_call_step
        self._data: mujoco.MjData | None = None

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
        return self._runtimes.get(entity_id)

    def set_state(self) -> None: ...
    def clone_state(self) -> None: ...
    def render(self) -> None: ...
    def reset(self, seed: int | None = 0) -> None:
        data = self._require_data()
        sampler = ResetSampler(seed)
        self._clean_runtime_stuff()
        for _ in range(128):
            candidate = self._sampling_candidate(sampler)
            print(candidate)
            self._create_runtime_stuff()
            mujoco.mj_resetData(self.mj_model, data)
            self._apply_candidate(candidate)
            mujoco.mj_forward(self.mj_model, data)
            return  # self.get_state()
        raise SimulacBaseError("Failed to sample valid reset state")

    def _debug_render(self):
        return mujoco.viewer.launch_passive(self.mj_model, self._data)

    def _sampling_candidate(self, sampler: ResetSampler) -> dict[str, dict[str, Any]]:
        candidate = {}
        for eid, entity in self._entities.items():
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
        return candidate

    def _clean_runtime_stuff(self) -> None:
        self._runtimes: dict[str, StuffRuntime] = dict()

    def _create_runtime_stuff(self) -> None:
        for eid, binding in self._bindings.items():
            ops = MujocoStuffRuntimeOps(eid, self.mj_model, self._data, binding)

            stuff_runtime = StuffRuntime(eid, ops)
            self._runtimes[eid] = stuff_runtime

    def _apply_candidate(self, candidate: dict[str, dict[str, Any]]) -> None:
        data = self._require_data()
        resolver = MujocoRefResolver(self.mj_model, data)

        for eid, values in candidate.items():
            runtime = self._runtimes[eid]
            pos = values.get("pos")
            if isinstance(pos, RefBase):
                pos = resolver.resolve_point(pos)
            if pos is not None:
                runtime.change_pos([float(pos[0]), float(pos[1]), float(pos[2])])

            rot = values.get("rot")
            if rot is not None and not isinstance(rot, RefBase):
                runtime.change_rot(rot)

        mujoco.mj_forward(self.mj_model, data)

        for eid, values in candidate.items():
            ops = [
                placeop
                for placeop in self.env.relations
                if placeop.entity.entity_id == eid
            ]
            for op in ops:
                self._apply_build_op(eid, op, resolver)
                mujoco.mj_forward(self.mj_model, data)
        mujoco.mj_setConst(self.mj_model, data)

    def _apply_build_op(
        self, eid: str, op: BuildOpBase, resolver: MujocoRefResolver
    ) -> None:
        if isinstance(op, PlaceOp):
            entity_id = op.entity.entity_id
            binding = self._bindings.get(entity_id, self._bindings[eid])
            data = self._require_data()

            target_point = resolver.resolve_point(op.target)

            if op.source is None:
                source_pos = data.xpos[binding.root_body_id]
                source_point = [
                    float(source_pos[0]),
                    float(source_pos[1]),
                    float(source_pos[2]),
                ]
            else:
                source_point = resolver.resolve_point(op.source)

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

        raise SimulacBaseError(f"Unsupported build op: {type(op).__name__}")

    def _constraints_pass(self, candidate: dict[str, dict[str, Any]]) -> bool:
        for eid, values in candidate.items():
            for c in values.get("constraints", {}).get("pos", []):
                if not self._constraint_pass(eid, c):
                    return False
        return True

    def _constraint_pass(self, eid: str, c: dict[str, Any]) -> bool:
        typ = c["type"]
        pos = self._require_data().xpos[self._bindings[eid].root_body_id].copy()

        if typ == "bbox":
            lo = c["min"]
            hi = c["max"]
            inside = all(
                float(lo[i]) <= float(pos[i]) <= float(hi[i]) for i in range(3)
            )
            return inside if c["mode"] == "inside" else not inside

        if typ == "distance":
            a, b = c["between"]
            pa = self._require_data().xpos[self._bindings[a].root_body_id]
            pb = self._require_data().xpos[self._bindings[b].root_body_id]
            d = sqrt(
                (float(pa[0]) - float(pb[0])) * (float(pa[0]) - float(pb[0]))
                + (float(pa[1]) - float(pb[1])) * (float(pa[1]) - float(pb[1]))
                + (float(pa[2]) - float(pb[2])) * (float(pa[2]) - float(pb[2]))
            )
            return float(c["min"]) <= d <= float(c["max"])

        raise SimulacBaseError(f"Unsupported constraint: {typ}")


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
        self._bindings: dict[str, MujocoEntityBinding] = {}

        env_ret = self.EnvironmentManagementService.get_environment(self.env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]
        self.env = env

        self._entities = {
            e.id: e for e in [*env.stuffs, *env.machines] if e.id is not None
        }

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
            self._bindings[stuff.id] = self.__build_binding(
                stuff.id, stuff.pos, stuff.rot, "stuff"
            )
        for machine in env.machines:
            self._bindings[machine.id] = self.__build_binding(
                machine.id, machine.pos, machine.rot, "machine"
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
            entities=self._entities,
            bindings=self._bindings,
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

    def __build_binding(
        self,
        entity_id: str,
        pos: RandomizableVec3,
        rot: RandomizableVec3,
        kind: Literal["stuff", "machine"],
    ) -> MujocoEntityBinding:
        root_name = f"{entity_id}/__root__"
        root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, root_name
        )

        if root_body_id < 0:
            raise SimulacBaseError(f"No MuJoCo root body for entity {entity_id!r}")

        body_ids = _subtree_body_ids(self.model, root_body_id)
        geom_ids = [
            gid
            for gid in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gid]) in body_ids
        ]
        joint_ids = [
            jid
            for jid in range(self.model.njnt)
            if int(self.model.jnt_bodyid[jid]) in body_ids
        ]
        actuator_ids = [
            aid
            for aid in range(self.model.nu)
            if int(self.model.actuator_trnid[aid][0]) in joint_ids
        ]

        root_freejoint_id = -1
        for jid in joint_ids:
            if (
                int(self.model.jnt_bodyid[jid]) == root_body_id
                and self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
            ):
                root_freejoint_id = jid
                break

        return MujocoEntityBinding(
            entity_id=entity_id,
            kind=kind,
            root_body_id=root_body_id,
            body_ids=body_ids,
            pos=pos,
            rot=rot,
            geom_ids=geom_ids,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
            root_freejoint_id=root_freejoint_id,
            mocap_id=int(self.model.body_mocapid[root_body_id]),
        )
