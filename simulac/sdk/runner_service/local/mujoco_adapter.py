from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING, Callable, MutableMapping

import mujoco
import mujoco.viewer

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterState,
)
from simulac.sdk.runner_service.common.runner import IRunner, IRunnerFactory
from simulac.sdk.runner_service.common.runner_service import IRunnerManagementService

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


@dataclass(slots=True)
class MujocoEntityBinding:
    entity_id: str
    kind: Literal["stuff", "machine", "camera", "light"]
    root_body_id: int
    pos: RandomizableVec3
    rot: RandomizableVec3
    body_ids: list[int] = field(default_factory=list[int])
    geom_ids: list[int] = field(default_factory=list[int])
    joint_ids: list[int] = field(default_factory=list[int])
    actuator_ids: list[int] = field(default_factory=list[int])
    root_freejoint_id: int = -1
    mocap_id: int = -1

class MujocoRunner(IRunner):
    def __init__(
        self,
        runner_id: str,
        env_id: str,
        mj_model: mujoco.MjModel,
        entities: dict[str, EnvironmentMachineEntity | EnvironmentStuffEntity],
        bindings: dict[str, MujocoEntityBinding],
        on_after_call_step: Callable[[str], None],
    ) -> None:
        self.runner_type = "mujoco"
        self.runner_id = runner_id
        self.env_id = env_id
        self.mj_model = mj_model
        self._entities = entities
        self._bindings = bindings
        self.state = {}
        self.on_after_call_step = on_after_call_step
        self._data: mujoco.MjData | None = None

    def initialize(self) -> None:
        self._data = mujoco.MjData(self.mj_model)

    def step(self, action: list[float]) -> None:
        if self._data is None:
            raise SimulacBaseError("Runner must be initialized before step()")

        self._data.ctrl = action

        mujoco.mj_step(self.mj_model, self._data)

        self.on_after_call_step(self.runner_id)

    def tick(self) -> None:
        mujoco.mj_step(self.mj_model, self._require_data())

    # FIXME: debug purpose for now. Should return state info mapped with self._env
    def get_state(self) -> None:
        for i in range(self.mj_model.nbody):
            print(self._data.body(i))
        breakpoint()

    def set_state(self) -> None: ...
    def clone_state(self) -> None: ...
    def render(self) -> None: ...
    def reset(self) -> None:
        if self._data is not None:
            mujoco.mj_resetData(self.mj_model, self._data)
    def reset(self, seed: int | None = 0) -> None:

    def _debug_render(self):
        return mujoco.viewer.launch_passive(self.mj_model, self._data)


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
            self.env_id,
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
