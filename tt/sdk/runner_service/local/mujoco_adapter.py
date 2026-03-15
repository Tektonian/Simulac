from __future__ import annotations
from abc import ABCMeta
from typing import TYPE_CHECKING, Callable, MutableMapping

import mujoco
import mujoco.viewer

from tt.base.error.error import TektonianBaseError
from tt.sdk.runner_service.common.runner import IRunner, IRunnerFactory
from tt.sdk.runner_service.common.runner_service import IRunnerManagementService

from tt.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterState,
)

if TYPE_CHECKING:
    from tt.sdk.environment_service.common.model.entity import (
        EnvironmentMJCFObjectEntity,
    )
    from tt.sdk.log_service.common.log_service import ILogService
    from tt.sdk.environment_service.common.environment_service import (
        IEnvironmentManagementService,
    )
    from tt.sdk.environment_service.common.environment import IEnvironment


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


class MujocoRunner(IRunner):
    def __init__(
        self,
        runner_id: str,
        env_id: str,
        mj_model: mujoco.MjModel,
        on_after_call_step: Callable[[str], None],
    ) -> None:
        self.runner_type = "mujoco"
        self.runner_id = runner_id
        self.mj_model = mj_model
        self.env_id = env_id
        self.state = {}
        self.on_after_call_step = on_after_call_step

        self._data: mujoco.MjData | None = None

    def initialize(self) -> None:
        self._data = mujoco.MjData(self.mj_model)

    def step(self, action: list[float]) -> None:
        if self._data is None:
            raise TektonianBaseError("Runner must be initialized before step()")

        self._data.ctrl = action

        mujoco.mj_step(self.mj_model, self._data)

        self.on_after_call_step(self.runner_id)

    def tick(self) -> None:
        mujoco.mj_step(self.mj_model, self._data)

    # FIXME: debug purpose for now. Should return state info mapped with self._env
    def get_state(self) -> None:
        for i in range(self.mj_model.nbody):
            print(self._data.body(i))
        breakpoint()

    def set_state(self) -> None: ...
    def clone_state(self) -> None: ...
    def render(self) -> None: ...
    def reset(self) -> None: ...

    def __debug_render(self):
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

        env_ret = self.EnvironmentManagementService.get_environment(self.env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]

        objects = env.objects

        for obj in objects:
            # TODO: URDF file is not handled here. Need handling code
            child = mujoco.MjSpec.from_file(obj.physics.mjcf_uri)

            child_bodies: list[mujoco.MjsBody] = child.bodies
            # change pos
            for body in child_bodies:
                if body.parent == child.worldbody:
                    body.pos = list(obj.pos)
                    body.quat = list(obj.quat)

            self.root_spec.attach(child, frame=self.root_frame, prefix=obj.uuid)
        """
        TODO: 
            1. implement mujoco parallel
            2. implement robos initialization code
            
        """
        self.model = self.root_spec.compile()

    def create_runner(self) -> IRunner:
        if self._step_count != 0:
            raise TektonianBaseError(
                "Cannot create new runner after calling step() function"
            )

        if self.model is None:
            raise TektonianBaseError("Adapter not initialized")

        def on_after_runner_step(runner_id: str):
            self._step_count_map[runner_id] += 1
            self._step_count += 1

        new_runner_id = f"run_{self._runner_count}"

        runner = MujocoRunner(
            new_runner_id,
            self.env_id,
            mj_model=self.model,
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
