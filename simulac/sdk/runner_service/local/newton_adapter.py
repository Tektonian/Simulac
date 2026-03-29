from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, MutableMapping

# Lazy imports for solve dependency issue and heavy package import issue
# Will be solved? https://peps.python.org/pep-0810/
# import warp as wp
# import newton
from simulac.base.error.error import SimulacBaseError
from simulac.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterState,
)
from simulac.sdk.runner_service.common.runner import IRunner

if TYPE_CHECKING:
    import newton
    import warp as wp

    from simulac.sdk.environment_service.common.environment_service import (
        IEnvironmentManagementService,
    )
    from simulac.sdk.log_service.common.log_service import ILogService
    from simulac.sdk.runner_service.common.runner_service import (
        IRunnerManagementService,
    )


class NewtonRunner(IRunner):
    def __init__(self, env_id: str, runner_idx: int, step: Callable[..., None]) -> None:
        self.runner_type = "newton"
        self.id = ""
        self.env_id = env_id

        self._runner_idx = runner_idx
        self._step = step

    def initialize(self) -> None:
        """Do nothing.
        Initialization will be hanlded on NewtonAdapter
        """

    def step(self, action: list[float]) -> None:
        self._step(self._runner_idx, action)

    def tick(self) -> None: ...

    def clone_state(self) -> None: ...
    def get_state(self) -> None: ...
    def render(self) -> None: ...
    def reset(self) -> None: ...
    def set_state(self) -> None: ...
    def _debug_render(self) -> Any: ...


class NewtonAdapter(IPhysicsEngineAdapter):
    def __init__(
        self,
        env_id: str,
        LogService: ILogService,
        RunnerManagementService: IRunnerManagementService,
        EnvironmentManagementService: IEnvironmentManagementService,
    ) -> None:
        try:
            # lazy import
            import newton
            import warp as wp
        except ImportError as e:
            raise ImportError(
                "".join(
                    [
                        "To run newton simulation. Please install 'newton' and 'warp' packages.",
                        "Use 'uv add newton' or 'pip install newton'",
                    ]
                )
            ) from e

        self.newton = newton
        self.wp = wp

        self.env_id = env_id
        self.LogService = LogService
        self.RunnerManagementService = RunnerManagementService
        self.EnvironmentManagementService = EnvironmentManagementService

        self._runner_count = 0
        self._step_count = 0
        self._step_count_map: MutableMapping[str, int] = dict()

        self.builder = newton.ModelBuilder()
        self.__scene = newton.ModelBuilder()

        self.model: newton.Model | None = None

        self._solver: newton.solvers.SolverBase | None = None
        self._state_0: newton.State | None = None
        self._state_1: newton.State | None = None
        self._control: newton.Control | None = None
        self._contacts: newton.Contacts | None = None
        self._viewer: newton.viewer.ViewerGL = newton.viewer.ViewerGL()

        env_ret = self.EnvironmentManagementService.get_environment(self.env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]

        self.env = env

        objects = self.env.objects

        for obj in objects:
            print(obj.physics.mjcf_uri)

            # TODO: remove if else code later
            if "franka" in obj.physics.mjcf_uri:
                self.__scene.add_urdf(obj.physics.mjcf_uri)
            else:
                # TODO: URDF file is not handled here. Need handling code
                self.__scene.add_mjcf(obj.physics.mjcf_uri, xform=(obj.pos + obj.quat))

    def initialize(self) -> None: ...

    def create_runner(self) -> IRunner:
        if self._step_count != 0:
            raise SimulacBaseError(
                "Cannot create new runner after calling step() function"
            )

        def runner_step(runner_idx: int, action: list[float]):
            if self.model is None:
                self.builder.replicate(self.__scene, world_count=self._runner_count)

                # TODO: Remove later
                self.__set_debug_env(self._runner_count)

                self.model = self.builder.finalize()
                self._solver = self.newton.solvers.SolverMuJoCo(self.model)
                self._state_0 = self.model.state()
                self._state_1 = self.model.state()
                self._control = self.model.control()
                self._contacts = self.model.contacts()

                # TODO: Remove later
                self.__set_debug_viewer(0)

            if self.model.max_dofs_per_articulation != len(action):
                self.LogService.warn(
                    f"Action dimension is not fit. expected: [{self.model.max_dofs_per_articulation}] input: [{len(action)}]"
                )

            if (
                self._control is None
                or self._contacts is None
                or self._state_0 is None
                or self._state_1 is None
                or self._solver is None
            ):
                # Maybe not reachable, just for type safing
                assert False, "unreachable"

            # Compute segment for this world instance
            s1 = self.model.max_dofs_per_articulation * runner_idx
            s2 = s1 + self.model.max_dofs_per_articulation

            # Copy action into the segment using wp.copy to avoid slice assignment
            src: wp.array[wp.float32] = self.wp.array(action, dtype=wp.float32)
            count = s2 - s1  # == self.model.max_dofs_per_articulation
            self.wp.copy(
                self._control.joint_target_pos,
                src,
                dest_offset=s1,
                src_offset=0,
                count=count,
            )

            for _ in range(10):
                self._state_0.clear_forces()
                self._solver.step(
                    self._state_0,
                    self._state_1,
                    self._control,
                    self._contacts,
                    1.0 / 60.0 / 4.0,  # TODO change with tick
                )
                self._state_0, self._state_1 = self._state_1, self._state_0
            self._step_count += 1
            self._step_count_map[f"run_{runner_idx}"] += 1

            # TODO: Remove later
            self.__set_debug_viewer((1.0 / 60.0 / 4.0) * self._step_count)

        runner = NewtonRunner(
            env_id=self.env.id, runner_idx=self._runner_count, step=runner_step
        )

        new_runner_id = f"run_{self._runner_count}"
        self.LogService.debug(
            f"new newton runner created env_id: {self.env_id} runner_id: {new_runner_id}"
        )
        self._step_count_map[new_runner_id] = 0
        self._runner_count += 1

        return runner

    def get_state(self) -> IPhysicsEngineAdapterState:
        return IPhysicsEngineAdapterState(
            self.env_id, self._runner_count, self._step_count_map
        )

    def __set_debug_env(self, count: int) -> None:
        """Debug purpose for now, handle 'fr3_franka_hand.urdf' asset setting"""
        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        for idx in range(count):
            s_b_9, s_a_9 = idx * 9, (idx + 1) * 9
            self.builder.joint_q[s_b_9:s_a_9] = [*init_q, 0.05, 0.05]
            self.builder.joint_target_pos[s_b_9:s_a_9] = [*init_q, 1.0, 1.0]

            """
            Newton MUST set this value manually for mjcf -> Think MJCF file parser error
            """
            self.builder.joint_target_ke[s_b_9:s_a_9] = [650.0] * 9
            self.builder.joint_target_kd[s_b_9:s_a_9] = [100.0] * 9

            self.builder.joint_effort_limit[s_b_9 : s_b_9 + 7] = [80.0] * 7
            self.builder.joint_effort_limit[s_b_9 + 7 : s_a_9] = [20.0] * 2
            self.builder.joint_armature[s_b_9 : s_b_9 + 7] = [0.1] * 7
            self.builder.joint_armature[s_b_9 + 7 : s_a_9] = [0.5] * 2

    def __set_debug_viewer(self, time: float):
        if self._step_count == 0:
            self._viewer.set_model(self.model)

        self._viewer.begin_frame(time)
        self._viewer.log_state(self._state_0)
        self._viewer.end_frame()
