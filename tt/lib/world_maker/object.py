from __future__ import annotations
from typing import List, Literal, Tuple

from torch import seed

from tt.base.error.error import TektonianBaseError
from tt.lib.world_maker.entity import Robot, Stuff

from tt.sdk import (
    environment_management_service,
    environment_build_service,
    runner_management_service,
)


class Environment:
    """TODO:
    1. Do not create ~Ojbect directly, add create_object or else in EnvironmentBuildService
    """

    def __init__(
        self,
        default_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
        env_uri_or_prebuilt_id: str | None = None,
    ) -> None:

        self.default_engine = default_engine
        env_ret = environment_management_service.create_environment(default_engine)

        if env_ret[0] is None:
            raise env_ret[1]

        self._env = env_ret[0]

    def place_stuff_entity(
        self,
        stuff: Stuff,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> StuffObject:

        stuff.entity.pos = pos
        stuff.entity.quat = quat

        entity_id = environment_build_service.add_entity(self._env.id, stuff.entity)

        stuff_object = StuffObject(entity_id, _prevent_user_direct_call=False)

        return stuff_object

    def place_robot_entity(
        self,
        robot: Robot,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> RobotObject:
        robot.entity.pos = pos
        robot.entity.quat = quat

        entity_id = environment_build_service.add_entity(self._env.id, robot.entity)

        robot_object = RobotObject(entity_id, _prevent_user_direct_call=False)

        return robot_object


class Runner:
    def __init__(
        self,
        env: Environment,
        seed: int | None = 0,
        tick: int | None = 5,  # 5ms
        /,
        *,
        runtime_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
    ):
        self.seed = seed
        self.tick_time = tick

        runner_ret = runner_management_service.create_runner(env._env.id)

        if runner_ret[0] is None:
            raise runner_ret[1]

        self._runner = runner_ret[0]

    def step(self, action: List[float]):
        self._runner.step(action)

    def tick(self): ...

    def get_state(self): ...


class StuffObject:
    def __init__(
        self, entity_id: str, /, *, _prevent_user_direct_call: bool = True
    ) -> None:
        if _prevent_user_direct_call == True:
            raise TektonianBaseError("Please do not create stuff object directly")

        self.entity_id = entity_id

    def set_mass(self, mass: float) -> None: ...

    def set_posture(self, pos: List[float]) -> None: ...

    def set_quat(self, quat: Tuple[float, float, float, float]) -> None: ...


class RobotObject:
    def __init__(
        self, entity_id: str, /, *, _prevent_user_direct_call: bool = True
    ) -> None:
        if _prevent_user_direct_call == True:
            raise TektonianBaseError("Please do not create stuff object directly")

        self.entity_id = entity_id

    def set_posture(self, pos: List[float]) -> None: ...


# region Will be implemented


class __World:
    def __init__(self) -> None: ...
    def place_env(self, env: Environment, env_num: int = 1) -> None: ...
    def get_state(self) -> object: ...
    def step(self, actions: list[list[float]]): ...


class __BIV:
    """Brain in a vat? or Agent?"""


# end-region
