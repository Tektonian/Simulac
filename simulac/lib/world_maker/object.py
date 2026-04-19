from __future__ import annotations

from typing import List, Literal, Tuple, overload

from simulac.base.error.error import SimulacBaseError
from simulac.sdk import obtain_runtime
from simulac.sdk.environment_service.common.model.entity import (
    EnvironmentMachineEntity,
    EnvironmentStuffEntity,
)

from .entity import Camera, Light, Robot, Stuff

# Sentinal pattern: https://python-patterns.guide/python/sentinel-object/
_CREATE_SENTINAL = object()


class Environment:
    """TODO:
    1. Do not create ~Ojbect directly, add create_object or else in EnvironmentBuildService
    """

    def __init__(
        self,
        default_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
        env_uri_or_prebuilt_id: str | None = None,
    ) -> None:
        self._world_maker = obtain_runtime().world_maker

        self.default_engine = default_engine
        self._env = self._world_maker.create_environment(
            default_engine, env_uri_or_prebuilt_id
        )

    @overload
    def place_entity(
        self,
        entity: Stuff,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> StuffObject: ...
    @overload
    def place_entity(
        self,
        entity: Camera,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> CameraObject: ...
    @overload
    def place_entity(
        self,
        entity: Light,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> LightObject: ...
    @overload
    def place_entity(
        self,
        entity: Robot,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> RobotObject: ...
    def place_entity(
        self,
        entity: Stuff | Robot | Camera | Light,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> StuffObject | RobotObject | CameraObject | LightObject:

        if isinstance(entity, Stuff):
            env_stuff_obj = self._world_maker.create_stuff_entity(
                entity.name, entity.obj_uri_or_prebuilt_name, "", ""
            )
            self._world_maker.add_entity(
                self._env.id, env_stuff_obj, pos=pos, quat=quat
            )
            return StuffObject(env_stuff_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Robot):
            env_robot_obj = self._world_maker.create_machine_entity(
                entity.name, entity.obj_uri_or_prebuilt_name
            )
            self._world_maker.add_entity(self._env.id, env_robot_obj)
            return RobotObject(env_robot_obj, _create_sentinal=_CREATE_SENTINAL)
        else:
            raise NotImplementedError("Camera and light are not implemented")


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

        self._world_maker = obtain_runtime().world_maker

        self._runner = self._world_maker.create_runner(env._env.id)

    def step(self, action: List[float]):
        self._runner.step(action)

    def tick(self): ...

    def get_state(self): ...

    def _debug_render(self):
        return self._runner._debug_render()


class StuffObject:
    def __init__(
        self,
        entity: EnvironmentStuffEntity,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

        self._entity = entity

    def set_mass(self, mass: float) -> None: ...

    def set_posture(self, pos: List[float]) -> None: ...

    def set_quat(self, quat: Tuple[float, float, float, float]) -> None: ...


class RobotObject:
    def __init__(
        self,
        entity: EnvironmentMachineEntity,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

        self._entity = entity

    def set_posture(self, pos: List[float]) -> None: ...


class CameraObject:
    def __init__(self):
        """not implemented yet"""


class LightObject:
    def __init__(self):
        """not implemented yet"""


# region Will be implemented


class __World:
    def __init__(self) -> None: ...
    def place_env(self, env: Environment, env_num: int = 1) -> None: ...
    def get_state(self) -> object: ...
    def step(self, actions: list[list[float]]): ...


class __BIV:
    """Brain in a vat? or Agent?"""


# end-region
