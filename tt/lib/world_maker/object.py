from __future__ import annotations

from typing import List, Literal, Tuple, overload

from tt.base.error.error import TektonianBaseError
from tt.sdk import obtain_runtime

from .entity import Camera, Light, Robot, Stuff


class Environment:
    """TODO:
    1. Do not create ~Ojbect directly, add create_object or else in EnvironmentBuildService
    """

    def __init__(
        self,
        default_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
        env_uri_or_prebuilt_id: str | None = None,
    ) -> None:
        self.world_maker = obtain_runtime().world_maker

        self.default_engine = default_engine
        self._env = self.world_maker.create_environment(
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
            # change entity pos and quat
            # should call 'self.world_maker.change_eneity_pos' ?
            entity._entity.pos = pos
            entity._entity.quat = quat
            entity_id = self.world_maker.add_entity(
                self._env.id, entity._entity, pos=pos, quat=quat
            )
            return StuffObject(entity_id, _prevent_user_direct_call=True)
        elif isinstance(entity, Robot):
            # change entity pos and quat
            # should call 'self.world_maker.change_eneity_pos' ?
            entity._entity.pos = pos
            entity._entity.quat = quat

            entity_id = self.world_maker.add_entity(self._env.id, entity._entity)

            return RobotObject(entity_id, _prevent_user_direct_call=True)
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

    def __debug_render(self):
        return self._runner.__debug_render()


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
