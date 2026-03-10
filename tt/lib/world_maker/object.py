from __future__ import annotations
from typing import List, Literal, Tuple

from torch import seed

from tt.base.error.error import TektonianBaseError
from tt.lib.world_maker.entity import Robot, Stuff


class Environment:
    def __init__(self, default_engine: Literal['mujoco', 'newton', 'genesis'] = 'mujoco', env_uri_or_prebuilt_id: str | None) -> None:
        self.default_engine = default_engine
        
    def place_stuff_entity(
        self,
        stuff: Stuff,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> StuffObject: ...

    def place_robot_entity(
        self,
        robot: Robot,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> RobotObject:...
        

class Runner:
    def __init__(
        self,
        env: Environment,
        seed: int | None = 0,
        tick: int | None = 5, # 5ms
        /,
        *,
        runtime_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
    ):
        self.seed = seed
        self.tick_time = tick

    def step(self, action: List[float]): ...
    
    def tick(self): ...
    
    def get_state(self): ...

class StuffObject:
    def __init__(
        self, entity_id: str, *, __prevent_user_direct_call: bool = True
    ) -> None:
        if __prevent_user_direct_call == True:
            raise TektonianBaseError("Please do not create stuff object directly")

        self.entity_id = entity_id

    def set_mass(self, mass: float) -> None: ...

    def set_pos(self, pos: Tuple[float, float, float]) -> None: ...

    def set_quat(self, quat: Tuple[float, float, float, float]) -> None: ...


class RobotObject:
    def __init__(
        self, entity_id: str, *, __prevent_user_direct_call: bool = True
    ) -> None:
        if __prevent_user_direct_call == True:
            raise TektonianBaseError("Please do not create stuff object directly")

        self.entity_id = entity_id


# region Will be implemented

class __World:
    def __init__(self) -> None:...
    def place_env(self, env: Environment, env_num: int = 1) -> None:...
    def get_state(self) -> object:...
    def step(self, actions: list[list[float]]): ...


class __BIV:
    """Brain in a vat? or Agent?
    """
# end-region