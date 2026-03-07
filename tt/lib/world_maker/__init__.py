from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Optional, Tuple, Literal, TypeVar


class World:
    def __init__(self) -> None:
        pass

    def place_env(self, env: Environment, env_num: int = 1) -> None: ...

    def get_state(self) -> object: ...

    def step(self, actions: list[float]): ...


class Environment:
    def __init__(
        self,
        env_uri_or_prebuilt_name: Optional[str] = None,
    ) -> None: ...

    def place_entity(
        self,
        entity: StuffObject,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> StuffObject: ...

    def place_robot_entity(self, robot: RobotEntity) -> RobotObject: ...

    def get_state(self) -> object: ...

    def replace_object(
        self, stuff_object: StuffObject, stuff: Stuff
    ) -> StuffObject: ...


class Stuff:
    def __init__(
        self,
        obj_uri_or_prebuilt_name: str,
        fixed: bool = True,
    ) -> None:
        pass


class StuffObject:
    def __init__(self) -> None: ...

    def set_mass(self, mass: float) -> None: ...

    def set_pos(self, pos: Tuple[float, float, float]) -> None: ...

    def set_friction(self, friction: float) -> None: ...

    def set_quat(
        self,
        quat: Tuple[float, float, float, float],
    ) -> None: ...


class RobotObject:
    def __init__(self) -> None: ...

    def set_target_position(self, control: list[float]): ...

    def set_target_velocity(self, control: list[float]): ...

    def set_target_force(self, control: list[float]): ...


class Robot:
    def __init__(
        self,
        obj_uri_or_prebuilt_name: str,
    ) -> None:
        pass


class Runner:
    def __init__(
        self,
        env: Environment,
        /,
        *,
        seed: Optional[int] = None,  # FIXME: need?
        tick: Optional[int] = 5,  # 5ms
        physics_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
    ): ...

    def tick(self): ...


class BIV:
    def __init__(self) -> None:
        pass
