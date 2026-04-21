from __future__ import annotations

from typing import List, Literal, Tuple, final, overload

from simulac.base.error.error import SimulacBaseError
from simulac.sdk import obtain_runtime
from simulac.sdk.environment_service.common.model.entity import (
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
    EnvironmentMachineEntity,
    EnvironmentStuffEntity,
)

from .entity import Camera, Light, Robot, Stuff

# Sentinal pattern: https://python-patterns.guide/python/sentinel-object/
_CREATE_SENTINAL = object()


class Environment:
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
        description: str | None = None,
    ) -> StuffObject: ...
    @overload
    def place_entity(
        self,
        entity: Camera,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        description: str | None = None,
    ) -> CameraObject: ...
    @overload
    def place_entity(
        self,
        entity: Light,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        description: str | None = None,
    ) -> LightObject: ...
    @overload
    def place_entity(
        self,
        entity: Robot,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        description: str | None = None,
    ) -> RobotObject: ...
    def place_entity(
        self,
        entity: Stuff | Robot | Camera | Light,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        description: str | None = None,
    ) -> StuffObject | RobotObject | CameraObject | LightObject:
        description = description or ""

        if isinstance(entity, Stuff):
            env_stuff_obj = self._world_maker.create_stuff_entity(
                entity.name or "", description, entity.obj_uri_or_prebuilt_name, "", ""
            )
            self._world_maker.add_entity(
                self._env.id, env_stuff_obj, pos=pos, quat=quat
            )
            return StuffObject(env_stuff_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Robot):
            env_robot_obj = self._world_maker.create_machine_entity(
                entity.name or "", description, entity.obj_uri_or_prebuilt_name
            )
            self._world_maker.add_entity(
                self._env.id, env_robot_obj, pos=pos, quat=quat
            )
            return RobotObject(env_robot_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Camera):
            env_camera_obj = self._world_maker.create_camera_entity(
                entity.name, description, entity.type
            )
            self._world_maker.add_entity(
                self._env.id, env_camera_obj, pos=pos, quat=quat
            )
            return CameraObject(env_camera_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Light):  # pyright: ignore[reportUnnecessaryIsInstance]
            env_light_obj = self._world_maker.create_light_entity(
                entity.name, description, entity.type
            )
            self._world_maker.add_entity(
                self._env.id, env_light_obj, pos=pos, quat=quat
            )
            return LightObject(env_light_obj, _create_sentinal=_CREATE_SENTINAL)
        else:
            raise NotImplementedError("Camera and light are not implemented")

    @overload
    def remove_object(
        self,
        object_or_object_id: StuffObject | RobotObject | CameraObject | LightObject,
    ) -> None: ...
    @overload
    def remove_object(self, object_or_object_id: str) -> None: ...
    def remove_object(
        self,
        object_or_object_id: StuffObject
        | RobotObject
        | CameraObject
        | LightObject
        | str,
    ) -> None:
        pass

    def get_object(
        self, object_id: str
    ) -> StuffObject | RobotObject | CameraObject | LightObject: ...

    def dump_env(self) -> dict:
        """Return definition of environment.
        Return type `dict` is json format

        Raises:
            SimulacBaseError: _description_

        Returns:
            dict: json format environment definition
        """
        ...


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
    def set_pos(self, pos: tuple[float, float, float]) -> None: ...
    def set_rot(self, rot: tuple[float, float, float]) -> None: ...
    def set_size(self, size: tuple[float, float, float]) -> None: ...
    def set_fixed(self, is_fixed: bool) -> None: ...


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

    def set_pos(self, pos: tuple[float, float, float]) -> None: ...
    def set_rot(self, rot: tuple[float, float, float]) -> None: ...
    def set_act_pos(self, pos: list[float]) -> None: ...

    def get_act_min(self) -> list[float]: ...
    def get_act_max(self) -> list[float]: ...


class CameraObject:
    def __init__(
        self,
        entity: EnvironmentCameraEntity,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")
        self._entity = entity

    def set_pos(self, pos: tuple[float, float, float]) -> None: ...
    def set_rot(self, rot: tuple[float, float, float]) -> None: ...
    def set_lookat(self, lookat: tuple[float, float, float]) -> None: ...
    def set_fov(self, fov: float) -> None: ...
    def set_aspect(self, aspect: float) -> None: ...
    def set_near(self, near: float) -> None: ...
    def set_far(self, far: float) -> None: ...

    def set_type(
        self,
        type: Literal[
            "rgb", "tactile", "depth", "pointcloud", "normal", "segmentation"
        ],
    ): ...


class LightObject:
    def __init__(
        self,
        entity: EnvironmentLightEntity,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")
        self._entity = entity

    def set_pos(self, pos: tuple[float, float, float]) -> None: ...
    def set_rot(self, rot: tuple[float, float, float]) -> None: ...
    def set_intensity(self, intensity: float) -> None: ...
    def set_type(
        self, type: Literal["ambient", "pointlight", "reactarea", "spot"]
    ) -> None: ...
    def set_color(self, color: tuple[int, int, int]) -> None: ...


# region Will be implemented


class __World:
    def __init__(self) -> None: ...
    def place_env(self, env: Environment, env_num: int = 1) -> None: ...
    def get_state(self) -> object: ...
    def step(self, actions: list[list[float]]): ...


class __BIV:
    """Brain in a vat? or Agent?"""


# end-region
