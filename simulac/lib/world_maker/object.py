from __future__ import annotations

from typing import Any, Generic, List, Literal, Tuple, cast, overload

from simulac.base.error.error import SimulacBaseError
from simulac.sdk import obtain_runtime
from simulac.sdk.environment_service.common.model.entity import (
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
    EnvironmentMachineEntity,
    EnvironmentStuffEntity,
)

from .entity import ActionT, Camera, Light, Robot, Stuff
from .randomize import (
    Randomizable,
    RandomizableBool,
    RandomizableColor,
    RandomizableFloat,
    RandomizableVec3,
    Vec3,
)

# Sentinal pattern: https://python-patterns.guide/python/sentinel-object/
_CREATE_SENTINAL = object()


class Environment:
    def __init__(
        self,
        default_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
        env_uri_or_prebuilt_id: str | None = None,
    ) -> None:
        self._runtime = obtain_runtime()
        self._world_maker = self._runtime.world_maker

        self.default_engine = default_engine
        self._env = self._world_maker.create_environment(
            default_engine, env_uri_or_prebuilt_id
        )
        self.__frozen = False

    def _freeze(self):
        self.__frozen = True

    def _assert_mutable(self):
        if self.__frozen:
            raise SimulacBaseError(
                "\n".join(
                    [
                        "You are trying to change definition of Environment after Runner creation",
                        "Use runner.get_runtime_object(obj).change_*() to mutate runtime state",
                        "It is not illegal, but we intentionally forbidden such actions",
                    ]
                )
            )

    @overload
    def place_entity(
        self,
        entity: Stuff,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        name: str = "",
        description: str | None = None,
    ) -> StuffObject: ...
    @overload
    def place_entity(
        self,
        entity: Camera,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        name: str = "",
        description: str | None = None,
    ) -> CameraObject: ...
    @overload
    def place_entity(
        self,
        entity: Light,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        name: str = "",
        description: str | None = None,
    ) -> LightObject: ...
    @overload
    def place_entity(
        self,
        entity: Robot[ActionT],
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        name: str = "",
        description: str | None = None,
    ) -> RobotObject[ActionT]: ...
    def place_entity(
        self,
        entity: Stuff | Robot[ActionT] | Camera | Light,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        name: str | None = None,
        description: str | None = None,
    ) -> StuffObject | RobotObject[ActionT] | CameraObject | LightObject:
        description = description or ""

        if isinstance(entity, Stuff):
            env_stuff_obj = self._world_maker.create_stuff_entity(
                name, description, entity.obj_uri_or_prebuilt_name, "", ""
            )
            self._world_maker.add_entity(
                self._env.id, env_stuff_obj, pos=pos, quat=quat
            )
            return StuffObject(env_stuff_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Robot):
            env_robot_obj = self._world_maker.create_machine_entity(
                name, description, entity.obj_uri_or_prebuilt_name
            )
            self._world_maker.add_entity(
                self._env.id, env_robot_obj, pos=pos, quat=quat
            )
            return cast(
                "RobotObject[ActionT]",
                RobotObject(env_robot_obj, _create_sentinal=_CREATE_SENTINAL),
            )
        elif isinstance(entity, Camera):
            env_camera_obj = self._world_maker.create_camera_entity(
                name, description, entity.type
            )
            self._world_maker.add_entity(
                self._env.id, env_camera_obj, pos=pos, quat=quat
            )
            return CameraObject(env_camera_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Light):  # pyright: ignore[reportUnnecessaryIsInstance]
            env_light_obj = self._world_maker.create_light_entity(
                name, description, entity.type
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
        object_or_object_id: StuffObject
        | RobotObject[Any]
        | CameraObject
        | LightObject,
    ) -> None: ...
    @overload
    def remove_object(self, object_or_object_id: str) -> None: ...
    def remove_object(
        self,
        object_or_object_id: StuffObject
        | RobotObject[Any]
        | CameraObject
        | LightObject
        | str,
    ) -> None:
        pass

    def get_object(
        self, object_id: str
    ) -> StuffObject | RobotObject[Any] | CameraObject | LightObject: ...

    def dump_env(self) -> dict:
        """Return definition of environment.
        Return type `dict` is json format

        Raises:
            SimulacBaseError: _description_

        Returns:
            dict: json format environment definition
        """
        ...


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

    def set_mass(self, mass: RandomizableFloat) -> None: ...
    def set_pos(self, pos: RandomizableVec3) -> None: ...
    def set_rot(self, rot: RandomizableVec3) -> None: ...
    def set_size(self, size: RandomizableVec3) -> None: ...
    def set_fixed(self, is_fixed: RandomizableBool) -> None: ...
    def set_friction(self, friction: RandomizableFloat) -> None: ...
    def set_density(self, density: RandomizableFloat) -> None: ...


class RobotObject(Generic[ActionT]):
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

    def set_pos(self, pos: RandomizableVec3) -> None: ...
    def set_rot(self, rot: RandomizableVec3) -> None: ...
    def set_act_pos(self, pos: Randomizable[ActionT]) -> None: ...

    def get_act_min(self) -> ActionT: ...
    def get_act_max(self) -> ActionT: ...


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

    def set_pos(self, pos: RandomizableVec3) -> None: ...
    def set_rot(self, rot: RandomizableVec3) -> None: ...
    def set_lookat(self, lookat: RandomizableVec3) -> None: ...
    def set_fov(self, fov: RandomizableFloat) -> None: ...
    def set_aspect(self, aspect: RandomizableFloat) -> None: ...
    def set_near(self, near: RandomizableFloat) -> None: ...
    def set_far(self, far: RandomizableFloat) -> None: ...

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

    def set_pos(self, pos: RandomizableVec3) -> None: ...
    def set_rot(self, rot: RandomizableVec3) -> None: ...
    def set_intensity(self, intensity: RandomizableFloat) -> None: ...
    def set_type(
        self, type: Literal["ambient", "pointlight", "reactarea", "spot"]
    ) -> None: ...
    def set_color(self, color: RandomizableColor) -> None: ...


class Runner:
    def __init__(
        self,
        env: Environment,
        seed: int | None = 0,
        tick: int | None = 5,  # 5ms
        record_location: str
        | None = None,  # save location of runtime recording data (a.k.a. Lerobot dataset format)
        /,
        *,
        runtime_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
    ):
        self.seed = seed
        self.tick_time = tick

        self._world_maker = obtain_runtime().world_maker

        self._runner = self._world_maker.create_runner(env._env.id)

        env._freeze()

    def step(self, action: List[float]):
        self._runner.step(action)

    def tick(self): ...

    type State = Any

    def reset(self) -> State: ...

    def get_state(self): ...

    @overload
    def get_runtime_object(self, obj: StuffObject) -> StuffRuntime: ...
    @overload
    def get_runtime_object(
        self, obj: RobotObject[ActionT]
    ) -> RobotRuntime[ActionT]: ...
    @overload
    def get_runtime_object(self, obj: LightObject) -> LightRuntime: ...
    @overload
    def get_runtime_object(self, obj: CameraObject) -> CameraRuntime: ...
    def get_runtime_object(
        self, obj: StuffObject | RobotObject[Any] | LightObject | CameraObject
    ) -> StuffRuntime | RobotRuntime[Any] | LightRuntime | CameraRuntime: ...

    def close(self) -> None: ...

    # For context manage
    # e.g., `with Runner(env) as runner:`
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...

    def _debug_render(self):
        return self._runner._debug_render()


class StuffRuntime:
    def __init__(
        self,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

    def change_mass(self, mass: float) -> None: ...
    def change_pos(self, pos: Vec3) -> None: ...
    def change_size(self, size: Vec3) -> None: ...
    def change_fixed(self, is_fixed: bool) -> None: ...
    def change_friction(self, friction: float) -> None: ...
    def change_density(self, density: float) -> None: ...


class RobotRuntime(Generic[ActionT]):
    def __init__(
        self,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

    def step(self, action: ActionT) -> None: ...
    def tick(self) -> None: ...

    def get_pos(self) -> Vec3: ...
    def get_vel(self) -> list[float]: ...


class CameraRuntime:
    def __init__(
        self,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

    def change_pos(self, pos: Vec3) -> None: ...
    def change_rot(self, rot: Vec3) -> None: ...


class LightRuntime:
    def __init__(
        self,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

    def change_pos(self, pos: Vec3) -> None: ...
    def change_rot(self, rot: Vec3) -> None: ...
    def change_intensity(self, intensity: float) -> None: ...
    def change_color(self, color: tuple[int, int, int]) -> None: ...

    # Needed?
    def __change_type(self): ...


# region Will be implemented


class ParallelRunner:
    def __init__(
        self,
        envs: list[Environment],
        seeds: list[int] | None = None,
        tick: list[int] | None = None,
        record_locations: list[str] | None = None,
    ) -> None: ...

    def step(self, actions: list[list[float]]) -> None: ...
    def tick(self) -> None: ...

    type State = Any

    def reset(self, seeds: list[int]) -> list[State]: ...

    def close(self) -> None: ...

    # For context manage
    # e.g., `with Runner(env) as runner:`
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...

    def at(self, idx: int) -> Runner: ...

    def get_state(self) -> object: ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Runner: ...


class __BIV:
    """Brain in a vat? or Agent?"""


# end-region
