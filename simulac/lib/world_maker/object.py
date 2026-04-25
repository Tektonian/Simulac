from __future__ import annotations

from typing import Any, Generic, List, Literal, Tuple, cast, overload
from typing import Any, Generic, Literal, Tuple, cast, overload

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

    # NOTE: @gangjeuk
    # Should be `place()`?
    @overload
    def place_entity(
        self,
        entity: Stuff,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> StuffObject: ...
    @overload
    def place_entity(
        self,
        entity: Camera,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> CameraObject: ...
    @overload
    def place_entity(
        self,
        entity: Light,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> LightObject: ...
    @overload
    def place_entity(
        self,
        entity: Robot[ActionT],
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> RobotObject[ActionT]: ...
    def place_entity(
        self,
        entity: Stuff | Robot[ActionT] | Camera | Light,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> StuffObject | RobotObject[ActionT] | CameraObject | LightObject:
        description = description or ""

        if isinstance(entity, Stuff):
            if entity_id is None:
                entity_id = f"stuff_{len(self._env.stuffs)}"
            env_stuff_obj = self._world_maker.create_stuff_entity(
                entity_id, description, entity.obj_uri_or_prebuilt_name, "", ""
            )
            self._world_maker.add_entity(
                self._env.id, env_stuff_obj, pos=pos, quat=quat
            )
            return StuffObject(env_stuff_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Robot):
            if entity_id is None:
                entity_id = f"robot_{len(self._env.machines)}"
            env_robot_obj = self._world_maker.create_machine_entity(
                entity_id, description, entity.obj_uri_or_prebuilt_name
            )
            self._world_maker.add_entity(
                self._env.id, env_robot_obj, pos=pos, quat=quat
            )
            return cast(
                "RobotObject[ActionT]",
                RobotObject(env_robot_obj, _create_sentinal=_CREATE_SENTINAL),
            )
        elif isinstance(entity, Camera):
            if entity_id is None:
                entity_id = f"camera_{len(self._env.machines)}"
            env_camera_obj = self._world_maker.create_camera_entity(
                entity_id, description, entity.type
            )
            self._world_maker.add_entity(
                self._env.id, env_camera_obj, pos=pos, quat=quat
            )
            return CameraObject(env_camera_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Light):  # pyright: ignore[reportUnnecessaryIsInstance]
            if entity_id is None:
                entity_id = f"light_{len(self._env.lights)}"
            env_light_obj = self._world_maker.create_light_entity(
                entity_id, description, entity.type
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

    def set_mass(self, mass: RandomizableFloat) -> None:
        # do assertion first
        # self._env._assert_mutate()
        ...

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
