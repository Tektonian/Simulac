from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, overload

import mujoco

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.runner_service.common.model.context import INativeContext
from simulac.sdk.runner_service.local.mujoco.binding import (
    MujocoCameraBinding,
    MujocoLightBinding,
    MujocoRobotBinding,
    MujocoStuffBinding,
)

if TYPE_CHECKING:
    from simulac.lib.world_maker.runner import CameraObject as LibCameraType
    from simulac.lib.world_maker.runner import RobotObject as LibRobotType
    from simulac.lib.world_maker.runner import StuffObject as LibStuffType


@dataclass(slots=True)
class MujocoNativeContext(INativeContext):
    model: mujoco.MjModel
    data: mujoco.MjData

    stuff_bindings: dict[str, MujocoStuffBinding]
    machine_bindings: dict[str, MujocoRobotBinding]
    camera_bindings: dict[str, MujocoCameraBinding]
    light_bindings: dict[str, MujocoLightBinding]

    @overload
    def binding(self, entity: LibStuffType) -> MujocoStuffBinding: ...
    @overload
    def binding(self, entity: LibCameraType) -> MujocoCameraBinding: ...
    @overload
    def binding(self, entity: LibRobotType) -> MujocoRobotBinding: ...
    @overload
    def binding(
        self, entity: str
    ) -> MujocoRobotBinding | MujocoStuffBinding | MujocoCameraBinding: ...

    def binding(
        self, entity: str | LibRobotType | LibStuffType | LibCameraType
    ) -> (
        MujocoStuffBinding
        | MujocoRobotBinding
        | MujocoCameraBinding
        | MujocoLightBinding
    ):
        entity_id = self.__entity_id(entity)
        binding = self.stuff_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self.machine_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self.camera_bindings.get(entity_id)
        if binding is not None:
            return binding

        binding = self.light_bindings.get(entity_id)
        if binding is not None:
            return binding

        raise SimulacBaseError(f"No MuJoCo binding for entity {entity_id!r}")

    def __entity_id(self, entity: str | LibRobotType | LibStuffType):
        if isinstance(entity, str):
            return entity
        entity_id = getattr(entity, "id", None)
        if isinstance(entity_id, str):
            return entity_id
        runtime = getattr(entity, "_runtime", None)
        runtime_id = getattr(runtime, "id", None)
        if isinstance(runtime_id, str):
            return runtime_id
        raw_entity = getattr(entity, "_entity", None)
        raw_entity_id = getattr(raw_entity, "id", None)
        if isinstance(raw_entity_id, str):
            return raw_entity_id
        raise SimulacBaseError(f"Cannot resolve entity id from {entity!r}")
