from __future__ import annotations  # 3.7+ 에서 필요

import json
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Union, overload
from urllib.parse import SplitResult, urlsplit

from simulac.base.error.error import SimulacBaseError
from simulac.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from simulac.base.result.result import ResultType

# Begin - Do not change import name @see `EnvironmentManagementService.__dump_type_for_class()``
from simulac.sdk.environment_service.common.model import constraint as constraint_model
from simulac.sdk.environment_service.common.model import entity as entity_model
from simulac.sdk.environment_service.common.model import ref as ref_model

# End
from simulac.sdk.environment_service.common.model.constraint import SceneConstraint
from simulac.sdk.environment_service.common.model.entity import (
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
    EnvironmentMachineEntity,
    EnvironmentStuffEntity,
)
from simulac.sdk.log_service.common.log_service import ILogService
from simulac.sdk.world_service.common.world_service import IWorldManagementService

from .environment import IEnvironment


@service_identifier("IEnvironmentManagementService")
class IEnvironmentManagementService(ServiceIdentifier["IEnvironmentManagementService"]):
    _ID_PREFIX = "env_"

    @property
    @abstractmethod
    def _environments(self) -> Mapping[str, IEnvironment]:
        pass

    @abstractmethod
    def get_environment(
        self, environment_id: str
    ) -> ResultType[IEnvironment, BaseException]:
        pass

    @abstractmethod
    def create_environment(
        self, engine: Literal["mujoco", "newton", "genesis"] = "mujoco"
    ) -> ResultType[IEnvironment, BaseException]: ...

    # FIXME: For testing remove it
    @abstractmethod
    def add_entity(
        self,
        env_id: str,
        entity: EnvironmentStuffEntity
        | EnvironmentMachineEntity
        | EnvironmentCameraEntity
        | EnvironmentLightEntity,
        entity_id: str | None = None,
        pos: Any = (0, 0, 0),
        rot: Any = (0, 0, 0),
        *,
        fixed: bool | None = None,
    ): ...

    @abstractmethod
    def load_env(self, path: Path) -> IEnvironment: ...

    @abstractmethod
    def dump_env(
        self,
        env_id: str,
        *,
        include_resolved_assets: bool = False,
        include_runtime_state: bool = False,
        validation: Literal["none", "warn", "raise"] = "warn",
    ) -> dict[str, Any]: ...


class EnvironmentManagementService(IEnvironmentManagementService):
    def __init__(
        self, LogService: ILogService, WorldManagementService: IWorldManagementService
    ) -> None:
        self.LogService = LogService
        self.WorldManagementService = WorldManagementService

        self.environments: dict[str, IEnvironment] = {}

    @property
    def _environments(self) -> Mapping[str, IEnvironment]:
        return self.environments

    def get_environment(self, environment_id: str):
        env = self.environments.get(environment_id)
        if env is None:
            return (None, SimulacBaseError("no environment found"))
        return (env, None)

    def create_environment(
        self, engine: Literal["mujoco", "newton", "genesis"] = "mujoco"
    ) -> ResultType[IEnvironment, BaseException]:
        env_id = f"{self._ID_PREFIX}{len(self._environments)}"

        world_ret = self.WorldManagementService.create_world(None)

        if world_ret[1] is not None:
            return (None, world_ret[1])

        env = Environment(
            id=env_id,
            world_id=world_ret[0].id,
            default_engine=engine,
        )
        self.environments[env_id] = env
        self.LogService.debug(f"Environment created {env.id}")
        return (env, None)

    # FIXME: for testing remove it
    def add_entity(
        self,
        env_id: str,
        entity: EnvironmentStuffEntity
        | EnvironmentMachineEntity
        | EnvironmentCameraEntity
        | EnvironmentLightEntity,
        entity_id: str | None = None,
        pos: Any = (0, 0, 0),
        rot: Any = (0, 0, 0),
        *,
        fixed: bool | None = None,
    ):
        env_ret = self.get_environment(env_id)
        if env_ret[0] is None:
            raise SimulacBaseError("No environment found")

        env = env_ret[0]
        entity.id = entity_id or entity.id
        entity.pos = pos
        entity.rot = rot

        if fixed is not None:
            if not isinstance(entity, EnvironmentStuffEntity):
                raise SimulacBaseError("fixed is only supported for Stuff entities")
            entity.fixed = fixed

        if isinstance(entity, EnvironmentStuffEntity):
            env.stuffs.append(entity)
        elif isinstance(entity, EnvironmentMachineEntity):
            env.machines.append(entity)
        elif isinstance(entity, EnvironmentCameraEntity):
            env.cameras.append(entity)
        elif isinstance(entity, EnvironmentLightEntity):
            env.lights.append(entity)
        else:
            raise SimulacBaseError(f"Unsupported environment entity: {entity!r}")

    def load_env(self, path: Path) -> IEnvironment:
        definition = self.validate_env(path)
        default_engine = definition.get("physics_engine", "mujoco")

        env_ret = self.create_environment(default_engine)

        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]
        env.env_json_uri = str(path)
        self.hydrate_environment(env, definition)
        return env

    def validate_env(self, path: Path) -> dict[str, Any]:
        try:
            definition = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SimulacBaseError(f"Failed to read environment dump: {path}") from exc

        if not isinstance(definition, dict):
            raise SimulacBaseError("Environment dump must be a JSON object")

        if definition.get("schema_version") != "simulac.env.v1":
            raise SimulacBaseError(
                f"Unsupported environment schema: {definition.get('schema_version')}"
            )

        return definition

    def hydrate_environment(
        self,
        env: IEnvironment,
        definition: dict[str, Any],
    ) -> None:
        env.physics_engine = definition.get("physics_engine", env.physics_engine)

        for item in definition.get("stuffs", []):
            self.__hydrate_stuff(env, item)

        for item in definition.get("machines", []):
            self.__hydrate_machine(env, item)

        for item in definition.get("cameras", []):
            self.__hydrate_camera(env, item)

        for item in definition.get("lights", []):
            self.__hydrate_light(env, item)

        env.relations = [
            self.__load_typed_value(item) for item in definition.get("relations", [])
        ]
        env.constraints = [
            self.__load_typed_value(item) for item in definition.get("constraints", [])
        ]

    def __hydrate_stuff(self, env: IEnvironment, item: dict[str, Any]) -> None:
        asset_uri = item.get("asset_uri")
        entity = EnvironmentStuffEntity(
            description=item.get("description", ""),
            asset_uri=asset_uri,
            original_asset_uri=asset_uri,
            size=self.__load_typed_value(item.get("size", (1, 1, 1))),
            mass=self.__load_typed_value(item.get("mass")),
            friction=self.__load_typed_value(item.get("friction")),
        )

        self.add_entity(
            env.id,
            entity,
            item.get("entity_id"),
            pos=self.__load_typed_value(item.get("pos", (0, 0, 0))),
            rot=self.__load_typed_value(item.get("rot", (0, 0, 0))),
            fixed=item.get("fixed"),
        )

    def __hydrate_machine(self, env: IEnvironment, item: dict[str, Any]) -> None:
        asset_uri = item.get("asset_uri")
        entity = EnvironmentMachineEntity(
            description=item.get("description", ""),
            asset_uri=asset_uri,
            original_asset_uri=asset_uri,
            init_position=self.__load_typed_value(item.get("init_position")),
            action_max=self.__load_typed_value(item.get("action_max")),
            action_min=self.__load_typed_value(item.get("action_min")),
        )

        self.add_entity(
            env.id,
            entity,
            item.get("entity_id"),
            pos=self.__load_typed_value(item.get("pos", (0, 0, 0))),
            rot=self.__load_typed_value(item.get("rot", (0, 0, 0))),
        )

    def __hydrate_camera(self, env: IEnvironment, item: dict[str, Any]) -> None:
        spec = self.__load_typed_value(item.get("spec", {}))
        if isinstance(spec, dict):
            spec = entity_model.CameraSpec(**spec)

        entity = EnvironmentCameraEntity(
            description=item.get("description", ""),
            spec=spec,
            attach=self.__load_typed_value(item.get("attach")),
            look_at=self.__load_typed_value(item.get("look_at")),
            track=self.__load_typed_value(item.get("track")),
        )

        self.add_entity(
            env.id,
            entity,
            item.get("entity_id"),
            pos=self.__load_typed_value(item.get("pos", (0, 0, 0))),
            rot=self.__load_typed_value(item.get("rot", (0, 0, 0))),
        )

    def __hydrate_light(self, env: IEnvironment, item: dict[str, Any]) -> None:
        entity = EnvironmentLightEntity(
            description=item.get("description", ""),
            spec=self.__load_typed_value(item.get("spec", {})),
            attach=self.__load_typed_value(item.get("attach")),
            look_at=self.__load_typed_value(item.get("look_at")),
            track=self.__load_typed_value(item.get("track")),
        )

        self.add_entity(
            env.id,
            entity,
            item.get("entity_id"),
            pos=self.__load_typed_value(item.get("pos", (0, 0, 0))),
            rot=self.__load_typed_value(item.get("rot", (0, 0, 0))),
        )

    def __dump_type_for_class(self, cls: type[Any]) -> str:
        module = cls.__module__.rsplit(".", 1)[-1]
        name = cls.__name__
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        # e.g.,
        # ref_model.EntityRef -> ref.entity_ref
        # entity_model.CameraSpec -> entity.camera_spec
        # constraint_model.EntityTarget -> constraint.entity_target

        return f"{module}.{snake}"

    def __dump_type(self, value: Any) -> str:
        return self.__dump_type_for_class(type(value))

    def __dump_type_registry(self) -> dict[str, type[Any]]:
        classes = [
            ref_model.EntityRef,
            ref_model.EntityPosRef,
            ref_model.EntityRotRef,
            ref_model.EntityQuatRef,
            ref_model.BodyRef,
            ref_model.BodyPosRef,
            ref_model.ColliderRef,
            ref_model.ColliderCenterRef,
            ref_model.ColliderPosRef,
            ref_model.ColliderBoundsRef,
            ref_model.BoundsCenterRef,
            ref_model.BoundsMinRef,
            ref_model.BoundsMaxRef,
            ref_model.BoundsSizeRef,
            ref_model.SurfaceRef,
            ref_model.SurfaceCenterRef,
            ref_model.SurfaceSampleRef,
            ref_model.SurfaceNormalRef,
            ref_model.SupportPointRef,
            ref_model.JointRef,
            ref_model.JointPosRef,
            ref_model.JointVelRef,
            ref_model.JointAxisRef,
            ref_model.JointRangeRef,
            ref_model.AnchorRef,
            ref_model.AnchorPosRef,
            ref_model.CameraRef,
            ref_model.CameraOutputRef,
            ref_model.CameraPosRef,
            ref_model.LightRef,
            ref_model.LightPosRef,
            ref_model.WorldPointRef,
            ref_model.SetEntityPosOp,
            ref_model.SetEntityRotOp,
            ref_model.SetEntityQuatOp,
            ref_model.SetEntitySizeOp,
            ref_model.SetEntityFixedOp,
            ref_model.SetEntityMassOp,
            ref_model.SetEntityFrictionOp,
            ref_model.SetColliderSizeOp,
            ref_model.SetColliderFrictionOp,
            ref_model.SetJointPosOp,
            ref_model.SetJointVelOp,
            ref_model.SetJointCtrlOp,
            ref_model.SetJointFrictionOp,
            ref_model.SetJointDampingOp,
            ref_model.SetActPosOp,
            ref_model.PlaceOp,
            ref_model.AttachOp,
            ref_model.LookAtOp,
            ref_model.FollowOp,
            ref_model.SetCameraPosOp,
            ref_model.SetCameraRotOp,
            ref_model.SetCameraTypeOp,
            ref_model.SetCameraFovOp,
            ref_model.SetCameraAspectOp,
            ref_model.SetCameraNearOp,
            ref_model.SetCameraFarOp,
            ref_model.SetLightPosOp,
            ref_model.SetLightRotOp,
            ref_model.SetLightIntensityOp,
            ref_model.SetLightColorOp,
            ref_model.SetLightAngleOp,
            ref_model.SetLightAreaSizeOp,
            ref_model.SetLightRangeOp,
            ref_model.SetLightDecayOp,
            ref_model.SetLightPenumbraOp,
            entity_model.CameraSpec,
            entity_model.AmbientLightSpec,
            entity_model.PointLightSpec,
            entity_model.SpotLightSpec,
            entity_model.AreaLightSpec,
            entity_model.AttachSpec,
            entity_model.LookAtSpec,
            entity_model.TrackSpec,
            entity_model.TransformSpec,
            constraint_model.EntityTarget,
            constraint_model.RefTarget,
            constraint_model.DistanceConstraint,
            constraint_model.BBoxConstraint,
            constraint_model.NonpenetrationConstraint,
        ]
        return {self.__dump_type_for_class(cls): cls for cls in classes}

    def __load_typed_value(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.__load_typed_value(v) for v in value]

        if not isinstance(value, dict):
            return value

        value_type = value.get("type")
        if value_type is None:
            return {k: self.__load_typed_value(v) for k, v in value.items()}

        cls = self.__dump_type_registry().get(value_type)
        if cls is None:
            return {k: self.__load_typed_value(v) for k, v in value.items()}

        return self.__load_dataclass(cls, value)

    def __load_dataclass(self, cls: type[Any], item: dict[str, Any]) -> Any:
        kwargs = {
            field.name: self.__load_typed_value(item[field.name])
            for field in fields(cls)
            if field.name in item
        }
        return cls(**kwargs)

    def __jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            result = {"type": self.__dump_type(value)}
            for field in fields(value):
                result[field.name] = self.__jsonable(getattr(value, field.name))
            return result
        if isinstance(value, list):
            return [self.__jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [self.__jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: self.__jsonable(v) for k, v in value.items()}
        if isinstance(value, Path):
            return str(value)
        return value

    def __dump_stuff(
        self,
        entity: EnvironmentStuffEntity,
        *,
        include_resolved_assets: bool,
    ) -> dict[str, Any]:
        return {
            "entity_id": entity.id,
            "description": entity.description,
            "asset_uri": entity.asset_uri,
            "pos": self.__jsonable(entity.pos),
            "rot": self.__jsonable(entity.rot),
            "size": self.__jsonable(entity.size),
            "fixed": entity.fixed,
            "mass": self.__jsonable(entity.mass),
            "friction": self.__jsonable(entity.friction),
        }

    def __dump_machine(
        self,
        entity: EnvironmentMachineEntity,
        *,
        include_resolved_assets: bool,
    ) -> dict[str, Any]:
        return {
            # "type": "machine",
            "entity_id": entity.id,
            "description": entity.description,
            "asset_uri": entity.asset_uri,
            "pos": self.__jsonable(entity.pos),
            "rot": self.__jsonable(entity.rot),
            "init_position": self.__jsonable(entity.init_position),
            "action_max": self.__jsonable(entity.action_max),
            "action_min": self.__jsonable(entity.action_min),
        }

    def __validate_dump(
        self,
        definition: dict[str, Any],
        validation: Literal["none", "warn", "raise"],
    ) -> None:
        if validation == "none":
            return

        entity_ids = {
            entity.get("entity_id")
            for group in ("stuffs", "machines", "cameras", "lights")
            for entity in definition[group]
            if entity.get("entity_id") is not None
        }

        errors: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                ref_entity_id = value.get("entity_id")
                if ref_entity_id is not None and ref_entity_id not in entity_ids:
                    errors.append(f"Unknown referenced entity_id: {ref_entity_id}")
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(definition)

        if not errors:
            return

        message = "Invalid environment dump:\n" + "\n".join(f"- {e}" for e in errors)
        if validation == "raise":
            raise SimulacBaseError(message)
        warnings.warn(message, stacklevel=2)

    def dump_env(
        self,
        env_id: str,
        *,
        include_resolved_assets: bool = False,
        include_runtime_state: bool = False,
        validation: Literal["none", "warn", "raise"] = "warn",
    ) -> dict[str, Any]:
        env_ret = self.get_environment(env_id)
        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]
        definition: dict[str, Any] = {
            "schema_version": "simulac.env.v1",
            "id": env.id,
            "world_id": env.world_id,
            "physics_engine": env.physics_engine,
            "stuffs": [
                self.__dump_stuff(e, include_resolved_assets=include_resolved_assets)
                for e in env.stuffs
            ],
            "machines": [
                self.__dump_machine(e, include_resolved_assets=include_resolved_assets)
                for e in env.machines
            ],
            "cameras": [self.__jsonable(e) for e in env.cameras],
            "lights": [self.__jsonable(e) for e in env.lights],
            "relations": [self.__jsonable(e) for e in env.relations],
            "constraints": [self.__jsonable(e) for e in env.constraints],
        }

        if include_runtime_state:
            definition["runtime_state"] = {}

        self.__validate_dump(definition, validation)
        return definition

    def dump_env_json(
        self,
        env_id: str,
        *,
        indent: int = 2,
        include_resolved_assets: bool = False,
        include_runtime_state: bool = False,
        validation: Literal["none", "warn", "raise"] = "warn",
    ) -> str:
        return json.dumps(
            self.dump_env(
                env_id,
                include_resolved_assets=include_resolved_assets,
                include_runtime_state=include_runtime_state,
                validation=validation,
            ),
            indent=indent,
            ensure_ascii=False,
        )


class Environment(IEnvironment):
    def __init__(
        self,
        id: str,
        world_id: str,
        default_engine: Literal["mujoco", "newton", "genesis"],
    ) -> None:
        self.id = id
        self.world_id = world_id

        self.env_json_uri = ""

        self.physics_engine = default_engine

        self.stuffs = []
        self.cameras = []
        self.lights = []
        self.machines = []
        self.relations = []
        self.constraints: list[SceneConstraint] = []

    def load_env(self): ...

    def snapshop(self): ...
