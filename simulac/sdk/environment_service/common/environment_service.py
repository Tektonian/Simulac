from __future__ import annotations  # 3.7+ 에서 필요

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Union, overload
from urllib.parse import SplitResult, urlsplit

from simulac.base.error.error import SimulacBaseError
from simulac.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from simulac.base.result.result import ResultType
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
        entity: MJCFPhysicsComponent | URDFPhysicsComponent | USDPhysicsComponent,
    ): ...


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
        entity: MJCFPhysicsComponent | URDFPhysicsComponent | USDPhysicsComponent,
    ):
        env_ret = self.get_environment(env_id)
        if env_ret[0] is None:
            raise SimulacBaseError("No environment found")

        env = env_ret[0]

        env.stuffs.append(entity)


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

    def load_env(self): ...

    def snapshop(self): ...
