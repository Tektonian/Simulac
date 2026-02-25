from __future__ import annotations  # 3.7+ 에서 필요
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlsplit, SplitResult

from tt.base.error.error import TektonianBaseError
from tt.base.instantiate.instantiate import service_identifier, ServiceIdentifier
from tt.base.result.result import ResultType
from tt.sdk.world_service.common.world_service import IWorldManagementService

from tt.sdk.environment_service.remote.environment import RemoteEnvironment

if TYPE_CHECKING:
    from .environment import IEnvironment


@service_identifier("IEnvironmentManagementService")
class IEnvironmentManagementService(ServiceIdentifier):

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
        self, env_json_uri: str, act_json_url: str, obs_json_url: str, seed: int
    ) -> ResultType[IEnvironment, BaseException]:
        pass


class EnvironmentManagementService(IEnvironmentManagementService):
    def __init__(self, WorldManagementService: IWorldManagementService) -> None:
        self.WorldManagementService = WorldManagementService

        self.environments: dict[str, IEnvironment] = {}

    @property
    def _environments(self) -> Mapping[str, IEnvironment]:
        return self.environments

    def get_environment(self, environment_id: str):
        env = self.environments.get(environment_id)
        if env is None:
            return (None, TektonianBaseError("no environment found"))
        return (env, None)

    def create_environment(
        self, env_json_uri: str, act_json_url: str, obs_json_url: str, seed: int
    ):
        url = urlsplit(env_json_uri)

        env_id = f"{self._ID_PREFIX}{len(self._environments)}"

        world_ret = self.WorldManagementService.create_world(None)

        if world_ret[1] is not None:
            return (None, world_ret[1])

        if url.scheme in ["http", "https"]:
            env = RemoteEnvironment(
                id=env_id,
                world_id=world_ret[0].id,
                env_json_uri=env_json_uri,
                act_json_uri=act_json_url,
                obs_json_uri=obs_json_url,
            )
            # Auto-register the created environment
            self.environments[env_id] = env
            return (env, None)
        else:
            raise TektonianBaseError("Local environment class is not implemented yet")
