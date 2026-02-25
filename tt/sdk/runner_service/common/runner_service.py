from __future__ import annotations  # 3.7+ 에서 필요
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from urllib.parse import urlsplit

from tt.base.error.error import TektonianBaseError
from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from tt.base.result.result import ResultType
from tt.sdk.environment_service.common.environment_service import (
    IEnvironmentManagementService,
)
from tt.sdk.runner_service.remote.runner import RemoteRunner

if TYPE_CHECKING:
    from .runner import IRunner


@service_identifier("IRunnerManagementService")
class IRunnerManagementService(ServiceIdentifier):

    _ID_PREFIX = "run_"

    @property
    @abstractmethod
    def _runners(self) -> List[IRunner]:
        pass

    @abstractmethod
    def get_runner(self, runner_id: str) -> ResultType[IRunner, BaseException]:
        pass

    @abstractmethod
    def create_runner(self, env_id: str) -> ResultType[str, BaseException]:
        pass

    @abstractmethod
    def step_runner(self, runner_id: str) -> object:
        pass


class RunnerManagementService(IRunnerManagementService):
    def __init__(
        self, EnvironmentManagementService: IEnvironmentManagementService
    ) -> None:
        self.EnvironmentManagementService = EnvironmentManagementService
        self.runners: List[IRunner] = []

    @property
    def _runners(self) -> List[IRunner]:
        return self.runners

    def get_runner(self, runner_id: str):
        for run in self.runners:
            if run.id == runner_id:
                return (run, None)
        return (None, TektonianBaseError("no runner found"))

    def create_runner(self, env_id: str):
        ret = self.EnvironmentManagementService.get_environment(env_id)

        if ret[0] is None:
            return (None, ret[1])

        env = ret[0]

        env_json_uri = (
            urlsplit(env.env_json_uri)
            if isinstance(env.env_json_uri, str)
            else env.env_json_uri
        )

        if env_json_uri.scheme in ["http", "https"]:
            runner_id = f"{self._ID_PREFIX}{len(self.runners) + 1}"
            runner = RemoteRunner(runner_id, env_id, {})
            self.runners.append(runner)
            return (runner_id, None)
        else:
            return (
                None,
                TektonianBaseError("Local runner class is not implemented yet"),
            )
