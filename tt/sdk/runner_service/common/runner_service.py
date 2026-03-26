from __future__ import annotations  # 3.7+ 에서 필요

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Mapping, MutableMapping, Tuple, Type
from urllib.parse import urlsplit

from tt.base.error.error import TektonianBaseError
from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from tt.base.result.result import ResultType
from tt.sdk.environment_service.common.environment_service import (
    IEnvironmentManagementService,
)
from tt.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterFactory,
)
from tt.sdk.runner_service.remote.runner import RemoteRunner

if TYPE_CHECKING:
    from tt.sdk.log_service.common.log_service import ILogService
    from tt.sdk.runner_service.common.physics_engine_adapter import (
        IPhysicsEngineAdapter,
    )

    from .runner import IRunner


@service_identifier("IRunnerManagementService")
class IRunnerManagementService(ServiceIdentifier["IRunnerManagementService"]):
    _ID_PREFIX = "run_"

    @property
    @abstractmethod
    def _runners(self) -> List[IRunner]:
        pass

    @abstractmethod
    def register_physics_adapter_factory(
        self,
        adapter_types: List[str],
        adapter_factory: Type[IPhysicsEngineAdapterFactory],
    ) -> None: ...

    @abstractmethod
    def get_runner(self, runner_id: str) -> ResultType[IRunner, BaseException]:
        pass

    @abstractmethod
    def remove_runner(self, runner_id: str) -> None: ...

    @abstractmethod
    def create_runner(self, env_id: str) -> ResultType[IRunner, BaseException]:
        pass

    @abstractmethod
    def step_runner(self, runner_id: str) -> object:
        pass


class RunnerManagementService(IRunnerManagementService):
    def __init__(
        self,
        EnvironmentManagementService: IEnvironmentManagementService,
        LogService: ILogService,
    ) -> None:
        self.EnvironmentManagementService = EnvironmentManagementService
        self.LogService = LogService
        self.runners: List[IRunner] = []
        self.physics_adapter: MutableMapping[
            str, IPhysicsEngineAdapter
        ] = {}  # env_id: adapter
        self.physics_adapter_factory: MutableMapping[  # engine: adapter_factory // e.g., {"mujoco": MujocoAdapter}
            str, Type[IPhysicsEngineAdapterFactory]
        ] = {}

    @property
    def _runners(self) -> List[IRunner]:
        return self.runners

    def register_physics_adapter_factory(
        self,
        adapter_types: List[str],
        adapter_factory: Type[IPhysicsEngineAdapterFactory],
    ) -> None:
        for adapter_type in adapter_types:
            if adapter_type in self.physics_adapter_factory:
                self.LogService.warn(
                    f'Replace physics adapter type of "{adapter_type}" {self.physics_adapter_factory[adapter_type]} into {adapter_factory}'
                )
            self.physics_adapter_factory[adapter_type] = adapter_factory

    def get_runner(self, runner_id: str):
        for run in self.runners:
            if run.id == runner_id:
                return (run, None)
        return (None, TektonianBaseError("no runner found"))

    def remove_runner(self, runner_id: str) -> None:
        for run in self.runners:
            if run.id == runner_id:
                self.runners.remove(run)

    def create_runner(self, env_id: str):
        # check adapter exist
        adapter = self.physics_adapter.get(env_id)
        if adapter is not None:
            runner = adapter.create_runner()
            self.runners.append(runner)
            return (runner, None)

        env_ret = self.EnvironmentManagementService.get_environment(env_id)

        if env_ret[0] is None:
            return (None, env_ret[1])

        env = env_ret[0]

        # region TODO: move initialization code -> use IPhysicsEngineAdapterFactory
        env_json_uri = (
            urlsplit(env.env_json_uri)
            if isinstance(env.env_json_uri, str)
            else env.env_json_uri
        )

        if env_json_uri.scheme in ["http", "https"]:
            pass

        if env.physics_engine == "mujoco" or env.physics_engine == "newton":
            factory = self.physics_adapter_factory.get(env.physics_engine)
            if factory is None:
                return (None, TektonianBaseError(f"No proper adaptor for mujoco"))

            adapter = factory.create_physics_engine_adapter(env.id)
            self.physics_adapter[env.id] = adapter
            runner = adapter.create_runner()
            runner.initialize()
            return (runner, None)

        else:
            return (None, TektonianBaseError(f"No proper adaptor for {env}"))
        # end-region

        if env_json_uri.scheme in ["http", "https"]:
            runner_id = f"{self._ID_PREFIX}{len(self.runners)}"
            # api routing example
            # http://0.0.0.0:3000/api/.../Tektonian/Lebero/env.json
            print(env_json_uri.path.split("/"))
            [owner, remote_env_id] = env_json_uri.path.split("/")[-3:-1]

            runner = RemoteRunner(
                runner_id,
                env_id,
                {},
                owner,
                remote_env_id,
                kwargs=__remote_runner_kwargs,
            )
            self.runners.append(runner)
            return (runner, None)
        else:
            return (
                None,
                TektonianBaseError("Local runner class is not implemented yet"),
            )
