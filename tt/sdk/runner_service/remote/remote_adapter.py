from __future__ import annotations

import json
import urllib.parse
from typing import TYPE_CHECKING, Any, Callable, MutableMapping

from websockets import connect

from tt.base.error.error import TektonianBaseError
from tt.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterState,
)
from tt.sdk.runner_service.common.runner import IRunner
from tt.sdk.runner_service.common.runner_service import IRunnerManagementService

if TYPE_CHECKING:
    from tt.sdk.environment_service.common.environment_service import (
        IEnvironmentManagementService,
    )
    from tt.sdk.log_service.common.log_service import ILogService


class RemoteRunner(IRunner):
    def __init__(
        self,
        runner_id: str,
        env_id: str,
        state: object,
        owner: str,
        remote_env_id: str,
        env_specific_kwars: dict[str, Any],
        on_after_call_step: Callable[[str], None],
    ) -> None:
        self.runner_type = "remote"
        self.runner_id = runner_id
        self.env_id = env_id

        self.state = state
        self.kwargs = env_specific_kwars

        self._socket = connect(
            f"ws://localhost:3000/api/container/{owner}/{remote_env_id}?{env_specific_kwars}"
        )

        self.on_after_call_step = on_after_call_step

        msg = json.dumps({"command": "build_env", "args": self.kwargs})
        print(msg)
        self._socket.send(msg)
        recv = self._socket.recv()
        # print(recv)

    def initialize(self) -> None: ...

    def step(self, action: list[float]) -> None:
        self._socket.send(json.dumps({"command": "step", "args": {"action": action}}))

        recv = self._socket.recv()
        self.on_after_call_step(self.runner_id)
        return recv

    def tick(self) -> None: ...

    def get_state(self) -> None: ...
    def render(self) -> None: ...
    def reset(self) -> None: ...
    def set_state(self) -> None: ...
    def clone_state(self) -> None: ...


class RemoteAdapter(IPhysicsEngineAdapter):
    def __init__(
        self,
        env_id: str,
        LogService: ILogService,
        RunnerManagementService: IRunnerManagementService,
        EnvironmentManagementService: IEnvironmentManagementService,
    ) -> None:
        self.LogService = LogService
        self.RunnerManagementService = RunnerManagementService
        self.EnvironmentManagementService = EnvironmentManagementService

        self._runner_count = 0
        self._step_count = 0
        self._step_count_map: MutableMapping[str, int] = dict()

        env_ret = self.EnvironmentManagementService.get_environment(env_id)

        if env_ret[0] is None:
            raise TektonianBaseError(f"No environment {env_id}")

        env = env_ret[0]
        self.env = env

    def create_runner(self) -> IRunner:
        if self._step_count != 0:
            raise TektonianBaseError(
                "Cannot create new runner after calling step() function"
            )

        env_uri = self.env.env_json_uri

        if isinstance(env_uri, str):
            env_uri = urllib.parse.urlsplit(env_uri)

        [owner, remote_env_id] = env_uri.path.split("/")[-3:-1]

        def on_after_runner_step(runner_id: str):
            self._step_count_map[runner_id] += 1
            self._step_count += 1

        new_runner_id = f"run_{self._runner_count}"
        runner = RemoteRunner(
            new_runner_id,
            self.env.id,
            {},
            owner,
            remote_env_id,
            self.env.benchmark_specific_args,
            on_after_call_step=on_after_runner_step,
        )

        self.LogService.debug(
            f"new remote runner created env_id: {self.env.id} runner_id: {new_runner_id}"
        )
        self._step_count_map[new_runner_id] = 0
        self._runner_count += 1

        return runner

    def get_state(self) -> IPhysicsEngineAdapterState:
        return IPhysicsEngineAdapterState(
            self.env.id, self._runner_count, self._step_count_map
        )
