from __future__ import annotations
from typing import TYPE_CHECKING, Any, MutableMapping, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import json

from websockets.sync.client import connect, ClientConnection
import json_numpy

from tt.base.error.error import TektonianBaseError
from tt.sdk.environment_service.common.environment_service import (
    IEnvironmentManagementService,
)
from tt.sdk.runner_service.common.runner import IRunner
from tt.sdk.runner_service.common.runner_service import IRunnerManagementService

if TYPE_CHECKING:
    from tt.base.instantiate.instantiate import IInstantiateService

type GymEnvStepReturnType = Tuple[
    dict[str, Any],  # obs
    dict[str, Any],  # reward
    bool,  # termination
    bool,  # truncation
    dict[str, Any],  # info
]


class BenchmarkEnvironment:
    def __init__(
        self,
        runner_id: str,
        benchmark_id: str,
        remote_env_id: str,
        seed: int,
        benchmark_specific_kwargs: dict[str, Any],
    ):
        self.runner_id = runner_id
        self.benchmark_id = benchmark_id
        self.remote_env_id = remote_env_id
        self.benchmark_specific_kwargs = benchmark_specific_kwargs

        self._socket: ClientConnection | None = None

    def _connect(self):

        if self._socket:
            return

        self._socket = connect(
            f"ws://localhost:3000/api/container/{self.benchmark_id}/{self.remote_env_id}?"
        )
        msg = json.dumps(
            {"command": "build_env", "args": self.benchmark_specific_kwargs}
        )
        self._socket.send(msg)
        recv = json_numpy.loads(self._socket.recv())
        print("connect")

    def step(self, action: list[float]) -> GymEnvStepReturnType:

        if self._socket is None:
            self._connect()

        self._socket.send(json.dumps({"command": "step", "args": {"action": action}}))

        recv = json_numpy.loads(self._socket.recv())
        print("recv")
        return recv

    def reset(self):

        if self._socket is None:
            self._connect()

        self._socket.send(json.dumps({"command": "reset", "args": {}}))
        recv = json_numpy.loads(self._socket.recv())
        return recv

    def close(self):
        if self._socket:
            self._socket.close()

    @property
    def action_space(self): ...
    @property
    def observation_space(self): ...


class BenchmarkVecEnvironment:
    def __init__(self, benchmark_envs: list[BenchmarkEnvironment]) -> None:
        self.benchmark_envs = benchmark_envs

    def step(self, actions: list[list[float]]) -> list[GymEnvStepReturnType]:
        """Send step to all envs concurrently and gather responses.
        Results are returned in the same order as the provided `envs` list.
        """

        if not self.benchmark_envs:
            return []

        for env in self.benchmark_envs:
            if env._socket is None:
                env._connect()

        # Phase 1: send in parallel
        def _send_payload(r: BenchmarkEnvironment, action: list[float]) -> None:
            payload: dict[str, Any] = {"command": "step", "args": {"action": action}}
            print("thread send")
            r._socket.send(json.dumps(payload))  # type: ignore[union-attr]

        with ThreadPoolExecutor(max_workers=len(self.benchmark_envs)) as ex:
            send_futs = [
                ex.submit(_send_payload, r[0], r[1])
                for r in zip(self.benchmark_envs, actions)
            ]
            # Ensure all sends complete (propagate any exceptions)
            for f in as_completed(send_futs):
                f.result()

            # Phase 2: recv in parallel, maintain order
            index_map = {ex.submit(lambda rr=r: rr._socket.recv()): i for i, r in enumerate(self.benchmark_envs)}  # type: ignore[union-attr]
            results: list[Any] = [None] * len(self.benchmark_envs)
            for f in as_completed(index_map):
                idx = index_map[f]
                results[idx] = f.result()

        return results

    def reset_parallel(self): ...
