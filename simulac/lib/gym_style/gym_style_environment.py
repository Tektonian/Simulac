from __future__ import annotations

import json
import os
import urllib
import urllib.parse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Tuple, Type, TypeAlias

import msgpack
import zstd
from websockets import InvalidHandshake
from websockets.sync.client import ClientConnection, connect

from simulac.base.error.error import SimulacBaseError
from simulac.sdk import obtain_runtime

GymEnvStepReturnType: TypeAlias = Type[
    Tuple[
        dict[str, Any],  # obs
        dict[str, Any],  # reward
        bool,  # termination
        bool,  # truncation
        dict[str, Any],  # info
    ]
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

        self._runtime = obtain_runtime()

        self.runner_id = runner_id
        self.benchmark_id = benchmark_id
        self.remote_env_id = remote_env_id
        self.benchmark_specific_kwargs = benchmark_specific_kwargs
        self.init_seed = seed

        self._socket: ClientConnection | None = None

    def _get_api_key(self) -> str:
        api_key = self._runtime.environment_variable.token
        if api_key is None:
            _API_KEY_ERROR_MESSAGE = "\n".join(
                [
                    "API key for the remote benchmark service was not found.",
                    "Get a token at https://tektonian.com/settings/token.",
                    "Then run `simulac login` in your terminal and paste the API key.",
                ]
            )
            self._runtime.logger.error(_API_KEY_ERROR_MESSAGE)
            raise SimulacBaseError(_API_KEY_ERROR_MESSAGE)
        return api_key

    def _build_ws_url(self) -> str:
        query_param = urllib.parse.urlencode(
            {"api_key": self._get_api_key(), "benchmark_id": self.benchmark_id}
        )
        base_url = urllib.parse.urlparse(self._runtime.environment_variable.base_url)
        ws_scheme = {"http": "ws", "https": "wss"}.get(base_url.scheme, "wss")
        return base_url._replace(
            scheme=ws_scheme,
            path=os.path.join(
                base_url.path,
                f"container/{self.benchmark_id}",
            ),
            query=query_param,
        ).geturl()

    def _ensure_connected(self) -> ClientConnection:
        """Try socket connection and return socket.
        Once connection made, socket requests to build env with `build_env` command.
        In most code pattern, this function is used for lazy connection

        Raises:
            SimulacBaseError: Socket connection failed

        Returns:
            ClientConnection: client socket
        """
        # Timeout seconds and retry count are set explicitly
        # Because of cold-start of remote container
        MAX_CONNECT_RETRIES = 3
        OPEN_TIMEOUT = 10

        if self._socket is not None:
            return self._socket

        url = self._build_ws_url()

        for attempt in range(1, MAX_CONNECT_RETRIES + 1):
            try:
                self._socket = connect(
                    url,
                    open_timeout=OPEN_TIMEOUT,
                    ping_interval=5,
                    ping_timeout=10,
                )
                break
            except (TimeoutError, InvalidHandshake) as err:
                if attempt == MAX_CONNECT_RETRIES:
                    self._runtime.telemetry.public_error(
                        event_name="simulac_connection_failed",
                        data={
                            "err": err.args,
                            "stacktrace": err.__traceback__,
                        },
                    )
                    raise SimulacBaseError(
                        "Unable to connect to the remote benchmark service. "
                        f"benchmark_id={self.benchmark_id!r}, attempts={attempt}."
                    ) from err

                self._runtime.logger.warn(
                    "Connection to the remote benchmark service failed. "
                    f"benchmark_id={self.benchmark_id!r}, "
                    f"attempt={attempt}/{MAX_CONNECT_RETRIES}, "
                    f"Retrying in {OPEN_TIMEOUT:.1f}s."
                )

        if self._socket is None:
            raise SimulacBaseError("Benchmark environment is not connected.")

        socket = self._socket

        self._send_command(
            socket,
            "build_env",
            env_id=self.remote_env_id,
            seed=self.init_seed,
            **self.benchmark_specific_kwargs,
        )
        self._receive_packed_message(socket)

        self._runtime.logger.debug(
            "Benchmark environment is ready. "
            f"benchmark_id={self.benchmark_id!r}, env_id={self.remote_env_id!r}"
        )
        return socket

    def _send_command(
        self,
        socket: ClientConnection,
        command: str,
        /,
        **args: Any,
    ) -> None:
        socket.send(json.dumps({"command": command, "args": args}))

    def _receive_packed_message(self, socket: ClientConnection) -> dict:
        payload = socket.recv(decode=False)
        return msgpack.unpackb(zstd.decompress(payload))

    def step(self, action: list[float]) -> GymEnvStepReturnType:

        socket = self._ensure_connected()
        self._send_command(socket, "step", action=list(action))
        """
        NOTE: Transfered data size
        On Libero
         - Before packing: ~2MB
         - After packing: 500KB
         - After compression: 200KB
        """
        return self._receive_packed_message(socket)

    def reset(self, seed: int = 0):
        socket = self._ensure_connected()
        self._send_command(socket, "reset", seed=seed)
        return self._receive_packed_message(socket)

    def close(self):
        if self._socket is None:
            return
        self._socket.close()
        self._socket = None

    @property
    def action_space(self): ...
    @property
    def observation_space(self): ...


class BenchmarkVecEnvironment:
    def __init__(self, benchmark_envs: list[BenchmarkEnvironment]) -> None:
        self._benchmark_envs = benchmark_envs
        self._runtime = obtain_runtime()

    def step(self, actions: list[list[float]]) -> list[GymEnvStepReturnType]:
        """Send step to all envs concurrently and gather responses.
        Results are returned in the same order as the provided `envs` list.
        """

        if len(actions) != len(self._benchmark_envs):
            self._runtime.logger.warn(
                "\n".join(
                    [
                        "Action list length is not same as Environment length.",
                        f"Action[{len(actions)}] != Environment[{len(self._benchmark_envs)}]",
                    ]
                )
            )

        if not self._benchmark_envs:
            return []

        # Phase 1: send in parallel
        def _send_payload(r: BenchmarkEnvironment, action: list[float]) -> None:
            socket = r._ensure_connected()
            r._send_command(socket, "step", action=list(action))

        def _recv_payload(r: BenchmarkEnvironment) -> GymEnvStepReturnType:
            socket = r._ensure_connected()
            return r._receive_packed_message(socket)

        with ThreadPoolExecutor(max_workers=len(self._benchmark_envs)) as ex:
            send_futs = [
                ex.submit(_send_payload, r[0], r[1])
                for r in zip(self._benchmark_envs, actions)
            ]
            # Ensure all sends complete (propagate any exceptions)
            for f in as_completed(send_futs):
                f.result()

            # Phase 2: recv in parallel, maintain order
            recv_futs = {
                ex.submit(_recv_payload, r): i
                for i, r in enumerate(self._benchmark_envs)
            }
            results: list[Any] = [None] * len(self._benchmark_envs)
            for f in as_completed(recv_futs):
                idx = recv_futs[f]
                results[idx] = f.result()

        return results

    def reset_parallel(self): ...
