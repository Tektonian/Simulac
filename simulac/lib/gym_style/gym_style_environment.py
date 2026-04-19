from __future__ import annotations

import json
import os
import traceback
import urllib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Tuple, Type, TypeAlias

import msgpack
import requests
import zstd
from websockets import ConnectionClosedError, InvalidHandshake
from websockets.sync.client import ClientConnection, connect

from simulac.base.error.error import SimulacBaseError
from simulac.sdk import obtain_runtime

GymEnvStepReturnType: TypeAlias = Tuple[
    dict[str, Any],  # obs
    float,  # reward
    bool,  # done
    dict[str, Any],  # info
]
GymEnvResetReturnType: TypeAlias = Tuple[
    dict[str, Any],  # obs
    dict[str, Any],  # info
]


class BenchmarkEnvironment:
    def __init__(
        self,
        owner_id: str,
        world_id: str,
        env_id: str,
        seed: int,
        benchmark_specific_kwargs: dict[str, Any],
    ):

        self._runtime = obtain_runtime()

        self.runner_id = str("")
        self.owner_id = owner_id
        self.world_id = world_id
        self.benchmark_id = f"{owner_id}/{world_id}"
        self.env_id = env_id
        self.benchmark_specific_kwargs = benchmark_specific_kwargs
        self.init_seed = seed

        self._socket: ClientConnection | None = None
        self._ticket: str | None = None

        # Warning message flags
        self._has_reset = False
        self._warned_step_before_reset = False

    def _create_ticket(self):
        url = f"{self._runtime.environment_variable.base_url}/container/{self.owner_id}/{self.world_id}/preflight"
        res = requests.post(
            url,
            headers={"tt-apikey": self._get_api_key()},
            timeout=10,
        )

        if res.status_code > 299:
            try:
                payload = res.json()
            except Exception:
                payload = {"error": "unknown", "message": res.text}
            finally:
                raise SimulacBaseError(
                    f"Failed to create ticket: {res.status_code} {payload['message']}"
                )

        res.raise_for_status()
        payload: dict = res.json()
        self._ticket = payload.get("ticket", None)

    def _get_api_key(self) -> str:
        api_key = self._runtime.environment_variable.token
        if api_key is None:
            _API_KEY_ERROR_MESSAGE = "\n".join(
                [
                    "API key for the remote benchmark service was not found.",
                    "Get a token at https://tektonian.com/settings/token.",
                    "Then run `simulac auth login` in your terminal and paste the API key.",
                ]
            )
            self._runtime.logger.error(_API_KEY_ERROR_MESSAGE)
            raise SimulacBaseError(_API_KEY_ERROR_MESSAGE)
        return api_key

    def _build_ws_url(self) -> str:
        query_param = urllib.parse.urlencode({"ticket": self._ticket})
        base_url = urllib.parse.urlparse(self._runtime.environment_variable.base_url)
        ws_scheme = {"http": "ws", "https": "wss"}.get(base_url.scheme, "wss")
        return base_url._replace(
            scheme=ws_scheme,
            path=os.path.join(
                base_url.path,
                f"container/{self.owner_id}/{self.world_id}",
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

        if self._ticket is None:
            self._create_ticket()

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
                            "benchmark": self.benchmark_id,
                            "stacktrace": traceback.format_exc(),
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
            env_id=self.env_id,
            seed=self.init_seed,
            **self.benchmark_specific_kwargs,
        )
        self._receive_packed_message(socket)

        self._runtime.logger.debug(
            "Benchmark environment is ready. "
            f"benchmark_id={self.benchmark_id!r}, env_id={self.env_id!r}"
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

        if not self._has_reset and not self._warned_step_before_reset:
            self._runtime.logger.warn(
                "\n".join(
                    [
                        "Unexpected behavior: step() was called before reset()."
                        f"benchmark_id={self.benchmark_id!r}, env_id={self.env_id!r}"
                    ]
                )
            )
            self._warned_step_before_reset = True

        self._send_command(socket, "step", action=list(action))
        """
        NOTE: Transfered data size
        On Libero
         - Before packing: ~2MB
         - After packing: 500KB
         - After compression: 200KB
        """
        rcvd = self._receive_packed_message(socket)
        try:
            obs: dict = rcvd["obs"]
            reward: float = rcvd["reward"]
            done: bool = rcvd["done"]
            info: dict = rcvd["info"]
        except KeyError as err:
            self._runtime._log_service.error(
                "\n".join(
                    [
                        "Benchmark environment runner received an invalid field",
                        "This error occurs when there is a problem with the benchmark protocol",
                        "If this issue persists, please let us know!",
                        "Discord channel: https://discord.gg/zbSaU8ZbS / Email: gangjeuk@tektonian.com",
                    ]
                )
            )
            self._runtime.telemetry.public_error(
                event_name="simulac_step_failed",
                data={
                    "err": err.args,
                    "benchmark": self.benchmark_id,
                    "stacktrace": traceback.format_exc(),
                },
            )
            raise err
        return (obs, reward, done, info)

    def reset(self, seed: int = 0) -> GymEnvResetReturnType:
        socket = self._ensure_connected()
        self._send_command(socket, "reset", seed=seed)
        rcvd = self._receive_packed_message(socket)
        try:
            self.runner_id = rcvd.get("id", "")
            obs: dict = rcvd["obs"]
            info: dict = rcvd["info"]
        except KeyError as err:
            self._runtime._log_service.error(
                "\n".join(
                    [
                        "Benchmark environment runner received an invalid field",
                        "This error occurs when there is a problem with the benchmark protocol",
                        "If this issue persists, please let us know!",
                        "Discord channel: https://discord.gg/zbSaU8ZbS / Email: gangjeuk@tektonian.com",
                    ]
                )
            )
            self._runtime.telemetry.public_error(
                event_name="simulac_reset_failed",
                data={
                    "err": err.args,
                    "benchmark": self.benchmark_id,
                    "stacktrace": traceback.format_exc(),
                },
            )
            raise err
        self._has_reset = True
        self._warned_step_before_reset = False
        return (obs, info)

    def close(self):
        self._has_reset = False
        self._warned_step_before_reset = False
        if self._socket is None:
            return
        try:
            self._send_command(self._socket, "close")
        except Exception:
            pass
        try:
            self._socket.close()
        finally:
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
                        "Action list length does not match the number of environments.",
                        f"Action[{len(actions)}] != Environment[{len(self._benchmark_envs)}]",
                    ]
                )
            )

        if not self._benchmark_envs:
            return []

        def _env_step(
            env: BenchmarkEnvironment | None, action: list[float] | None
        ) -> GymEnvStepReturnType:
            """
            Param `env` and `action` could be None because of `zip` function
            and length difference is permitted
            """
            if env is not None and action is not None:
                return env.step(action)
            blame = "Environment" if env is None else "Action"
            return (
                {},
                0.0,
                False,
                {"error": f"No observation returned since {blame} is empty"},
            )

        with ThreadPoolExecutor(max_workers=len(self._benchmark_envs)) as ex:
            ret_futs = {
                ex.submit(_env_step, env, action): i
                for i, (env, action) in enumerate(zip(self._benchmark_envs, actions))
            }

            ret: list[Any] = [None] * len(self._benchmark_envs)
            for f in as_completed(ret_futs):
                idx = ret_futs[f]
                ret[idx] = f.result()
        return ret

    def reset(self, seeds: list[int]) -> list[GymEnvResetReturnType]:
        if len(seeds) != len(self._benchmark_envs):
            self._runtime.logger.warn(
                "\n".join(
                    [
                        "Seed list length does not match the number of environments.",
                        f"Seed[{len(seeds)}] != Environment[{len(self._benchmark_envs)}]",
                    ]
                )
            )

        def _env_reset(
            env: BenchmarkEnvironment | None, seed: int | None
        ) -> GymEnvResetReturnType:
            """
            Param `env` and `seed` could be None because of `zip` function
            and length difference is permitted
            """
            blame = "Environment" if env is None else "Seed"
            if env is not None and seed is not None:
                return env.reset(seed)
            return ({}, {"error": f"No observation returned since {blame} is empty"})

        with ThreadPoolExecutor(max_workers=len(self._benchmark_envs)) as ex:
            ret_futs = {
                ex.submit(_env_reset, env, seed): i
                for i, (env, seed) in enumerate(zip(self._benchmark_envs, seeds))
            }

            ret: list[Any] = [None] * len(self._benchmark_envs)
            for f in as_completed(ret_futs):
                idx = ret_futs[f]
                ret[idx] = f.result()
        return ret

    def close(self):
        with ThreadPoolExecutor(max_workers=len(self._benchmark_envs)) as ex:
            close_futs = [ex.submit(env.close) for env in self._benchmark_envs]
            for f in as_completed(close_futs):
                f.result()
