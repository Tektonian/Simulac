from __future__ import annotations

import json
import os
import socket
import traceback
import urllib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Tuple, Type, TypeAlias

import msgpack
import requests
import zstd
from websockets import (
    ConnectionClosed,
    ConnectionClosedError,
    InvalidHandshake,
)
from websockets.sync.client import ClientConnection, connect

from simulac.base.error.error import SimulacBaseError
from simulac.sdk import obtain_runtime

type GymEnvStepReturnType = Tuple[
    dict[str, Any],  # obs
    float,  # reward
    bool,  # done
    dict[str, Any],  # info
]
"""Gym style `.step()` function return type.
If follows basic gymnasium style of `(obs, reward, done, info) = env.reset()`
For detailed `.keys()` of `obs` and `info` goto https://tektonian.com/benchmark page,
and see specs for each benchmark.

Raises:
    SimulacBaseError: When error occured in server side

"""
type GymEnvResetReturnType = Tuple[
    dict[str, Any],  # obs
    dict[str, Any],  # info
]
"""Gym style `.reset()` function return type.
If follows basic gymnasium style of `(obs, info) = env.reset()`

Raises:
    SimulacBaseError: When error occured in server side
"""

__KEYERROR_MESSAGE = "\n".join(
    [
        "Benchmark environment runner received an invalid field",
        "This error occurs when there is a problem with the benchmark protocol",
        "If this issue persists, please let us know!",
        "Discord channel: https://discord.gg/zbSaU8ZbS / Email: gangjeuk@tektonian.com",
    ]
)


class BenchmarkEnvironment:
    """Gym-style client for a remote benchmark environment."""

    def __init__(
        self,
        owner_id: str,
        world_id: str,
        env_id: str,
        seed: int,
        benchmark_specific_kwargs: dict[str, Any],
        *,
        error_recovery_enabled: bool = False,
    ):

        self._runtime = obtain_runtime()

        self.runner_id: str = str("")
        self.owner_id = owner_id
        self.world_id = world_id
        self.benchmark_id = f"{owner_id}/{world_id}"
        self.env_id = env_id
        self.benchmark_specific_kwargs = benchmark_specific_kwargs
        self.init_seed = seed

        self._socket: ClientConnection | None = None
        self._ticket: str | None = None

        # Step history storage for error recovering
        # When connection is lost, repeat senario with same step value
        self.__step_history: list[list[float]] = []
        self.__error_recovery_count = 0
        self.__MAX_ERROR_RECOVERY_COUNT = 5
        self.__last_reset_seed = seed
        # TODO: @gangjeuk
        # Need to discuss about how to deal with unrecoverable errors
        # e.g., `Expected Action space is 4, but given 7` <- We should not handle this, but we do now
        # option 1 - add error code for backend server code
        # option 2 - seperate error recovery by cases
        # Currently, we use option 2 by using `_error_recovery_enabled` in `init_bench()` and `make_vec()`
        # Is it good enough?
        self._error_recovery_enabled = error_recovery_enabled

        # Warning message flags
        self._has_reset = False
        self._warned_step_before_reset = False

    def _is_network_alive(self):
        """Return whether the local network appears reachable."""
        try:
            # Connect to Google's public DNS (8.8.8.8) on port 53
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except Exception:
            pass
        return False

    def _create_ticket(self):
        """Create a remote benchmark ticket for websocket connection."""
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
        """Return the configured API key for the benchmark service.

        Raises:
            SimulacBaseError: If no API key is configured.
        """
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
        """Build the websocket URL for the current benchmark ticket."""
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
        """Create the websocket connection lazily and build the remote environment.

        Raises:
            SimulacBaseError: If the socket connection or environment build fails.

        Returns:
            Active client websocket connection.
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
        """Send one JSON command over the benchmark websocket."""
        socket.send(json.dumps({"command": command, "args": args}))

    def _receive_packed_message(self, socket: ClientConnection) -> dict:
        """Receive and decode one compressed msgpack response."""
        payload = socket.recv(decode=False)
        return msgpack.unpackb(zstd.decompress(payload))

    def _drop_socket(self) -> None:
        """Close and clear local websocket state."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass

        self._socket = None
        self._ticket = None

    def _parse_step_response(self, rcvd: dict) -> GymEnvStepReturnType:
        """Parse a backend step response into the public Gym-style tuple."""
        try:
            return (
                rcvd["obs"],
                rcvd["reward"],
                rcvd["done"],
                rcvd["info"],
            )
        except KeyError as err:
            self._runtime.logger.error(__KEYERROR_MESSAGE)
            self._runtime.telemetry.public_error(
                event_name="simulac_step_failed",
                data={
                    "err": err.args,
                    "benchmark": self.benchmark_id,
                    "stacktrace": traceback.format_exc(),
                },
            )
            raise

    def _step_once(self, action: list[float]) -> GymEnvStepReturnType:
        """Send one step command without recovery handling."""
        socket = self._ensure_connected()
        self._send_command(socket, "step", action=list(action))
        rcvd = self._receive_packed_message(socket)
        return self._parse_step_response(rcvd)

    def _reset_once(self, seed: int) -> GymEnvResetReturnType:
        """Send one reset command without recovery handling."""
        socket = self._ensure_connected()
        self._send_command(socket, "reset", seed=seed)
        rcvd = self._receive_packed_message(socket)

        try:
            self.runner_id = rcvd.get("id", "")
            return rcvd["obs"], rcvd["info"]
        except KeyError as err:
            self._runtime.logger.error(__KEYERROR_MESSAGE)
            self._runtime.telemetry.public_error(
                event_name="simulac_reset_failed",
                data={
                    "err": err.args,
                    "benchmark": self.benchmark_id,
                    "stacktrace": traceback.format_exc(),
                },
            )
            raise

    def _set_error_recovery_enabled(self, enabled: bool):
        """Enable or disable connection recovery for this environment."""
        self._error_recovery_enabled = enabled

    def _recover_and_replay(self, pending_action: list[float]) -> GymEnvStepReturnType:
        """Reconnect, reset with the last seed, and replay actions after transport loss."""
        history = [list(action) for action in self.__step_history]
        actions_to_replay = history + [list(pending_action)]

        last_result: GymEnvStepReturnType | None = None

        self._runtime.logger.warn(
            "\n".join(
                [
                    "Recovering benchmark environment after connection failure.",
                    f"Benchmark: {self.benchmark_id}",
                    f"Runner: {self.runner_id}",
                    f"attempt={self.__error_recovery_count}/{self.__MAX_ERROR_RECOVERY_COUNT}",
                ]
            )
        )

        for attempt in range(1, self.__MAX_ERROR_RECOVERY_COUNT + 1):
            try:
                self._runtime.logger.debug(
                    f"Runner: {self.runner_id}, Recovering attempt: {attempt}"
                )
                self._drop_socket()
                self._ensure_connected()
                self._reset_once(self.__last_reset_seed)

                for idx, action in enumerate(actions_to_replay):
                    last_result = self._step_once(action)
                    self._runtime.logger.debug(
                        f"recovery={idx + 1}/{len(actions_to_replay)}"
                    )
                self._runtime.logger.info(f"Runner: {self.runner_id} recovered")
                self.__step_history = actions_to_replay
                self.__error_recovery_count = 0
                return last_result

            except (
                ConnectionClosed,
                ConnectionClosedError,
                TimeoutError,
                OSError,
            ) as _:
                self.__error_recovery_count = attempt
                continue

        raise SimulacBaseError(
            f"Failed to recover benchmark environment after "
            f"{self.__MAX_ERROR_RECOVERY_COUNT} attempts."
        )

    def step(self, action: list[float]) -> GymEnvStepReturnType:
        """Send one action to the remote environment.

        Args:
            action: Action vector for one environment step.

        Returns:
            Gym-style step tuple of observation, reward, done, and info.
        """
        action_copy = list(action)
        recovered = False

        if not self._has_reset and not self._warned_step_before_reset:
            self._runtime.logger.warn(
                "\n".join(
                    [
                        "Unexpected behavior: step() was called before reset().",
                        f"benchmark_id={self.benchmark_id!r}, env_id={self.env_id!r}",
                    ]
                )
            )
            self._warned_step_before_reset = True

        try:
            result = self._step_once(action_copy)
        except (ConnectionClosed, ConnectionClosedError, TimeoutError, OSError) as err:
            if (
                self._is_network_alive() is False
                or self._error_recovery_enabled is False
            ):
                raise err
            result = self._recover_and_replay(action_copy)
            recovered = True

        if not recovered:
            self.__step_history.append(action_copy)
        return result

    def reset(self, seed: int = 0) -> GymEnvResetReturnType:
        """Reset the remote environment and clear replay history.

        Args:
            seed: Reset seed.

        Returns:
            Gym-style reset tuple of observation and info.
        """
        result = self._reset_once(seed)

        self.__last_reset_seed = seed
        self.__step_history.clear()
        self.__error_recovery_count = 0

        self._has_reset = True
        self._warned_step_before_reset = False

        return result

    def close(self):
        """Close the remote runner and drop local connection state."""
        self._has_reset = False
        self._warned_step_before_reset = False
        if self._socket is None:
            return
        try:
            self._send_command(self._socket, "close")
        except Exception:
            pass
        self._drop_socket()

    @property
    def action_space(self): ...
    @property
    def observation_space(self): ...


class BenchmarkVecEnvironment:
    """Vectorized wrapper over multiple BenchmarkEnvironment instances."""

    def __init__(self, benchmark_envs: list[BenchmarkEnvironment]) -> None:
        self._benchmark_envs = benchmark_envs
        self._runtime = obtain_runtime()

    def step(self, actions: list[list[float]]) -> list[GymEnvStepReturnType]:
        """Step all benchmark environments and preserve input order.

        Args:
            actions: Action vectors aligned with the wrapped environments.

        Returns:
            Step results aligned with the wrapped environments.
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
        """Reset all benchmark environments and preserve input order.

        Args:
            seeds: Reset seeds aligned with the wrapped environments.

        Returns:
            Reset results aligned with the wrapped environments.
        """
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
        """Close all benchmark environments."""
        with ThreadPoolExecutor(max_workers=len(self._benchmark_envs)) as ex:
            close_futs = [ex.submit(env.close) for env in self._benchmark_envs]
            for f in as_completed(close_futs):
                f.result()
