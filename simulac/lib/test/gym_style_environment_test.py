from __future__ import annotations

import pytest
from websockets import ConnectionClosedError
from websockets.frames import Close

from simulac.base.error.error import SimulacBaseError
from simulac.lib.gym_style.gym_style_environment import (
    BenchmarkEnvironment,
    BenchmarkVecEnvironment,
)


class _FakeBenchmarkEnvironment:
    def __init__(self, name: str) -> None:
        self.name = name
        self._pending_messages: list[tuple[str, object]] = []
        self.command_log: list[tuple[str, object]] = []

    def _ensure_connected(self) -> "_FakeBenchmarkEnvironment":
        return self

    def _send_command(self, _socket: object, command: str, /, **args: object) -> None:
        self.command_log.append((command, args))
        if command == "reset":
            self._pending_messages.append(
                (
                    command,
                    (
                        {"obs": f"{self.name}-reset-{args['seed']}"},
                        {"info": f"{self.name}-reset-info"},
                    ),
                )
            )
            return

        if command == "step":
            self._pending_messages.append(
                (
                    command,
                    (
                        {"obs": f"{self.name}-step-{args['action']}"},
                        1.0,
                        False,
                        {"info": f"{self.name}-step-info"},
                    ),
                )
            )
            return

        raise AssertionError(f"Unexpected command: {command}")

    def _receive_packed_message(self, _socket: object) -> object:
        if not self._pending_messages:
            raise AssertionError("No pending message to receive.")
        return self._pending_messages.pop(0)[1]

    def reset(self, seed: int) -> tuple[dict[str, object], dict[str, object]]:
        self._send_command(None, "reset", seed=seed)
        return self._receive_packed_message(None)

    def step(
        self, action: list[float]
    ) -> tuple[dict[str, object], float, bool, dict[str, object]]:
        self._send_command(None, "step", action=action)
        return self._receive_packed_message(None)


def test_vec_env_return_keep_order() -> None:
    envs = [_FakeBenchmarkEnvironment("env0"), _FakeBenchmarkEnvironment("env1")]
    vec_env = BenchmarkVecEnvironment(envs)  # type: ignore[arg-type]

    reset_results = vec_env.reset([7, 11])

    assert reset_results == [
        ({"obs": "env0-reset-7"}, {"info": "env0-reset-info"}),
        ({"obs": "env1-reset-11"}, {"info": "env1-reset-info"}),
    ]

    step_results = vec_env.step([[0.1, 0.2], [0.3, 0.4]])

    assert step_results == [
        ({"obs": "env0-step-[0.1, 0.2]"}, 1.0, False, {"info": "env0-step-info"}),
        ({"obs": "env1-step-[0.3, 0.4]"}, 1.0, False, {"info": "env1-step-info"}),
    ]


class _FakeRecoveryBenchmarkEnvironment(BenchmarkEnvironment):
    def __init__(self) -> None:
        super().__init__("Tektonian", "Metaworld", "assembly-v3", 0, {})
        self._set_error_recovery_enabled(True)
        self._socket = object()
        self.send_calls: list[tuple[str, dict[str, object]]] = []
        self.send_error_plan: dict[str, list[BaseException]] = {
            "reset": [],
            "step": [],
        }
        self.recv_error_plan: dict[str, list[BaseException]] = {
            "reset": [],
            "step": [],
        }
        self._last_command: str | None = None
        self._last_args: dict[str, object] = {}

    def _ensure_connected(self) -> object:
        self._socket = object()
        return self._socket

    def _send_command(self, _socket: object, command: str, /, **args: object) -> None:
        self.send_calls.append((command, dict(args)))
        self._last_command = command
        self._last_args = dict(args)

        planned_errors = self.send_error_plan.get(command, [])
        if planned_errors:
            raise planned_errors.pop(0)

    def _receive_packed_message(self, _socket: object) -> object:
        if self._last_command is None:
            raise AssertionError("No command was sent before receiving.")

        planned_errors = self.recv_error_plan.get(self._last_command, [])
        if planned_errors:
            raise planned_errors.pop(0)

        if self._last_command == "reset":
            seed = self._last_args.get("seed")
            if not isinstance(seed, int):
                raise AssertionError(
                    f"reset args should include int seed, got {seed!r}"
                )
            return {
                "id": f"runner-{seed}",
                "obs": {"seed": seed},
                "info": {"seed": seed},
            }

        if self._last_command == "step":
            action = self._last_args.get("action")
            if not isinstance(action, list):
                raise AssertionError(
                    f"step args should include action list, got {action!r}"
                )
            return {
                "obs": {"action": list(action)},
                "reward": 1.0,
                "done": False,
                "info": {"action_size": len(action)},
            }

        return {"obs": {}, "reward": 0.0, "done": False, "info": {}}


def _find_step_actions(
    env: _FakeRecoveryBenchmarkEnvironment,
) -> list[list[float]]:
    return [list(call[1]["action"]) for call in env.send_calls if call[0] == "step"]


def _set_error_plan(
    env: _FakeRecoveryBenchmarkEnvironment,
    *,
    send_step: list[BaseException] | None = None,
    send_reset: list[BaseException] | None = None,
    recv_step: list[BaseException] | None = None,
    recv_reset: list[BaseException] | None = None,
) -> None:
    if send_step is not None:
        env.send_error_plan["step"] = send_step
    if send_reset is not None:
        env.send_error_plan["reset"] = send_reset
    if recv_step is not None:
        env.recv_error_plan["step"] = recv_step
    if recv_reset is not None:
        env.recv_error_plan["reset"] = recv_reset


@pytest.mark.parametrize("phase", ["send", "recv"])
def test_error_recovery_transport_error_recovered(phase: str, monkeypatch) -> None:
    env = _FakeRecoveryBenchmarkEnvironment()
    env.reset(0)
    action = [0, 0, 0, 0]

    if phase == "send":
        _set_error_plan(env, send_step=[OSError("WebSocket Send failed!")])
    else:
        _set_error_plan(env, recv_step=[OSError("WebSocket Recv failed!")])
    monkeypatch.setattr(env, "_is_network_alive", lambda: True)

    result = env.step(action)

    assert result == (
        {"action": action},
        1.0,
        False,
        {"action_size": 4},
    )
    assert _find_step_actions(env) == [action, action]
    assert env._BenchmarkEnvironment__step_history == [action]


def test_error_recovery_connection_closed_is_recovered(monkeypatch) -> None:
    env = _FakeRecoveryBenchmarkEnvironment()
    env.reset(0)
    monkeypatch.setattr(env, "_is_network_alive", lambda: True)
    action = [0, 0, 0, 0, 0, 0, 0]

    _set_error_plan(
        env,
        recv_step=[
            ConnectionClosedError(
                Close(1011, "AssertionError('Actions should be size 4, got 7')"),
                Close(1011, "AssertionError('Actions should be size 4, got 7')"),
                True,
            )
        ],
    )

    result = env.step(action)

    assert result == (
        {"action": action},
        1.0,
        False,
        {"action_size": len(action)},
    )
    assert _find_step_actions(env) == [action, action]
    assert env._BenchmarkEnvironment__error_recovery_count == 0
    assert env._BenchmarkEnvironment__step_history == [action]


def test_error_recovery_network_unreachable(monkeypatch) -> None:
    env = _FakeRecoveryBenchmarkEnvironment()
    env.reset(0)

    _set_error_plan(env, send_step=[OSError("Network down")])
    monkeypatch.setattr(env, "_is_network_alive", lambda: False)

    action = [0, 0, 0, 0]
    with pytest.raises(OSError, match="Network down"):
        env.step(action)

    assert env._BenchmarkEnvironment__error_recovery_count == 0
    assert env._BenchmarkEnvironment__step_history == []


def test_error_recovery_failed_after_max_attempts(monkeypatch) -> None:
    env = _FakeRecoveryBenchmarkEnvironment()
    env.reset(0)

    _set_error_plan(
        env,
        send_step=[OSError("initial send fail")],
        send_reset=[OSError("reset send fail")] * 5,
    )
    monkeypatch.setattr(env, "_is_network_alive", lambda: True)

    action = [0, 0, 0, 0]
    with pytest.raises(
        SimulacBaseError, match="Failed to recover benchmark environment"
    ):
        env.step(action)

    assert env._BenchmarkEnvironment__error_recovery_count == 5
    assert env._BenchmarkEnvironment__step_history == []


def test_error_recovery_partial_success(monkeypatch) -> None:
    env = _FakeRecoveryBenchmarkEnvironment()
    env.reset(0)

    _set_error_plan(
        env,
        send_step=[OSError("initial send fail")],
        send_reset=[
            OSError("reset send fail"),
            OSError("reset send fail"),
        ],
    )
    monkeypatch.setattr(env, "_is_network_alive", lambda: True)

    action = [1, 1, 1, 1]
    result = env.step(action)

    assert result == (
        {"action": action},
        1.0,
        False,
        {"action_size": 4},
    )
    assert env._BenchmarkEnvironment__error_recovery_count == 0
    assert env._BenchmarkEnvironment__step_history == [action]


def test_error_recovery_reset_clears_history() -> None:
    env = _FakeRecoveryBenchmarkEnvironment()
    env.reset(0)
    env.step([0, 0, 0, 0])
    env.step([1, 1, 1, 1])
    assert env._BenchmarkEnvironment__step_history == [[0, 0, 0, 0], [1, 1, 1, 1]]

    env.reset(11)
    assert env._BenchmarkEnvironment__step_history == []


def test_error_recovery_replays_step_history(monkeypatch) -> None:
    env = _FakeRecoveryBenchmarkEnvironment()
    env.reset(0)
    env.step([0, 0, 0, 0])
    env.step([0.1, 0.2, 0.3, 0.4])

    _set_error_plan(env, send_step=[OSError("initial send fail")])
    monkeypatch.setattr(env, "_is_network_alive", lambda: True)

    env.step([0.2, 0.3, 0.4, 0.5])
    assert env._BenchmarkEnvironment__step_history == [
        [0, 0, 0, 0],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
    ]
    assert _find_step_actions(env) == [
        [0, 0, 0, 0],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0, 0, 0, 0],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
    ]


def test_error_recovery_uses_last_reset_seed(monkeypatch) -> None:
    env = _FakeRecoveryBenchmarkEnvironment()
    env.reset(99)

    original_reset_once = env._reset_once
    called_seeds: list[int] = []

    def tracking_reset_once(seed: int):
        called_seeds.append(seed)
        return original_reset_once(seed)

    monkeypatch.setattr(env, "_reset_once", tracking_reset_once)
    _set_error_plan(env, send_step=[OSError("initial send fail")])
    monkeypatch.setattr(env, "_is_network_alive", lambda: True)

    env.step([0, 0, 0, 0])
    assert called_seeds == [99]
    assert env.runner_id == "runner-99"
