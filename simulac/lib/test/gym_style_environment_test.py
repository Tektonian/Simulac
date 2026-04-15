from __future__ import annotations

from simulac.lib.gym_style.gym_style_environment import BenchmarkVecEnvironment


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
