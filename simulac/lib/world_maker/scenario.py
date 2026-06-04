from __future__ import annotations

from dataclasses import dataclass

from .runner import Runner, RuntimeState


class ScenarioEnvironment:
    """Environment that doesn't support modification of the Environment like `add_entity()` or `remove_entity()`"""

    ...


@dataclass
class ScenarioContext:
    runner: Runner
    state: RuntimeState
    seed: int | None


class Scenario:
    def on_build(self, env: ScenarioEnvironment) -> None: ...

    def after_reset(self, ctx: ScenarioContext) -> None: ...
    def before_tick(self, ctx: ScenarioContext) -> None: ...

    def after_tick(self, ctx: ScenarioContext) -> None: ...
