from __future__ import annotations

from dataclasses import dataclass

from .runner import Runner, RuntimeState


class ScenarioEnvironment:
    """
    Read-only Environment that doesn't support modification of the Environment,
    like `add_entity()` or `remove_entity()`

    Environment so it can
    inspect prepared objects without changing the scene definition.
    """


@dataclass
class ScenarioContext:
    """Runtime context passed to Scenario lifecycle hooks."""

    runner: Runner
    state: RuntimeState
    seed: int | None


class Scenario:
    """Base class for runtime scenario logic.

    Subclass this to define automatic behavior that runs around Runner reset
    and tick calls, such as moving objects, respawning objects, or changing
    runtime state based on conditions.

    Expected usage pattern is:
        class ConveyorScenario(Scenario): ...

        Candi 1:
            runner = Runner(env, scenario=ConveyorScenario)
        Candi 2:
            runner = Runner(env, scenario=ConveyorScenario())
        Candi 3:
            scenario = ConveyorScenario()
            env = Environment()
            scenario.build(env)
            runner = Runner(env, scenario=scenario)
    """

    def on_build(self, env: ScenarioEnvironment) -> None:
        """Inspect the prepared scenario environment before runner execution."""

    def after_reset(self, ctx: ScenarioContext) -> None:
        """Run scenario logic immediately after runner reset."""

    def before_tick(self, ctx: ScenarioContext) -> None:
        """Run scenario logic immediately before one runner tick."""

    def after_tick(self, ctx: ScenarioContext) -> None:
        """Run scenario logic immediately after one runner tick."""
