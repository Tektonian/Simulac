from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from .model.context import INativeContext
    from .model.runtime import CameraRuntime, RobotRuntime, RuntimeState, StuffRuntime


@dataclass
class IRunner(ABC):
    __ID_PREVIX = "run_"

    runner_type: str
    id: str
    env_id: str
    state: object

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def step(self, action: list[float]) -> RuntimeState:
        pass

    @abstractmethod
    def tick(self) -> RuntimeState: ...

    @abstractmethod
    def reset(self, seed: int | None = 0) -> RuntimeState: ...

    @abstractmethod
    def sync(self) -> RuntimeState: ...

    @abstractmethod
    def set_state(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> RuntimeState: ...

    @abstractmethod
    def get_runtime_object(
        self, entity_id: str
    ) -> StuffRuntime | RobotRuntime | CameraRuntime: ...

    @abstractmethod
    def clone_state(self) -> None: ...

    @abstractmethod
    def snapshot(self) -> None:
        """Take a snapshot (like a screenshot).
        This is for future use, when we implemented web browser renderer
        """

    @abstractmethod
    def _debug_render(self) -> Any:
        """Run adapter specific rendering engine. Should be used for debugging"""

    @abstractmethod
    def context(self, engine: str) -> "INativeContext": ...


class IRunnerFactory(ABC):
    @abstractmethod
    def create_runner(self) -> IRunner: ...
