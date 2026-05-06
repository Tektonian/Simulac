from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from .model.runtime import StuffRuntime


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
    def step(self, action: list[float]) -> None:
        pass

    @abstractmethod
    def tick(self) -> None: ...

    @abstractmethod
    def reset(self, seed: int | None = 0) -> None: ...

    @abstractmethod
    def set_state(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> None:
        pass

    @abstractmethod
    def get_runtime_object(self, entity_id: str) -> StuffRuntime: ...

    @abstractmethod
    def clone_state(self) -> None: ...

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def _debug_render(self) -> Any:
        """Run adapter specific rendering engine. Should be used for debugging"""


class IRunnerFactory(ABC):
    @abstractmethod
    def create_runner(self) -> IRunner: ...
