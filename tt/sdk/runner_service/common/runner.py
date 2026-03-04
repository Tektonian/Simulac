from abc import ABC, abstractmethod
from dataclasses import dataclass

from tt.sdk.runner_service.common.physics_engine_adapter import PhysicsEngineAdapter


@dataclass
class IRunner(ABC):

    id: str
    env_id: str
    state: object

    @abstractmethod
    def step(self, action: object) -> object:
        pass

    @abstractmethod
    def set_state(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> None:
        pass

    @abstractmethod
    def clone_state(self) -> None: ...

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def create_physics_engine_adapter(self) -> PhysicsEngineAdapter: ...
