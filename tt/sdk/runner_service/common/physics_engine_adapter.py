from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from tt.sdk.environment_service.common.model.entity import (
        EnvironmentMJCFObjectEntity,
    )
    from tt.sdk.runner_service.common.runner import IRunner


class PhysicsEngineAdapter(ABC):

    @abstractmethod
    # TODO: change parameter 'object' type
    def add_object(self, obj: EnvironmentMJCFObjectEntity) -> str:  # TODO
        pass

    @abstractmethod
    def remove_object(self, obj_id: int) -> None:
        pass

    @abstractmethod
    def start_physics_engine(self) -> None: ...

    @abstractmethod
    def step(self, dt: float) -> None:
        pass

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_state(self, obj_id: int) -> object: ...
