from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, MutableMapping

from tt.sdk.runner_service.common.runner import IRunnerFactory

if TYPE_CHECKING:
    from tt.sdk.environment_service.common.model.entity import (
        EnvironmentMJCFObjectEntity,
    )


class IPhysicsEngineAdapterFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_physics_engine_adapter(env_id: str) -> IPhysicsEngineAdapter: ...


@dataclass(frozen=True)
class IPhysicsEngineAdapterState:
    env_id: str
    runner_count: int
    step_count_map: MutableMapping[str, int]


class IPhysicsEngineAdapter(IRunnerFactory, ABC):
    @abstractmethod
    def get_state(self) -> IPhysicsEngineAdapterState: ...
