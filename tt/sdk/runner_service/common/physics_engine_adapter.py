from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import deprecated


if TYPE_CHECKING:
    from tt.sdk.environment_service.common.model.entity import (
        EnvironmentMJCFObjectEntity,
    )
    from tt.sdk.runner_service.common.runner import IRunner, IRunnerFactory


class IPhysicsEngineAdapter(ABC):

    @abstractmethod
    def initialize(self) -> IRunnerFactory: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_state(self, obj_id: int) -> object: ...
