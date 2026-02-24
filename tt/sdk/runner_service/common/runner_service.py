from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from tt.base.result.result import ResultType


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
    def render(self) -> None:
        pass


@service_identifier("IRunnerService")
class IRunnerService(ServiceIdentifier):

    __ID_PREFIX = "run_"

    @property
    @abstractmethod
    def _runners(self) -> List[IRunner]:
        pass

    @abstractmethod
    def get_runner(self, runner_id: str) -> ResultType[IRunner, BaseException]:
        pass

    @abstractmethod
    def create_runner(self) -> ResultType[str, BaseException]:
        pass
