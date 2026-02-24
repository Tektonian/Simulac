from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlsplit, SplitResult

from tt.base.error.error import TektonianBaseError
from tt.base.instantiate.instantiate import service_identifier, ServiceIdentifier
from tt.base.result.result import ResultType


# https://gymnasium.farama.org/api/env/
# https://docs.wandb.ai/models/ref/python/experiments/run#property-run-entity
@dataclass
class IEnvironment(ABC):
    id: str
    world_id: str
    env_uri: str | SplitResult = field(init=False)

    objects: list[object] = field(default_factory=list[object])
    proprioception: list[object] = field(default_factory=list[object])
    cameras: list[object] = field(default_factory=list[object])

    @abstractmethod
    def snapshop(self):
        pass

    @abstractmethod
    def load_env(self):
        pass

    def __post_init__(self):
        url = urlsplit(self.env_uri) if isinstance(self.env_uri, str) else self.env_uri

        if url.scheme not in ["http", "https", "file"]:
            raise TektonianBaseError("not valid schema")

        self.url = url


@service_identifier("IEnvironmentManagementService")
class IEnvironmentManagementService(ServiceIdentifier):

    __ID_PREFIX = "env_"

    @property
    @abstractmethod
    def _environments(self) -> Mapping[str, IEnvironment]:
        pass

    @abstractmethod
    def get_environment(
        self, environment_id: str
    ) -> ResultType[IEnvironment, BaseException]:
        pass

    @abstractmethod
    def create_environment(self, env_uri: str, seed: int) -> IEnvironment:
        pass

    @abstractmethod
    def register_environment(
        self, environment: IEnvironment
    ) -> ResultType[IEnvironment, BaseException]:
        pass
