from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Mapping

from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from tt.base.result.result import ResultType
from tt.sdk.environment_service.common.environment_service import IEnvironment


@dataclass
class IWorld:
    id: str
    environments: list[IEnvironment] = field(default_factory=list[IEnvironment])


@service_identifier("IWorldManagementService")
class IWorldManagementService(ServiceIdentifier):

    @property
    @abstractmethod
    def _worlds(self) -> List[IWorld]:
        pass

    @abstractmethod
    def get_world(self, world_id: str) -> ResultType[IWorld, BaseException]:
        pass

    @abstractmethod
    def create_world(
        self, environments: List[IEnvironment]
    ) -> ResultType[IWorld, BaseException]:
        pass


class WorldManagementService(IWorldManagementService):
    def __init__(self) -> None:
        self.worlds: List[IWorld] = []

    @property
    def _worlds(self) -> List[IWorld]:
        return self.worlds
