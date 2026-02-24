from abc import ABC, abstractmethod
from typing import Mapping, Union, Tuple, List, Callable, Any
from dataclasses import dataclass

import numpy as np

from tt.base.result.result import ResultType
from tt.base.instantiate.instantiate import service_identifier, ServiceIdentifier


@dataclass
class ISimulation:

    task: str
    reward: int
    terminated: bool
    truncated: bool
    extra: object


@service_identifier("ISimulationManagementService")
class ISimulationManagementService(ServiceIdentifier):

    __ID_PREFIX = "sim_"

    @property
    @abstractmethod
    def _simulations(self) -> Mapping[str, ISimulation]:
        pass

    @abstractmethod
    def get_simulation(
        self, simulation_id: str
    ) -> ResultType[ISimulation, BaseException]:
        pass

    @abstractmethod
    def register_simulation(
        self, simulation: ISimulation
    ) -> ResultType[ISimulation, BaseException]:
        pass

    @abstractmethod
    def step(
        self, simulation_or_id: ISimulation | str, action: np.ndarray | List[float]
    ) -> object:
        pass
