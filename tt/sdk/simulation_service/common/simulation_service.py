from abc import ABC, abstractmethod
from typing import Mapping, Union, Tuple, List, Callable, Any
from dataclasses import dataclass

import numpy as np

from tt.base.error.error import TektonianBaseError
from tt.base.result.result import ResultType
from tt.base.instantiate.instantiate import service_identifier, ServiceIdentifier


@dataclass
class ISimulation:
    id: str
    task: str
    reward: int
    terminated: bool
    truncated: bool
    extra: object


@service_identifier("ISimulationManagementService")
class ISimulationManagementService(ServiceIdentifier):

    _ID_PREFIX = "sim_"

    @property
    @abstractmethod
    def _simulations(self) -> List[ISimulation]:
        pass

    @abstractmethod
    def get_simulation(
        self, simulation_id: str
    ) -> ResultType[ISimulation, BaseException]:
        pass


class SimulationManagementService(ISimulationManagementService):
    def __init__(self) -> None:
        self.simulations: List[ISimulation] = []

    @property
    def _simulations(self) -> List[ISimulation]:
        return self.simulations

    def get_simulation(self, simulation_id: str):
        for sim in self.simulations:
            if sim.id == simulation_id:
                return (sim, None)

        return (None, TektonianBaseError("no registered simulation"))
