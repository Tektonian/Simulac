import sys

from tt.base.envvar.envvar import IEnvvarService
from tt.sdk.environment_service.common.environment_build_service import (
    IEnvironmentBuildService,
)
from tt.sdk.environment_service.common.environment_service import (
    IEnvironmentManagementService,
)
from tt.sdk.log_service.common.log_service import ILogService
from tt.sdk.runner_service.common.runner_service import IRunnerManagementService

from .main import instantiate_service
from .world_maker import WorldMakerFacade


class SimulacRuntime:
    def __init__(self):
        old_exception_hook = sys.excepthook
        old_display_hook = sys.displayhook
        old_unraisable_hook = sys.unraisablehook

        flags = sys.flags

        debug = flags.debug
        dev_mode = flags.dev_mode
        verbose = flags.verbose
        quite = flags.quiet

        self.runner_management_service: IRunnerManagementService = (
            instantiate_service.service_accessor.get(IRunnerManagementService)
        )

        self.log_service: ILogService = instantiate_service.service_accessor.get(
            ILogService
        )
        self.environment_management_service: IEnvironmentManagementService = (
            instantiate_service.service_accessor.get(IEnvironmentManagementService)
        )
        self.environment_build_service: IEnvironmentBuildService = (
            instantiate_service.service_accessor.get(IEnvironmentBuildService)
        )
        self.envvar_service: IEnvvarService = instantiate_service.service_accessor.get(
            IEnvvarService
        )

        self._world_maker: WorldMakerFacade = instantiate_service.create_instance(
            WorldMakerFacade
        )

    @property
    def world_maker(self):
        """Return world maker facade"""
        return self._world_maker


_SINGLETON_SIMULATION_OBJECT_CACHE: SimulacRuntime | None = None


def obtain_runtime() -> SimulacRuntime:
    global _SINGLETON_SIMULATION_OBJECT_CACHE
    if _SINGLETON_SIMULATION_OBJECT_CACHE:
        return _SINGLETON_SIMULATION_OBJECT_CACHE
    else:
        _SINGLETON_SIMULATION_OBJECT_CACHE = SimulacRuntime()
        return _SINGLETON_SIMULATION_OBJECT_CACHE
