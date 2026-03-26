from typing import Any

from tt.base.envvar.envvar import IEnvvarService
from tt.base.envvar.envvar_service import EnvvarService
from tt.base.instantiate.extensions import (
    get_singleton_service_descriptors,
    register_singleton,
)
from tt.base.instantiate.instantiate_service import InstantiateService
from tt.base.instantiate.service_collection import ServiceCollection
from tt.sdk.environment_service.common.environment_build_service import (
    EnvironmentBuildService,
    IEnvironmentBuildService,
)
from tt.sdk.environment_service.common.environment_service import (
    EnvironmentManagementService,
    IEnvironmentManagementService,
)
from tt.sdk.log_service.common.log_service import ILogService, LogService
from tt.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterFactory,
)
from tt.sdk.runner_service.common.runner_service import (
    IRunnerManagementService,
    RunnerManagementService,
)
from tt.sdk.runner_service.local.mujoco_adapter import MujocoAdapter
from tt.sdk.runner_service.local.newton_adapter import NewtonAdapter
from tt.sdk.runner_service.remote.remote_adapter import RemoteAdapter
from tt.sdk.telemetry_service.common.telemetry_service import (
    ITelemetryService,
    TelemetryService,
)
from tt.sdk.world_service.common.world_service import (
    IWorldManagementService,
    WorldManagementService,
)

# ======================================================
# PLEASE DO NOT CHANGE THE ORDER OF EACH '# region'
# ======================================================

# region register singleton services

register_singleton(ILogService, LogService)
register_singleton(IEnvvarService, EnvvarService)
register_singleton(ITelemetryService, TelemetryService)
register_singleton(IRunnerManagementService, RunnerManagementService)
register_singleton(IWorldManagementService, WorldManagementService)
register_singleton(IEnvironmentManagementService, EnvironmentManagementService)
register_singleton(IEnvironmentBuildService, EnvironmentBuildService)

# end-region

# region register services

services = get_singleton_service_descriptors()
services = ServiceCollection(services)

instantiate_service = InstantiateService(services)

# end-region

# region get services TODO
# Getting services directly is anti-pattern. Should be change in the future

runner_management_service: IRunnerManagementService = (
    instantiate_service.service_accessor.get(IRunnerManagementService)
)
log_service: ILogService = instantiate_service.service_accessor.get(ILogService)
environment_management_service: IEnvironmentManagementService = (
    instantiate_service.service_accessor.get(IEnvironmentManagementService)
)
environment_build_service: IEnvironmentBuildService = (
    instantiate_service.service_accessor.get(IEnvironmentBuildService)
)


# end-region


# region register physical engine adapter
class MujocoAdapterFactory(IPhysicsEngineAdapterFactory):
    def __init__(self) -> None: ...
    @staticmethod
    def create_physics_engine_adapter(env_id: str) -> IPhysicsEngineAdapter:
        return MujocoAdapter(
            env_id,
            log_service,
            runner_management_service,
            environment_management_service,
        )


runner_management_service.register_physics_adapter_factory(
    ["mujoco"],
    instantiate_service.create_instance(
        MujocoAdapterFactory,
    ),
)


class NewtonAdapterFactory(IPhysicsEngineAdapterFactory):
    def __init__(self) -> None: ...
    @staticmethod
    def create_physics_engine_adapter(env_id: str) -> IPhysicsEngineAdapter:
        return NewtonAdapter(
            env_id,
            log_service,
            runner_management_service,
            environment_management_service,
        )


runner_management_service.register_physics_adapter_factory(
    ["newton"],
    instantiate_service.create_instance(
        NewtonAdapterFactory,
    ),
)


class RemoteAdapterFactory(IPhysicsEngineAdapterFactory):
    def __init__(self) -> None: ...
    @staticmethod
    def create_physics_engine_adapter(env_id: str) -> IPhysicsEngineAdapter:
        return RemoteAdapter(
            env_id,
            log_service,
            runner_management_service,
            environment_management_service,
        )


runner_management_service.register_physics_adapter_factory(
    ["remote"],
    instantiate_service.create_instance(
        RemoteAdapterFactory,
    ),
)

# end-region
