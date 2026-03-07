from tt.base.instantiate.extensions import (
    register_singleton,
    get_singleton_service_descriptors,
)
from tt.base.instantiate.service_collection import ServiceCollection
from tt.base.instantiate.instantiate_service import InstantiateService
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
from tt.sdk.simulation_service.common.simulation_service import (
    ISimulationManagementService,
    SimulationManagementService,
)
from tt.sdk.world_service.common.world_service import (
    IWorldManagementService,
    WorldManagementService,
)

from tt.sdk.runner_service.local.mujoco_adapter import MujocoAdapter
from tt.sdk.runner_service.local.newton_adapter import NewtonAdapter

register_singleton(ILogService, LogService)
register_singleton(ISimulationManagementService, SimulationManagementService)
register_singleton(IRunnerManagementService, RunnerManagementService)
register_singleton(IWorldManagementService, WorldManagementService)
register_singleton(IEnvironmentManagementService, EnvironmentManagementService)


services = get_singleton_service_descriptors()
services = ServiceCollection(services)

instantiate_service = InstantiateService(services)

runner_management_service: IRunnerManagementService = (
    instantiate_service.service_accessor.get(IRunnerManagementService)
)
log_service: ILogService = instantiate_service.service_accessor.get(ILogService)
environment_management_service: IEnvironmentManagementService = (
    instantiate_service.service_accessor.get(IEnvironmentManagementService)
)


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


runner_management_service.register_physics_adapter(
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
        )


runner_management_service.register_physics_adapter(
    ["newton"],
    instantiate_service.create_instance(
        NewtonAdapterFactory,
    ),
)


simulation_service: ISimulationManagementService = (
    instantiate_service.service_accessor.get(SimulationManagementService)
)
