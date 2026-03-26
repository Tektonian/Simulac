from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tt.sdk.environment_service.common.environment_service import (
    EnvironmentManagementService,
)
from tt.sdk.environment_service.common.model.component import (
    MJCFPhysicsComponent,
    RenderingComponent,
    URDFPhysicsComponent,
    USDPhysicsComponent,
)
from tt.sdk.environment_service.common.model.entity import (
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
    EnvironmentMachineEntity,
    EnvironmentObjectEntity,
)
from tt.sdk.log_service.common.log_service import ILogService
from tt.sdk.runner_service.common.runner_service import RunnerManagementService

if TYPE_CHECKING:
    from tt.sdk.environment_service.common.environment import IEnvironment
    from tt.sdk.runner_service.common.runner import IRunner

    Position = tuple[float, float, float]
    Quaternion = tuple[float, float, float, float]
    type WorldEntity = (
        EnvironmentObjectEntity
        | EnvironmentMachineEntity
        | EnvironmentCameraEntity
        | EnvironmentLightEntity
    )
    type PhysicsComponent = (
        MJCFPhysicsComponent | USDPhysicsComponent | URDFPhysicsComponent
    )


class WorldMakerFacade:
    def __init__(
        self,
        LogService: ILogService,
        RunnerManagementService: RunnerManagementService,
        EnvironmentManagementService: EnvironmentManagementService,
    ):
        self.LogService = LogService
        self.RunnerManagementService = RunnerManagementService
        self.EnvironmentManagementService = EnvironmentManagementService

    def create_environment(
        self,
        default_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
        env_uri_or_prebuilt_id: str | None = None,
    ) -> IEnvironment:
        env_ret = self.EnvironmentManagementService.create_environment(default_engine)

        if env_ret[0] is None:
            raise env_ret[1]

        return env_ret[0]

    def create_entity(
        self,
        name: str,
        physics_uri_or_prebuilt_name: str,
        mesh_uri: str,
        texture_uri: str,
    ) -> WorldEntity:
        """_summary_
            TODO:
                1. Handle various name. Expected strings are
                    - Tektonian/cup/cup0 [object with remote owner]
                    - https://tektonian.com/~~ [remote asset]
                    - ./home/mjcf.xml [local asset]
                2. Seperate cases for mjcf, urdf, usd
        Args:
            obj_uri_or_prebuilt_name (str): _description_
        """
        rendering = RenderingComponent(mesh_uri, texture_uri)
        physics = MJCFPhysicsComponent(physics_uri_or_prebuilt_name)
        entity = EnvironmentObjectEntity(rendering, physics, name)

        return entity

    def add_entity(
        self,
        env_id: str,
        entity: WorldEntity,
        *,
        pos: Position = (0, 0, 0),
        quat: Quaternion = (0, 0, 0, 1),
    ) -> str:
        entity.pos = pos
        entity.quat = quat

        env_ret = self.EnvironmentManagementService.get_environment(env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]

        entity_id = ""

        if isinstance(entity, EnvironmentObjectEntity):
            env.objects.append(entity)
            entity_id = f"ent_stu_{len(env.objects)}"
        elif isinstance(entity, EnvironmentMachineEntity):
            env.machines.append(entity)
            entity_id = f"ent_mac_{len(env.objects)}"
        elif isinstance(entity, EnvironmentCameraEntity):
            env.cameras.append(entity)
            entity_id = f"ent_cam_{len(env.objects)}"
        elif isinstance(entity, EnvironmentLightEntity):
            env.lights.append(entity)
            entity_id = f"ent_lig_{len(env.objects)}"

        return entity_id

    def create_runner(self, env_id: str) -> IRunner:

        runner_ret = self.RunnerManagementService.create_runner(env_id)

        if runner_ret[0] is None:
            raise runner_ret[1]

        return runner_ret[0]

    def change_entity_pos(self, entity_id: str, pos: Position) -> None: ...

    def change_entity_quat(self, entity_id: str, quat: Quaternion) -> None: ...
