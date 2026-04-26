from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.environment_service.common.environment_service import (
    IEnvironmentManagementService,
)
from simulac.sdk.environment_service.common.model.component import (
    MJCFPhysicsComponent,
    RenderingComponent,
    URDFPhysicsComponent,
    USDPhysicsComponent,
)
from simulac.sdk.environment_service.common.model.entity import (
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
    EnvironmentMachineEntity,
    EnvironmentStuffEntity,
)
from simulac.sdk.log_service.common.log_service import ILogService
from simulac.sdk.runner_service.common.runner_service import (
    IRunnerManagementService,
)

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.environment import IEnvironment
    from simulac.sdk.runner_service.common.runner import IRunner

    Position = tuple[float, float, float]
    Quaternion = tuple[float, float, float, float]
    type WorldEntity = (
        EnvironmentStuffEntity
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
        RunnerManagementService: IRunnerManagementService,
        EnvironmentManagementService: IEnvironmentManagementService,
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

    def create_stuff_entity(
        self,
        id: str,
        description: str,
        physics_uri_or_prebuilt_name: str,
    ) -> EnvironmentStuffEntity:
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
        # TODO: @gangjeuk
        # handle both cases, file://home/gangjeuk/fanda.xml and https://remote/fanda.xml
        rendering = RenderingComponent(mesh_uri, texture_uri)
        physics = MJCFPhysicsComponent(physics_uri_or_prebuilt_name)
        entity = EnvironmentStuffEntity(None, name, "", rendering, physics)

        return entity

    def create_machine_entity(
        self,
        id: str,
        description: str,
        physics_uri_or_prebuilt_name: str,
    ) -> EnvironmentMachineEntity:
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
        # TODO: @gangjeuk
        # handle both cases, file://home/gangjeuk/fanda.xml and https://remote/fanda.xml
        entity = EnvironmentMachineEntity(None, name, "", physics_uri_or_prebuilt_name)

        return entity

    def create_camera_entity(
        self,
        id: str,
        description: str,
        type: Literal[
            "rgb", "tactile", "depth", "pointcloud", "normal", "segmentation"
        ] = "rgb",
    ):
        entity = EnvironmentCameraEntity(None, name, "", type)
        return entity

    def create_light_entity(
        self,
        id: str,
        description: str,
        type: Literal["ambient", "pointlight", "reactarea", "spot"],
    ):
        entity = EnvironmentLightEntity(None, name, "", type, (0xFF, 0xFF, 0xFF))
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

        if entity.id is not None:
            # Should not reach
            raise SimulacBaseError("Entity should not have id property")

        if isinstance(entity, EnvironmentStuffEntity):
            entity_id = f"ent_stu_{len(env.objects)}"
            entity.id = entity_id
            env.objects.append(entity)
        elif isinstance(entity, EnvironmentMachineEntity):
            entity_id = f"ent_mac_{len(env.machines)}"
            entity.id = entity_id
            env.machines.append(entity)
        elif isinstance(entity, EnvironmentCameraEntity):
            entity_id = f"ent_cam_{len(env.cameras)}"
            entity.id = entity_id
            env.cameras.append(entity)
        elif isinstance(entity, EnvironmentLightEntity):  # pyright: ignore[reportUnnecessaryIsInstance]
            entity_id = f"ent_lig_{len(env.lights)}"
            entity.id = entity_id
            env.lights.append(entity)
        else:
            # Should not reach
            raise SimulacBaseError(f"Unknown environment entity type: {entity}")

        return entity_id

    def create_runner(self, env_id: str) -> IRunner:

        runner_ret = self.RunnerManagementService.create_runner(env_id)

        if runner_ret[0] is None:
            raise runner_ret[1]

        return runner_ret[0]

    def change_entity_pos(self, entity_id: str, pos: Position) -> None: ...

    def change_entity_quat(self, entity_id: str, quat: Quaternion) -> None: ...
