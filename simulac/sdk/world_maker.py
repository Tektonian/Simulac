from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.environment_service.common.environment_build_service import (
    IEnvironmentBuildService,
)
from simulac.sdk.environment_service.common.environment_service import (
    IEnvironmentManagementService,
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
    from simulac.sdk.environment_service.common.model.entity import (
        CameraSpec,
        LightSpec,
    )
    from simulac.sdk.environment_service.common.randomize import (
        RandomizableVec3,
    )
    from simulac.sdk.runner_service.common.runner import IRunner

    WorldEntity = (
        EnvironmentStuffEntity
        | EnvironmentMachineEntity
        | EnvironmentCameraEntity
        | EnvironmentLightEntity
    )


class WorldMakerFacade:
    def __init__(
        self,
        LogService: ILogService,
        RunnerManagementService: IRunnerManagementService,
        EnvironmentManagementService: IEnvironmentManagementService,
        EnvironmentBuildService: IEnvironmentBuildService,
    ):
        self.LogService = LogService
        self.RunnerManagementService = RunnerManagementService
        self.EnvironmentManagementService = EnvironmentManagementService
        self.EnvironmentBuildService = EnvironmentBuildService

    def create_environment(
        self,
        default_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
        env_uri_or_prebuilt_id: str | None = None,
    ) -> IEnvironment:
        # TODO: @gangjeuk
        # handle pre built env `env_uri_or_prebuilt_id`
        env_ret = self.EnvironmentManagementService.create_environment(default_engine)

        if env_ret[0] is None:
            raise env_ret[1]

        return env_ret[0]

    def create_stuff_entity(
        self,
        asset_uri_or_prebuilt_name: str,
        *,
        description: str = "",
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
        entity = EnvironmentStuffEntity(None, description, asset_uri_or_prebuilt_name)

        return entity

    def create_machine_entity(
        self,
        asset_uri_or_prebuilt_name: str,
        *,
        description: str = "",
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
        entity = EnvironmentMachineEntity(None, description, asset_uri_or_prebuilt_name)

        return entity

    def create_camera_entity(
        self,
        spec: CameraSpec,
        *,
        description: str,
    ):
        entity = EnvironmentCameraEntity(
            None,
            description,
            spec,
        )
        return entity

    def create_light_entity(
        self,
        spec: LightSpec,
        *,
        description: str = "",
    ):
        entity = EnvironmentLightEntity(None, description, spec=spec)
        return entity

    def add_entity(
        self,
        env_id: str,
        entity: WorldEntity,
        entity_id: str | None = None,
        pos: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
    ) -> str:
        env_ret = self.EnvironmentManagementService.get_environment(env_id)
        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]

        return self.EnvironmentBuildService.add_entity(
            env.id, entity, entity_id, pos, rot
        )

    def create_runner(self, env_id: str) -> IRunner:

        runner_ret = self.RunnerManagementService.create_runner(env_id)

        if runner_ret[0] is None:
            raise runner_ret[1]

        return runner_ret[0]

    def change_entity_pos(self, entity_id: str, pos: Position) -> None: ...

    def change_entity_quat(self, entity_id: str, quat: Quaternion) -> None: ...
