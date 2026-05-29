from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.asset_service.common.asset_service import IAssetService
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
        AssetService: IAssetService,
    ):
        self.LogService = LogService
        self.RunnerManagementService = RunnerManagementService
        self.EnvironmentManagementService = EnvironmentManagementService
        self.EnvironmentBuildService = EnvironmentBuildService
        self.AssetService = AssetService

    def create_environment(
        self,
        default_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
        env_uri_or_prebuilt_id: str | None = None,
    ) -> IEnvironment:
        if env_uri_or_prebuilt_id is not None:
            source = Path(env_uri_or_prebuilt_id)
            if not source.exists():
                raise SimulacBaseError(
                    f"Unsupported environment source: {env_uri_or_prebuilt_id}"
                )
            return self.EnvironmentManagementService.load_env(source)

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
        pos: RandomizableVec3 | PointRefType = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        *,
        fixed: bool | None = None,
    ) -> str:
        env_ret = self.EnvironmentManagementService.get_environment(env_id)
        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]

        return self.EnvironmentBuildService.add_entity(
            env.id, entity, entity_id, pos, rot, fixed=fixed
        )

    def create_runner(self, env_id: str) -> IRunner:
        env_ret = self.EnvironmentManagementService.get_environment(env_id)
        if env_ret[0] is None:
            raise env_ret[1]

        self.AssetService.resolve_environment_assets(env_ret[0])

        runner_ret = self.RunnerManagementService.create_runner(env_id)
        if runner_ret[0] is None:
            raise runner_ret[1]

        return runner_ret[0]

    def dump_env(
        self,
        env_id: str,
        *,
        include_resolved_assets: bool = False,
        include_runtime_state: bool = False,
        validation: Literal["none", "warn", "raise"] = "warn",
    ) -> dict[str, Any]:
        return self.EnvironmentManagementService.dump_env(
            env_id,
            include_resolved_assets=include_resolved_assets,
            include_runtime_state=include_runtime_state,
            validation=validation,
        )

    def dump_env_json(
        self,
        env_id: str,
        *,
        indent: int = 2,
        include_resolved_assets: bool = False,
        include_runtime_state: bool = False,
        validation: Literal["none", "warn", "raise"] = "warn",
    ) -> str:
        return self.EnvironmentManagementService.dump_env_json(
            env_id,
            indent=indent,
            include_resolved_assets=include_resolved_assets,
            include_runtime_state=include_runtime_state,
            validation=validation,
        )

    def save_env(
        self,
        env_id: str,
        path: str | Path,
        *,
        overwrite: bool = False,
        indent: int = 2,
        include_resolved_assets: bool = False,
        include_runtime_state: bool = False,
        validation: Literal["none", "warn", "raise"] = "warn",
    ) -> Path:
        output_path = Path(path)
        if output_path.exists() and not overwrite:
            raise SimulacBaseError(f"Environment dump already exists: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.dump_env_json(
                env_id,
                indent=indent,
                include_resolved_assets=include_resolved_assets,
                include_runtime_state=include_runtime_state,
                validation=validation,
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path
