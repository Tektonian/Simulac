from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, List, MutableMapping, Tuple, TypeVar, Union
from urllib.parse import SplitResult

from tt.base.error.error import TektonianBaseError
from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier
from tt.sdk.environment_service.common.environment_service import (
    IEnvironmentManagementService,
)
from tt.sdk.environment_service.common.model.entity import (
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
    EnvironmentMachineEntity,
    EnvironmentObjectEntity,
)
from tt.sdk.log_service.common.log_service import ILogService

if TYPE_CHECKING:
    import urllib.parse

    from tt.sdk.environment_service.common.environment import IEnvironment
    from tt.sdk.environment_service.common.model.component import (
        MJCFPhysicsComponent,
        URDFPhysicsComponent,
        USDPhysicsComponent,
    )

type EnvironmentEntityType = Union[
    EnvironmentObjectEntity,
    EnvironmentMachineEntity,
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
]


@service_identifier("IEnvironmentBuildService")
class IEnvironmentBuildService(ServiceIdentifier["IEnvironmentBuildService"]):
    @abstractmethod
    def add_entity(
        self,
        env_id: str,
        entity: EnvironmentEntityType,
    ) -> str: ...

    @abstractmethod
    def remove_entity(self, entity_id: str): ...

    @abstractmethod
    def replace_entity(self, entity_id: str, render_uri: str, physics_uri: str): ...

    @abstractmethod
    def change_pos(self, entity_id: str, pos: Tuple[float, float, float]): ...

    @abstractmethod
    def change_quat(self, entity_id: str, quat: Tuple[float, float, float, float]): ...

    @abstractmethod
    def export_env_json(self) -> str: ...

    @abstractmethod
    def export_act_json(self) -> str: ...

    @abstractmethod
    def export_obs_json(self) -> str: ...

    @abstractmethod
    def import_env(self, dir_path: str) -> None:
        """Import entire prebuilt environment
        Expect directory like below
            .
            └── dir_path/
                ├── env.json  <- environment schema
                ├── obs.json  <- observation schema
                ├── act.json  <- action schema
                └── random.py <- for reset logic

        Args:
            dir_path (str): directory path
        """

    @abstractmethod
    def __load_mjcf(self, path: str) -> str: ...

    @abstractmethod
    def __load_urdf(self, path: str) -> str: ...

    @abstractmethod
    def __load_usd(self, path: str) -> str: ...

    @abstractmethod
    def build_env(self) -> IEnvironment: ...


class EnvironmentBuildService(IEnvironmentBuildService):

    __ID_PREFIX = "thg_"

    def __init__(
        self,
        EnvironmentManagementService: IEnvironmentManagementService,
        LogService: ILogService,
    ) -> None:
        self.EnvironmentManagementService = EnvironmentManagementService
        self.LogService = LogService

        self.env_entities_map: MutableMapping[
            str,
            List[EnvironmentEntityType],
        ] = {}

        self.entities_map: MutableMapping[
            str,
            EnvironmentEntityType,
        ] = {}

    def add_entity(
        self,
        env_id: str,
        entity: EnvironmentEntityType,
    ):
        """TODO: only support local file for now
        1. add remote support http, https
        2. handle each entity type
        3. need mjcf, urdf, usd file parser
        4. compatibility check
        """
        env = self.__get_env(env_id)
        new_entity_id = (
            f"{self.__ID_PREFIX}{len(self.env_entities_map.get(env.id, []))}"
        )

        # region TODO: Remove entity copy code here
        entity = deepcopy(entity)
        entity.uuid = new_entity_id

        print(entity, entity.uuid)
        # end-region

        if env.id not in self.env_entities_map.keys():
            self.env_entities_map[env.id] = []

        if isinstance(entity, EnvironmentObjectEntity):
            env.objects.append(entity)
            self.LogService.debug(
                f"Environment {env.id} append Object Enitty {new_entity_id}"
            )
        elif isinstance(entity, EnvironmentMachineEntity):
            env.machines.append(entity)
            self.LogService.debug(
                f"Environment {env.id} append Machine Enitty {new_entity_id}"
            )

        self.env_entities_map[env.id].append(entity)
        self.entities_map[new_entity_id] = entity

        return new_entity_id

    def change_pos(self, entity_id: str, pos: Tuple[float, float, float]):
        entity = self.entities_map.get(entity_id, None)

        if entity is None:
            raise TektonianBaseError(f"No entity id {entity_id}")

        entity.pos = pos

    def change_quat(self, entity_id: str, quat: Tuple[float, float, float, float]):
        entity = self.entities_map.get(entity_id, None)

        if entity is None:
            raise TektonianBaseError(f"No entity id {entity_id}")

        entity.quat = quat

    def __get_env(self, env_id: str) -> IEnvironment:
        env_ret = self.EnvironmentManagementService.get_environment(env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        return env_ret[0]

    def __compatibility_check(
        self,
        target_physics: Literal["mujoco", "newton", "genesis"],
    ) -> bool:
        """Check compatibility between assets and physics engine

        Compatibility table

        | Type | Mujoco | Newton | Genesis |
        |:-----|:-----:|:-----:|:-----:|
        | MJCF | O | O (with add_mjcf()) | O (with morphs.MJCF) |
        | USDF | △ (need add `<mujoco/>` tag in .usdf file) | O (with add_urdf()) | O (with morphs.URDF) |
        | USD  | X | O (with add_usd()) | X |

        """

        return True
