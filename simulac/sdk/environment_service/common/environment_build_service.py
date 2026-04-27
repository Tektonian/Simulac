from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, NoReturn, Tuple, Union
from urllib.parse import SplitResult

from simulac.base.error.error import SimulacBaseError
from simulac.base.instantiate.instantiate import ServiceIdentifier, service_identifier
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

if TYPE_CHECKING:
    import urllib.parse

    from simulac.sdk.environment_service.common.environment import IEnvironment
    from simulac.sdk.environment_service.common.model.component import (
        MJCFPhysicsComponent,
        URDFPhysicsComponent,
        USDPhysicsComponent,
    )
    from simulac.sdk.environment_service.common.randomize import RandomizableVec3

type EnvironmentEntityType = Union[
    EnvironmentStuffEntity,
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
        entity_id: str | None = None,
        pos: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
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
    def __init__(
        self,
        EnvironmentManagementService: IEnvironmentManagementService,
        LogService: ILogService,
    ) -> None:
        self.EnvironmentManagementService = EnvironmentManagementService
        self.LogService = LogService

        self._entities_by_env: defaultdict[
            str,
            list[EnvironmentEntityType],
        ] = defaultdict(list)

        self._entities_by_id: dict[
            str,
            EnvironmentEntityType,
        ] = {}

    def add_entity(
        self,
        env_id: str,
        entity: EnvironmentEntityType,
        entity_id: str | None = None,
        pos: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
    ):
        """TODO: only support local file for now
        1. add remote support http, https
        2. [o] - handle each entity type (2026/04/26)
        3. need mjcf, urdf, usd file parser
        4. compatibility check
        """
        env = self.__get_env(env_id)
        env_entities = self._entities_by_env[env.id]
        new_entity_id = entity_id or self.__gen_entity_id(env, entity)

        # Set entity id
        if entity.id is not None:
            self.LogService.error(
                "\n".join(
                    [
                        "Expected not registered entity, but we got already registered entity",
                        f"{self.__entity_debug_context(self.__entity_kind(entity), entity)}",
                    ]
                )
            )
        entity.id = new_entity_id
        entity.pos = pos
        entity.rot = rot

        self.__append_entity_to_env(env, entity)
        env_entities.append(entity)
        self._entities_by_id[new_entity_id] = entity

        return new_entity_id

    def change_pos(self, entity_id: str, pos: Tuple[float, float, float]):
        self.__get_entity(entity_id).pos = pos

    def change_quat(self, entity_id: str, quat: Tuple[float, float, float, float]):
        self.__get_entity(entity_id).quat = quat

    def __get_env(self, env_id: str) -> IEnvironment:
        env_ret = self.EnvironmentManagementService.get_environment(env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        return env_ret[0]

    def __get_entity(self, entity_id: str) -> EnvironmentEntityType:
        entity = self._entities_by_id.get(entity_id)

        if entity is None:
            known_ids = ", ".join(sorted(self._entities_by_id)) or "<none>"
            raise SimulacBaseError(
                "\n".join(
                    [
                        f"No entity id: {entity_id!r}",
                        f"Known entity ids: {known_ids}",
                    ]
                )
            )

        return entity

    def __gen_entity_id(self, env: IEnvironment, entity: EnvironmentEntityType):
        if isinstance(entity, EnvironmentStuffEntity):
            return f"stuff_{len(env.stuffs)}"
        elif isinstance(entity, EnvironmentMachineEntity):
            return f"machine_{len(env.machines)}"
        elif isinstance(entity, EnvironmentCameraEntity):
            return f"camera_{len(env.cameras)}"
        elif isinstance(entity, EnvironmentLightEntity):  # pyright: ignore[reportUnnecessaryIsInstance]
            return f"light_{len(env.lights)}"

        # Should not reach
        raise SimulacBaseError(f"Unsupported entity type: {type(entity).__name__}")

    def __append_entity_to_env(
        self, env: IEnvironment, entity: EnvironmentEntityType
    ) -> None:
        entity_kind = self.__entity_kind(entity)

        if entity.id is None:
            self.__raise_entity_append_error(
                env,
                entity,
                entity_kind,
                f"Cannot append {type(entity).__name__}: entity id is missing.",
            )

        duplicated_entity = self.__find_entity_by_id(env, entity.id)
        if duplicated_entity is not None:
            duplicated_kind, duplicated = duplicated_entity
            self.__raise_entity_append_error(
                env,
                entity,
                entity_kind,
                (
                    f"Cannot append {type(entity).__name__}: duplicated entity id "
                    f"{entity.id!r} in environment {env.id!r}."
                ),
                duplicated_entity=duplicated,
                duplicated_kind=duplicated_kind,
            )

        if isinstance(entity, EnvironmentStuffEntity):
            env.stuffs.append(entity)
        elif isinstance(entity, EnvironmentMachineEntity):
            env.machines.append(entity)
        elif isinstance(entity, EnvironmentCameraEntity):
            env.cameras.append(entity)
        elif isinstance(entity, EnvironmentLightEntity):  # pyright: ignore[reportUnnecessaryIsInstance]
            env.lights.append(entity)
        else:
            # Should not reach
            self.__raise_entity_append_error(
                env,
                entity,
                entity_kind,
                f"Unsupported entity type: {type(entity).__name__}",
            )

    def __find_entity_by_id(
        self, env: IEnvironment, entity_id: str
    ) -> tuple[str, EnvironmentEntityType] | None:
        for kind, entity in self.__iter_env_entities(env):
            if entity.id == entity_id:
                return kind, entity

        return None

    def __iter_env_entities(
        self, env: IEnvironment
    ) -> Iterable[tuple[str, EnvironmentEntityType]]:
        for entity in env.stuffs:
            yield "stuff", entity

        for entity in env.machines:
            yield "machine", entity

        for entity in env.cameras:
            yield "camera", entity

        for entity in env.lights:
            yield "light", entity

    def __entity_kind(self, entity: EnvironmentEntityType) -> str:
        if isinstance(entity, EnvironmentStuffEntity):
            return "stuff"
        if isinstance(entity, EnvironmentMachineEntity):
            return "machine"
        if isinstance(entity, EnvironmentCameraEntity):
            return "camera"
        if isinstance(entity, EnvironmentLightEntity):  # pyright: ignore[reportUnnecessaryIsInstance] # pyright[]
            return "light"

        return "unknown"

    # region Pretty debug message
    def __raise_entity_append_error(
        self,
        env: IEnvironment,
        entity: EnvironmentEntityType,
        entity_kind: str,
        reason: str,
        *,
        duplicated_entity: EnvironmentEntityType | None = None,
        duplicated_kind: str | None = None,
    ) -> NoReturn:
        lines = [
            reason,
            f"Environment: {env.id!r}",
            "Incoming entity:",
            f"- {self.__entity_debug_line(entity_kind, entity)}",
        ]

        if duplicated_entity is not None and duplicated_kind is not None:
            lines.extend(
                [
                    "Duplicated entity already in this environment:",
                    f"- {self.__entity_debug_line(duplicated_kind, duplicated_entity)}",
                ]
            )

        lines.extend(
            [
                f"Current entities in environment {env.id!r}:",
                self.__env_entities_debug_lines(env),
            ]
        )

        raise SimulacBaseError(
            "\n".join(lines),
            context={
                "env_id": env.id,
                "incoming_entity": self.__entity_debug_context(entity_kind, entity),
                "duplicated_entity": (
                    self.__entity_debug_context(duplicated_kind, duplicated_entity)
                    if duplicated_entity is not None and duplicated_kind is not None
                    else None
                ),
                "current_entities": [
                    self.__entity_debug_context(kind, current_entity)
                    for kind, current_entity in self.__iter_env_entities(env)
                ],
            },
        )

    def __env_entities_debug_lines(self, env: IEnvironment) -> str:
        lines = [
            f"- {self.__entity_debug_line(kind, entity)}"
            for kind, entity in self.__iter_env_entities(env)
        ]

        return "\n".join(lines) if lines else "- <none>"

    def __entity_debug_line(self, kind: str, entity: EnvironmentEntityType) -> str:
        debug_info = self.__entity_debug_context(kind, entity)
        return ", ".join(f"{key}={value!r}" for key, value in debug_info.items())

    def __entity_debug_context(
        self, kind: str, entity: EnvironmentEntityType
    ) -> dict[str, object]:
        debug_info: dict[str, object] = {
            "kind": kind,
            "id": entity.id,
            "type": type(entity).__name__,
            "description": entity.description,
            "pos": entity.pos,
            "quat": entity.quat,
        }

        if isinstance(entity, EnvironmentStuffEntity):
            debug_info["size"] = entity.size
            debug_info["fixed"] = entity.fixed
        elif isinstance(entity, EnvironmentCameraEntity):
            debug_info["spec"] = entity.spec
        elif isinstance(entity, EnvironmentLightEntity):
            debug_info["spec"] = entity.spec

        return debug_info

    # end-region
    def __compatibility_check(
        self,
        target_physics: Literal["mujoco", "newton", "genesis"],
    ) -> bool:
        """Check compatibility between assets and physics engine

        Compatibility table

        | Type | Mujoco | Newton | Genesis |
        |:-----|:-----:|:-----:|:-----:|
        | MJCF | O | O (with add_mjcf()) | O (with morphs.MJCF) |
        | USDF | △ (requires `<mujoco/>`) | O (with add_urdf()) | O (with morphs.URDF) |
        | USD  | X | O (with add_usd()) | X |

        """

        return True
