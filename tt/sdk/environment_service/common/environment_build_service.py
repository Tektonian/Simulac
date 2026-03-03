from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Literal, NewType, Union

from tt.base.instantiate.instantiate import ServiceIdentifier, service_identifier

if TYPE_CHECKING:
    from tt.sdk.environment_service.common.model.component import (
        MJCFPhysicsComponent,
        URDFPhysicsComponent,
        USDPhysicsComponent,
    )


@service_identifier("IEnvironmentBuildService")
class IEnvironmentBuildService(ServiceIdentifier):

    @abstractmethod
    def add_entity(
        self, entity: MJCFPhysicsComponent | URDFPhysicsComponent | USDPhysicsComponent
    ): ...

    @abstractmethod
    def remove_entity(self, entity_id: str): ...

    @abstractmethod
    def to_json(self) -> str: ...


class EnvironmentBuildService(IEnvironmentBuildService):

    def __init__(self) -> None:
        self.entitys: (
            List[MJCFPhysicsComponent]
            | List[URDFPhysicsComponent]
            | List[USDPhysicsComponent]
        ) = []

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
