from __future__ import annotations
from dataclasses import dataclass, field
from uuid import uuid4
from typing import TYPE_CHECKING, Optional, Tuple


if TYPE_CHECKING:
    from tt.sdk.environment_service.common.model.component import (
        MJCFPhysicsComponent,
        URDFPhysicsComponent,
        USDPhysicsComponent,
        RenderingComponent,
    )


class EnvironmentMJCFObjectEntity:
    def __init__(
        self,
        rendering: RenderingComponent,
        physics: MJCFPhysicsComponent,
        name: Optional[str] = None,
    ) -> None:
        self.uuid = str(uuid4())
        self.name = name
        self.physics = physics
        self.rendering = rendering

        self.pos: Tuple[float, float, float] = (0, 0, 0)
        self.quat: Tuple[float, float, float, float] = (1, 0, 0, 0)

    def to_json(self):
        return dict(
            uuid=self.uuid,
            mjcf_uri=self.physics.mjcf_uri,
            pos=self.pos,
            quat=self.quat,
            friction=self.physics.friction,
            solimp=self.physics.solimp,
            solref=self.physics.solref,
            mass=self.physics.mass,
            density=self.physics.density,
        )


# region TODO
class EnvironmentURDFObjectEntity: ...


class EnvironmentUSDObjectEntity: ...


# end-region
