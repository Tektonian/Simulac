from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple
from uuid import uuid4

if TYPE_CHECKING:
    from tt.sdk.environment_service.common.model.component import (
        MJCFPhysicsComponent,
        RenderingComponent,
        URDFPhysicsComponent,
        USDPhysicsComponent,
    )


class EnvironmentCameraEntity:
    def __init__(
        self,
        name: str,
        description: str,
        type: Literal[
            "rgb", "tactile", "depth", "pointcloud", "normal", "segmentation"
        ] = "rgb",
        mode: Literal["fixed", "track"] = "track",
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        lookat: Tuple[float, float, float] = (0, 0, 0),
        fov: float = 50.0,
        aspect: float = 1.0,
        near: float = 100.0,
        far: float = 1000.0,
    ) -> None:
        self.name = name
        self.description = description
        self.type = type
        self.mode = mode
        self.pos = pos
        self.quat = quat
        self.lookat = lookat
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far


class EnvironmentMachineEntity:
    def __init__(
        self,
        name: str,
        description: str,
        uri: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> None:
        self.name = name
        self.description = description
        self.uri = uri
        self.pos = pos
        self.quat = quat

        self.init_position: List[float] | None = None
        self.action_min: List[float] | None = None
        self.action_max: List[float] | None = None


class EnvironmentLightEntity:
    def __init__(
        self,
        type: Literal["ambient", "pointlight", "reactarea", "spot"],
        color: str,
        intensity: float = 0.8,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> None:
        self.type = type
        self.color = color
        self.intensity = intensity
        self.pos = pos
        self.quat = quat


class EnvironmentObjectEntity:
    def __init__(
        self,
        rendering: RenderingComponent,
        physics: MJCFPhysicsComponent | URDFPhysicsComponent | USDPhysicsComponent,
        name: Optional[str] = None,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        size: Tuple[float, float, float] = (1, 1, 1),
        fixed: bool = True,
    ) -> None:
        self.uuid = str(uuid4())
        self.name = name
        self.physics = physics
        self.rendering = rendering

        self.pos = pos
        self.quat = quat
        self.size = size
        self.fixed = fixed

    def to_dict(self):
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
