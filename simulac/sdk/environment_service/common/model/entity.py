from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.model.component import (
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
        id: str,
        description: str,
        uri: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> None:
        self.id = id
        self.name = name
        self.description = description
        self.uri = uri
        self.pos = pos
        self.quat = quat

        # Should have same length
        self.init_position: list[float] | None = None
        self.action_min: list[float] | None = None
        self.action_max: list[float] | None = None


class EnvironmentLightEntity:
    def __init__(
        self,
        id: str,
        description: str,
        type: Literal["ambient", "pointlight", "reactarea", "spot"],
        color: Tuple[int, int, int],
        intensity: float = 0.8,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> None:
        self.id = id
        self.description = description
        self.type = type
        self.color = color
        self.intensity = intensity
        self.pos = pos
        self.quat = quat


class EnvironmentStuffEntity:
    def __init__(
        self,
        id: str,
        description: str,
        rendering: RenderingComponent,
        physics: MJCFPhysicsComponent | URDFPhysicsComponent | USDPhysicsComponent,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        size: Tuple[float, float, float] = (1, 1, 1),
        fixed: bool = True,
    ) -> None:
        self.id = id
        self.description = description
        self.physics = physics
        self.rendering = rendering

        self.pos = pos
        self.quat = quat
        self.size = size
        self.fixed = fixed

    def to_dict(self):
        return dict(
            id=self.id,
            mjcf_uri=self.physics.mjcf_uri,
            pos=self.pos,
            quat=self.quat,
            friction=self.physics.friction,
            solimp=self.physics.solimp,
            solref=self.physics.solref,
            mass=self.physics.mass,
            density=self.physics.density,
        )
