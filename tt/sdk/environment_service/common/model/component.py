from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class RenderingComponent:
    mesh_uri: str
    texture_uri: Optional[str] = None
    color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    scale: Tuple[float, float, float] = (1, 1, 1)

    # additional field for web rendering
    material_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MJCFPhysicsComponent:
    mjcf_uri: str
    friction: Tuple[float, float, float] = (1, 0.005, 0.0001)
    solimp: Tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2)
    solref: Tuple[float, float] = (0.02, 1)
    mass: Optional[float] = None
    density: float = 1000


@dataclass
class URDFPhysicsComponent:
    urdf_uri: str


@dataclass
class USDPhysicsComponent:
    usd_uri: str
