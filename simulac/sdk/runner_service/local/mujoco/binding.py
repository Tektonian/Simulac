from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.randomize import RandomizableVec3


@dataclass(slots=True)
class MujocoEntityBinding:
    entity_id: str
    kind: Literal["stuff", "machine", "camera", "light"]
    root_body_id: int
    pos: RandomizableVec3
    rot: RandomizableVec3
    body_ids: list[int] = field(default_factory=list[int])
    geom_ids: list[int] = field(default_factory=list[int])
    joint_ids: list[int] = field(default_factory=list[int])
    actuator_ids: list[int] = field(default_factory=list[int])
    root_freejoint_id: int = -1
    mocap_id: int = -1
