from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal


@dataclass(slots=True)
class MujocoStuffBinding:
    entity_id: str
    root_body_id: int
    body_ids: list[int] = field(default_factory=list[int])
    geom_ids: list[int] = field(default_factory=list[int])
    site_ids: list[int] = field(default_factory=list[int])
    joint_ids: list[int] = field(default_factory=list[int])
    actuator_ids: list[int] = field(default_factory=list[int])
    geoms: dict[str, "MujocoGeomBinding"] = field(default_factory=dict)
    sites: dict[str, "MujocoSiteBinding"] = field(default_factory=dict)
    sensors: dict[str, "MujocoSensorBinding"] = field(default_factory=dict)
    root_freejoint_id: int = -1
    mocap_id: int = -1
    joints: dict[str, "MujocoJointBinding"] = field(default_factory=dict)


@dataclass(slots=True)
class MujocoRobotBinding:
    entity_id: str
    name: str
    full_name: str

    root_body_id: int
    root_body_name: str
    root_body_full_name: str

    body_ids: list[int] = field(default_factory=list)
    geom_ids: list[int] = field(default_factory=list)
    site_ids: list[int] = field(default_factory=list)
    joint_ids: list[int] = field(default_factory=list)
    actuator_ids: list[int] = field(default_factory=list)

    links: dict[str, "MujocoLinkBinding"] = field(default_factory=dict)
    geoms: dict[str, "MujocoGeomBinding"] = field(default_factory=dict)
    sites: dict[str, "MujocoSiteBinding"] = field(default_factory=dict)
    joints: dict[str, "MujocoJointBinding"] = field(default_factory=dict)
    actuators: dict[str, "MujocoActuatorBinding"] = field(default_factory=dict)
    sensors: dict[str, "MujocoSensorBinding"] = field(default_factory=dict)

    root_freejoint_id: int = -1
    mocap_id: int = -1


@dataclass(slots=True)
class MujocoLinkBinding:
    entity_id: str
    full_name: str
    name: str
    body_id: int

    parent_body_id: int
    child_body_ids: list[int] = field(default_factory=list)

    geom_ids: list[int] = field(default_factory=list)
    joint_ids: list[int] = field(default_factory=list)

    mocap_id: int = -1


@dataclass(slots=True)
class MujocoJointBinding:
    entity_id: str
    full_name: str
    name: str
    joint_id: int
    body_id: int
    joint_type: int

    qpos_addr: int
    qvel_addr: int
    qpos_dim: int
    qvel_dim: int

    axis: tuple[float, float, float]
    range: tuple[float, float] | None = None
    limited: bool = False

    actuator_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class MujocoActuatorBinding:
    entity_id: str
    full_name: str
    name: str
    actuator_id: int

    target_type: Literal["joint", "tendon", "site", "body", "unknown"]
    target_id: int | None
    target_name: str | None

    ctrl_range: tuple[float, float] | None = None
    force_range: tuple[float, float] | None = None
    act_range: tuple[float, float] | None = None

    group: int = 0


@dataclass(slots=True)
class MujocoGeomBinding:
    entity_id: str
    full_name: str
    name: str
    geom_id: int
    body_id: int


@dataclass(slots=True)
class MujocoSiteBinding:
    entity_id: str
    full_name: str
    name: str
    site_id: int
    body_id: int


@dataclass(slots=True)
class MujocoSensorBinding:
    entity_id: str
    full_name: str
    name: str
    sensor_id: int
    sensor_type: int
    obj_type: int
    obj_id: int
    adr: int
    dim: int


@dataclass(slots=True)
class MujocoCameraBinding:
    entity_id: str

    camera_id: int
    camera_name: str
    camera_full_name: str

    # below four are for attached or following cameras
    root_body_id: int | None
    root_body_name: str | None
    root_body_full_name: str | None
    mocap_id: int = -1


@dataclass(slots=True)
class MujocoLightBinding:
    entity_id: str

    light_id: int
    light_name: str
    light_full_name: str

    root_body_id: int
    root_body_name: str
    root_body_full_name: str

    light_type: str
    mocap_id: int = -1
