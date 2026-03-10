from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, SplitResult

from tt.base.error.error import TektonianBaseError
from tt.sdk.environment_service.common.model.entity import (
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
    EnvironmentMachineEntity,
    EnvironmentObjectEntity,
)

if TYPE_CHECKING:
    from tt.sdk.environment_service.common.model.entity import (
        EnvironmentMJCFObjectEntity,
        EnvironmentURDFObjectEntity,
    )


# https://gymnasium.farama.org/api/env/
# https://docs.wandb.ai/models/ref/python/experiments/run#property-run-entity
@dataclass
class IEnvironment(ABC):
    id: str
    world_id: str
    env_json_uri: str | SplitResult
    act_json_uri: str | SplitResult
    obs_json_uri: str | SplitResult

    physics_engine: Literal["mujoco", "newton", "genesis", "remote"]
    solver: Literal[""]  # TODO: add later

    benchmark_specific_args: dict[str, Any]

    objects: list[EnvironmentObjectEntity] = field(default_factory=list)
    cameras: list[EnvironmentCameraEntity] = field(default_factory=list)
    lights: list[EnvironmentLightEntity] = field(default_factory=list)
    machines: list[EnvironmentMachineEntity] = field(default_factory=list)
    # region TODO: implment later
    particles: list[Any] = field(default_factory=list)
    soft_bodies: list[Any] = field(default_factory=list)
    randoms: list[dict[str, dict[str, dict[str, str]]]] = field(default_factory=list)
    constraints: list[dict[str, str]] = field(default_factory=list)

    # end-region
    @abstractmethod
    def snapshop(self):
        pass

    @abstractmethod
    def load_env(self):
        pass

    def __post_init__(self):
        url = (
            urlsplit(self.env_json_uri)
            if isinstance(self.env_json_uri, str)
            else self.env_json_uri
        )

        if url.scheme not in ["http", "https", "file", "memory"]:
            raise TektonianBaseError("not valid schema")
