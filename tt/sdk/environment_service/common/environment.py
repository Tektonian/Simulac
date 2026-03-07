from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit, SplitResult

from tt.base.error.error import TektonianBaseError

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

    objects: list[EnvironmentMJCFObjectEntity | EnvironmentURDFObjectEntity] = field(
        default_factory=list
    )
    proprioception: list[EnvironmentMJCFObjectEntity | EnvironmentURDFObjectEntity] = (
        field(default_factory=list)
    )
    cameras: list[EnvironmentMJCFObjectEntity | EnvironmentURDFObjectEntity] = field(
        default_factory=list
    )
    lights: list[EnvironmentMJCFObjectEntity | EnvironmentURDFObjectEntity] = field(
        default_factory=list
    )

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
