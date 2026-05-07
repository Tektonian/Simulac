from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from simulac.sdk.environment_service.common.model.ref import (
    RefBase,
)

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.randomize import (
        RandomSpec,
    )


class ResetSampler:
    def __init__(self, seed: int | None) -> None:
        self.rng = random.Random(seed)

    def sample(self, value: RefBase | RandomSpec[Any]) -> Any:
        if isinstance(value, RefBase):
            return value
        elif isinstance(value, tuple):
            return value
        print(value)
        typ = value["type"]
        if typ == "uniform":
            sampled = self._uniform(value["min"], value["max"])
            print(sampled)
            return sampled
        if typ == "normal":
            sampled = self._normal(value["mean"], value["std"])
            if "clip_min" in value:
                sampled = self._max_like(sampled, value["clip_min"])
            if "clip_max" in value:
                sampled = self._min_like(sampled, value["clip_max"])
            print(sampled)
            return sampled
        if typ == "choice":
            return self.rng.choice(value["values"])

        raise SimulacBaseError(f"Unsupported random spec: {value}")

    def _is_random_spec(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        rand_type = value.get("type")
        if rand_type == "uniform" or rand_type == "normal" or rand_type == "choice":
            return True
        return False

    def constraints(self, value: Any) -> list[dict[str, Any]]:
        return list(value.get("constraints", [])) if self._is_random_spec(value) else []

    type RandomInputType = float | list[float] | tuple[float]

    def _uniform(
        self,
        lo: RandomInputType,
        hi: RandomInputType,
    ):
        if isinstance(lo, tuple):
            return tuple(self.rng.uniform(a, b) for a, b in zip(lo, hi))
        return self.rng.uniform(lo, hi)

    def _normal(self, mean: RandomInputType, std: RandomInputType):
        if isinstance(mean, tuple):
            return tuple(self.rng.gauss(m, s) for m, s in zip(mean, std))
        return self.rng.gauss(mean, std)

    def _min_like(self, value: RandomInputType, limit: RandomInputType):
        if isinstance(value, tuple):
            return tuple(min(v, l) for v, l in zip(value, limit))
        return min(value, limit)

    def _max_like(self, value: RandomInputType, limit: RandomInputType):
        if isinstance(value, tuple):
            return tuple(max(v, l) for v, l in zip(value, limit))
        return max(value, limit)
