from __future__ import annotations

import random
from typing import TYPE_CHECKING, overload

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.environment_service.common.model.ref import (
    RefBase,
    SupportPointRef,
    SurfaceSampleRef,
    WorldPointRef,
)

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.randomize import RandomSpec


class ResetSampler:
    def __init__(self, seed: int | None) -> None:
        self.rng = random.Random(seed)

    @overload
    def sample(self, value: RefBase) -> RefBase: ...
    @overload
    def sample[T](self, value: tuple[T]) -> tuple[T]: ...
    @overload
    def sample[T](self, value: RandomSpec[T]) -> RandomSpec[T]: ...
    def sample[T](self, value: list[T]) -> list[T]:
        if isinstance(value, RefBase):
            return self.sample_ref(value)

        if isinstance(value, tuple):
            return tuple(self.sample(item) for item in value)

        if isinstance(value, list):
            return [self.sample(item) for item in value]

        if not self._is_random_spec(value):
            return value

        typ = value["type"]

        if typ == "uniform":
            return self._uniform(value["min"], value["max"])

        if typ == "normal":
            sampled = self._normal(value["mean"], value["std"])
            if "clip_min" in value:
                sampled = self._max_like(sampled, value["clip_min"])
            if "clip_max" in value:
                sampled = self._min_like(sampled, value["clip_max"])
            return sampled

        if typ == "choice":
            return self.sample(self.rng.choice(value["values"]))

        raise SimulacBaseError(f"Unsupported random spec: {value}")

    def sample_ref(self, ref: RefBase) -> RefBase:
        if isinstance(ref, SurfaceSampleRef):
            return SurfaceSampleRef(
                entity_id=ref.entity_id,
                collider_name=ref.collider_name,
                side=ref.side,
                margin=float(self.sample(ref.margin)),
            )

        if isinstance(ref, SupportPointRef):
            direction = self.sample(ref.direction)
            return SupportPointRef(
                entity_id=ref.entity_id,
                collider_name=ref.collider_name,
                direction=(
                    float(direction[0]),
                    float(direction[1]),
                    float(direction[2]),
                ),
                frame=ref.frame,
            )

        if isinstance(ref, WorldPointRef):
            pos = self.sample(ref.pos)
            return WorldPointRef(
                pos=(
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                )
            )

        return ref

    def _is_random_spec(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return value.get("type") in {"uniform", "normal", "choice"}

    def constraints(self, value: Any) -> list[dict[str, Any]]:
        return list(value.get("constraints", [])) if self._is_random_spec(value) else []

    type RandomInputType = float | list[float] | tuple[float, ...]

    def _uniform(
        self,
        lo: RandomInputType,
        hi: RandomInputType,
    ) -> RandomInputType:
        if isinstance(lo, tuple):
            return tuple(self.rng.uniform(float(a), float(b)) for a, b in zip(lo, hi))
        if isinstance(lo, list):
            return [self.rng.uniform(float(a), float(b)) for a, b in zip(lo, hi)]
        return self.rng.uniform(float(lo), float(hi))

    def _normal(
        self,
        mean: RandomInputType,
        std: RandomInputType,
    ) -> RandomInputType:
        if isinstance(mean, tuple):
            return tuple(self.rng.gauss(float(m), float(s)) for m, s in zip(mean, std))
        if isinstance(mean, list):
            return [self.rng.gauss(float(m), float(s)) for m, s in zip(mean, std)]
        return self.rng.gauss(float(mean), float(std))

    def _min_like(
        self,
        value: RandomInputType,
        limit: RandomInputType,
    ) -> RandomInputType:
        if isinstance(value, tuple):
            return tuple(min(float(v), float(l)) for v, l in zip(value, limit))
        if isinstance(value, list):
            return [min(float(v), float(l)) for v, l in zip(value, limit)]
        return min(float(value), float(limit))

    def _max_like(
        self,
        value: RandomInputType,
        limit: RandomInputType,
    ) -> RandomInputType:
        if isinstance(value, tuple):
            return tuple(max(float(v), float(l)) for v, l in zip(value, limit))
        if isinstance(value, list):
            return [max(float(v), float(l)) for v, l in zip(value, limit)]
        return max(float(value), float(limit))
