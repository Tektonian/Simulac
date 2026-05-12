from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.environment_service.common.model.entity import (
    EnvironmentCameraEntity,
    EnvironmentLightEntity,
    EnvironmentMachineEntity,
    EnvironmentStuffEntity,
)
from simulac.sdk.environment_service.common.model.ref import EntityRef, RefBase


@runtime_checkable
class EntityLike(Protocol):
    entity_id: str


@dataclass(frozen=True, slots=True)
class EntityTarget:
    entity_id: str


@dataclass(frozen=True, slots=True)
class RefTarget:
    ref: RefBase


PointConstraintTarget: TypeAlias = str | EntityLike | RefBase
EntityConstraintTarget: TypeAlias = str | EntityLike | EntityRef


@dataclass(frozen=True, slots=True)
class DistanceConstraint:
    a: EntityTarget | RefTarget
    b: EntityTarget | RefTarget
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True, slots=True)
class BBoxConstraint:
    target: EntityTarget | RefTarget
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    mode: Literal["inside", "outside"] = "inside"


@dataclass(frozen=True, slots=True)
class NonpenetrationConstraint:
    entities: tuple[EntityTarget, ...]


SceneConstraint: TypeAlias = (
    DistanceConstraint | BBoxConstraint | NonpenetrationConstraint
)


class Constraint:
    @staticmethod
    def distance(
        a: PointConstraintTarget,
        b: PointConstraintTarget,
        *,
        min: float | None = None,
        max: float | None = None,
    ) -> DistanceConstraint:
        if min is None and max is None:
            raise SimulacBaseError("Constraint.distance requires min or max")

        return DistanceConstraint(
            a=_normalize_point_target(a),
            b=_normalize_point_target(b),
            min=min,
            max=max,
        )

    @staticmethod
    def bbox(
        target: PointConstraintTarget,
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        *,
        mode: Literal["inside", "outside"] = "inside",
    ) -> BBoxConstraint:
        return BBoxConstraint(
            target=_normalize_point_target(target),
            lower=lower,
            upper=upper,
            mode=mode,
        )

    @staticmethod
    def nonpenetration(
        *entities: EntityConstraintTarget,
    ) -> NonpenetrationConstraint:
        if len(entities) < 2:
            raise SimulacBaseError(
                "Constraint.nonpenetration requires at least two entities"
            )

        return NonpenetrationConstraint(
            entities=tuple(_normalize_entity_target(entity) for entity in entities),
        )


def _normalize_point_target(
    target: PointConstraintTarget,
) -> EntityTarget | RefTarget:
    if isinstance(target, str):
        return EntityTarget(target)

    if isinstance(target, RefBase):
        return RefTarget(target)

    if isinstance(
        target,
        (
            EnvironmentStuffEntity,
            EnvironmentMachineEntity,
            EnvironmentLightEntity,
            EnvironmentCameraEntity,
        ),
    ):
        return EntityTarget(target.entity_id)

    raise SimulacBaseError(f"Unsupported constraint target: {target!r}")


def _normalize_entity_target(target: EntityConstraintTarget) -> EntityTarget:
    if isinstance(target, str):
        return EntityTarget(target)

    if isinstance(target, EntityRef):
        return EntityTarget(target.entity_id)

    if isinstance(
        target,
        (
            EnvironmentStuffEntity,
            EnvironmentMachineEntity,
            EnvironmentLightEntity,
            EnvironmentCameraEntity,
        ),
    ):
        return EntityTarget(target.entity_id)

    raise SimulacBaseError(
        f"Constraint.nonpenetration only supports entity targets: {target!r}"
    )
