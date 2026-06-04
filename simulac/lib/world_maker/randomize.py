from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Sequence,
    final,
)

from simulac.base.error.error import SimulacBaseError
from simulac.base.types.geometry import Vec3
from simulac.lib.world_maker.object import RobotObject, StuffObject
from simulac.sdk.environment_service.common.model.constraint import (
    BBoxConstraint,
    DistanceConstraint,
    NonpenetrationConstraint,
)
from simulac.sdk.environment_service.common.model.constraint import (
    Constraint as SDKConstraint,
)
from simulac.sdk.environment_service.common.model.ref import RefBase

if TYPE_CHECKING:
    from simulac.base.types.geometry import Vec3
    from simulac.sdk.environment_service.common.model.constraint import (
        BBoxConstraint,
        DistanceConstraint,
        NonpenetrationConstraint,
    )
    from simulac.sdk.environment_service.common.randomize import (
        ChoiceRandomSpec,
        EntryConstraintSpec,
        NormalRandomSpec,
        PlaneConstraintSpec,
        RandomConstraint,
        UniformRandomSpec,
        ValueT,
    )


@final
class Constraint:
    """Helpers for building typed constraint specs."""

    @staticmethod
    def distance(
        a: StuffObject | RobotObject | str,
        b: StuffObject | RobotObject | str,
        *,
        min: float | None = None,
        max: float | None = None,
    ) -> DistanceConstraint:
        return SDKConstraint.distance(
            _to_sdk_target(a), _to_sdk_target(b), min=min, max=max
        )

    @staticmethod
    def nonpenetration(
        *between: StuffObject | RobotObject | str,
    ) -> NonpenetrationConstraint:
        return SDKConstraint.nonpenetration(
            *(_to_entity_id(target) for target in between)
        )

    @staticmethod
    def bbox(
        target: StuffObject | RobotObject | str,
        lower: Vec3,
        upper: Vec3,
        *,
        mode: Literal["inside", "outside"] = "inside",
    ) -> BBoxConstraint:
        return SDKConstraint.bbox(_to_sdk_target(target), lower, upper, mode=mode)

    @staticmethod
    def __plane(
        side: SideType,
        *,
        of: str,
        point: Vec3,
        normal: Vec3,
        targets: Sequence[str],
        align_up_min_dot: float,
    ) -> PlaneConstraintSpec:
        """Constrain targets to one side of a plane.
        TODO: @gangjeuk
        How to check?
        """
        return {
            "type": "plane",
            "side": side,
            "of": of,
            "point": point,
            "normal": normal,
            "targets": list(targets),
            "align_up_min_dot": align_up_min_dot,
        }

    @staticmethod
    def __entry(path: str) -> EntryConstraintSpec:
        """Reference a reusable external constraint entry."""
        return {"type": "entry", "path": path}


def _to_sdk_target(target: RefBase | StuffObject | RobotObject | str) -> str | RefBase:
    if isinstance(target, RefBase):
        return target

    return _to_entity_id(target)


def _to_entity_id(target: StuffObject | RobotObject | str) -> str:
    if isinstance(target, str):
        return target

    if isinstance(target, (StuffObject, RobotObject)):  # pyright: ignore[reportUnnecessaryIsInstance]
        entity_id = target._entity.id
        if entity_id is None:
            raise SimulacBaseError("Entity must be added before using it in Constraint")
        return entity_id

    raise SimulacBaseError(f"Unsupported constraint target: {target!r}")


@final
class Randomize:
    """Helpers for building typed randomization specs.

    Examples:
        Randomize.uniform(0.1, 0.3)
        Randomize.uniform((0.1, 0.2, 0.1), (0.1, 0.2, 0.3))
        Randomize.choice((0.0, 0.0, 0.2), (0.1, 0.0, 0.2))
    """

    @staticmethod
    def uniform(
        min: ValueT,
        max: ValueT,
    ) -> UniformRandomSpec[ValueT]:
        """Create a uniformly sampled spec between `min` and `max`.

        Examples:
            Randomize.uniform(0.1, 0.3)
            Randomize.uniform((0.1, 0.2, 0.1), (0.1, 0.2, 0.3))
        """
        spec: UniformRandomSpec[ValueT] = {
            "type": "uniform",
            "min": min,
            "max": max,
        }
        return spec

    @staticmethod
    def normal(
        mean: ValueT,
        std: ValueT,
        *,
        clip_min: ValueT | None = None,
        clip_max: ValueT | None = None,
    ) -> NormalRandomSpec[ValueT]:
        """Create a normal-distribution spec around `mean`.

        Examples:
            Randomize.normal(60.0, 5.0, clip_min=45.0, clip_max=75.0)
            Randomize.normal((0.0, 0.0, 0.2), (0.05, 0.05, 0.1))
        """
        spec: NormalRandomSpec[ValueT] = {
            "type": "normal",
            "mean": mean,
            "std": std,
        }
        if clip_min is not None:
            spec["clip_min"] = clip_min
        if clip_max is not None:
            spec["clip_max"] = clip_max
        return spec

    @staticmethod
    def choice(
        *values: ValueT,
    ) -> ChoiceRandomSpec[ValueT]:
        """Create a discrete choice spec.

        Examples:
            Randomize.choice("small", "medium", "large")
            Randomize.choice((0.0, 0.0, 0.2), (0.1, 0.0, 0.2))
        """
        spec: ChoiceRandomSpec[ValueT] = {
            "type": "choice",
            "values": list(values),
        }
        return spec
