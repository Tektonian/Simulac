from __future__ import annotations

from typing import (
    Generic,
    Literal,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    final,
)

from typing_extensions import Required, TypeAlias

ValueT = TypeVar("ValueT")

type Vec3 = tuple[float, float, float]
type Quat = tuple[float, float, float, float]
type ColorRgb = tuple[int, int, int]

type SideType = Literal["above", "on", "below", "under"]
type BetweenType = tuple[str, str]

class NonpenetrationConstraintSpec(TypedDict):
    type: Literal["nonpenetration"]
    between: list[str]


class BboxConstraintSpec(TypedDict, total=False):
    type: Required[Literal["bbox"]]
    min: Required[Vec3]
    max: Required[Vec3]
    mode: Required[Literal["inside", "outside"]]
    center: Vec3


class DistanceConstraintSpec(TypedDict):
    type: Literal["distance"]
    between: list[str]
    min: float
    max: float


class PlaneConstraintSpec(TypedDict):
    type: Literal["plane"]
    side: SideType
    of: str
    point: Vec3
    normal: Vec3
    targets: list[str]
    align_up_min_dot: float


class EntryConstraintSpec(TypedDict):
    type: Literal["entry"]
    path: str


type RandomConstraint = Union[
    NonpenetrationConstraintSpec,
    BboxConstraintSpec,
    DistanceConstraintSpec,
    PlaneConstraintSpec,
    EntryConstraintSpec,
]


@final
class Constraint:
    """Helpers for building typed constraint specs."""

    @staticmethod
    def __nonpenetration(*between: str) -> NonpenetrationConstraintSpec:
        """Prevent penetration between named objects.
        TODO: @gangjeuk
        How to check?
        """
        return {"type": "nonpenetration", "between": list(between)}

    @staticmethod
    def bbox(
        min: Vec3,
        max: Vec3,
        *,
        mode: Literal["inside", "outside"],
        center: Vec3 | None = None,
    ) -> BboxConstraintSpec:
        """Keep the sampled value inside or outside an axis-aligned box."""
        spec: BboxConstraintSpec = {
            "type": "bbox",
            "min": min,
            "max": max,
            "mode": mode,
        }
        if center is not None:
            spec["center"] = center
        return spec

    @staticmethod
    def distance(*between: str, min: float, max: float) -> DistanceConstraintSpec:
        """Clamp the distance between named objects."""
        return {"type": "distance", "between": list(between), "min": min, "max": max}

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
    def entry(path: str) -> EntryConstraintSpec:
        """Reference a reusable external constraint entry."""
        return {"type": "entry", "path": path}


class UniformRandomSpec(TypedDict, Generic[ValueT], total=False):
    type: Required[Literal["uniform"]]
    min: Required[ValueT]
    max: Required[ValueT]
    constraints: list[RandomConstraint]


class NormalRandomSpec(TypedDict, Generic[ValueT], total=False):
    type: Required[Literal["normal"]]
    mean: Required[ValueT]
    std: Required[ValueT]
    clip_min: ValueT
    clip_max: ValueT
    constraints: list[RandomConstraint]


class ChoiceRandomSpec(TypedDict, Generic[ValueT], total=False):
    type: Required[Literal["choice"]]
    values: Required[list[ValueT]]
    constraints: list[RandomConstraint]

class EntryRandomSpec(TypedDict, Generic[ValueT], total=False):
    type: Required[Literal['entry']]
    path: Required[str]

RandomSpec: TypeAlias = Union[
    UniformRandomSpec[ValueT],
    NormalRandomSpec[ValueT],
    ChoiceRandomSpec[ValueT],
    EntryRandomSpec[ValueT]
]
Randomizable: TypeAlias = Union[ValueT, RandomSpec[ValueT]]

FloatRandomSpec: TypeAlias = RandomSpec[float]
Vec3RandomSpec: TypeAlias = RandomSpec[Vec3]
QuatRandomSpec: TypeAlias = RandomSpec[Quat]
ColorRandomSpec: TypeAlias = RandomSpec[ColorRgb]
FloatListRandomSpec: TypeAlias = RandomSpec[list[float]]
BoolRandomSpec: TypeAlias = RandomSpec[bool]

RandomizableFloat: TypeAlias = Randomizable[float]
RandomizableVec3: TypeAlias = Randomizable[Vec3]
RandomizableQuat: TypeAlias = Randomizable[Quat]
RandomizableColor: TypeAlias = Randomizable[ColorRgb]
RandomizableFloatList: TypeAlias = Randomizable[list[float]]
RandomizableBool: TypeAlias = Randomizable[bool]


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
        *,
        constraints: Sequence[RandomConstraint] = (),
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
        if constraints:
            spec["constraints"] = list(constraints)
        return spec

    @staticmethod
    def normal(
        mean: ValueT,
        std: ValueT,
        *,
        clip_min: ValueT | None = None,
        clip_max: ValueT | None = None,
        constraints: Sequence[RandomConstraint] = (),
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
        if constraints:
            spec["constraints"] = list(constraints)
        return spec

    @staticmethod
    def choice(
        *values: ValueT,
        constraints: Sequence[RandomConstraint] = (),
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
        if constraints:
            spec["constraints"] = list(constraints)
        return spec
