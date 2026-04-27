from __future__ import annotations

from typing import (
    Generic,
    Literal,
    TypedDict,
    TypeVar,
    Union,
)

from typing_extensions import Required, TypeAlias

from simulac.base.types.geometry import ColorRgb, Quat, Vec3

ValueT = TypeVar("ValueT")

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
    type: Required[Literal["entry"]]
    path: Required[str]


RandomSpec: TypeAlias = Union[
    UniformRandomSpec[ValueT],
    NormalRandomSpec[ValueT],
    ChoiceRandomSpec[ValueT],
    EntryRandomSpec[ValueT],
]
Randomizable: TypeAlias = Union[ValueT, RandomSpec[ValueT]]

FloatRandomSpec: TypeAlias = RandomSpec[float]
Vec3RandomSpec: TypeAlias = RandomSpec[Vec3]
QuatRandomSpec: TypeAlias = RandomSpec[Quat]
ColorRandomSpec: TypeAlias = RandomSpec[ColorRgb]
FloatListRandomSpec: TypeAlias = RandomSpec[list[float]]
BoolRandomSpec: TypeAlias = RandomSpec[bool]

RandomizableFloat: TypeAlias = Randomizable[float]
type RandomizableTuple[*Ts] = Randomizable[tuple[*Ts]]
RandomizableVec3: TypeAlias = Randomizable[Vec3]
RandomizableQuat: TypeAlias = Randomizable[Quat]
RandomizableColor: TypeAlias = Randomizable[ColorRgb]
RandomizableFloatList: TypeAlias = Randomizable[list[float]]
RandomizableBool: TypeAlias = Randomizable[bool]
