from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING, Literal

import mujoco

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.environment_service.common.model.constraint import (
    BBoxConstraint,
    DistanceConstraint,
    EntityTarget,
    NonpenetrationConstraint,
    RefTarget,
    SceneConstraint,
)
from simulac.sdk.runner_service.local.mujoco.binding import (
    MujocoCameraBinding,
    MujocoRobotBinding,
    MujocoStuffBinding,
)
from simulac.sdk.runner_service.local.mujoco.resolver import MujocoRefResolver

if TYPE_CHECKING:
    from simulac.sdk.log_service.common.log_service import ILogService


@dataclass(frozen=True, slots=True)
class MujocoConstraintFailure:
    constraint_type: str
    message: str
    entities: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)

    def key(self) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
        return (
            self.constraint_type,
            self.message,
            self.entities,
            self.refs,
        )

    def format(self) -> str:
        identity_parts: list[str] = []

        if self.entities:
            identity_parts.append(f"entities={self.entities!r}")

        if self.refs:
            identity_parts.append(f"refs={self.refs!r}")

        detail_parts = [f"{key}={value!r}" for key, value in self.details.items()]

        suffix_parts = identity_parts + detail_parts

        if suffix_parts:
            return (
                f"[{self.constraint_type}] {self.message} ({', '.join(suffix_parts)})"
            )

        return f"[{self.constraint_type}] {self.message}"


@dataclass(frozen=True, slots=True)
class MujocoConstraintEvaluation:
    passed: bool
    failures: tuple[MujocoConstraintFailure, ...] = ()


class MujocoConstraintEvaluator:
    """TODO: @gangjeuk
    add debug message, when user .reset() failed so many times
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        resolver: MujocoRefResolver,
        bindings: dict[
            str,
            MujocoRobotBinding
            | MujocoStuffBinding
            | MujocoCameraBinding,
        ],
    ) -> None:
        self.model = model
        self.data = data
        self.resolver = resolver
        self.bindings = bindings

        # Constraint failed message cache for debugging
        self.failed_message = dict[str, int]()

    def evaluate(
        self, constraints: list[SceneConstraint]
    ) -> MujocoConstraintEvaluation:
        failures: list[MujocoConstraintFailure] = []

        for constraint in constraints:
            passed = self._passed(constraint)
            if passed is not True:
                failures.append(passed)

        return MujocoConstraintEvaluation(passed=not failures, failures=tuple(failures))

    def _passed(
        self, constraint: SceneConstraint
    ) -> Literal[True] | MujocoConstraintFailure:
        if isinstance(constraint, DistanceConstraint):
            return self._distance_passed(constraint)

        if isinstance(constraint, BBoxConstraint):
            return self._bbox_passed(constraint)

        if isinstance(constraint, DistanceConstraint):
            return self._distance_passed(constraint)

        if isinstance(constraint, NonpenetrationConstraint):  # pyright: ignore[reportUnnecessaryIsInstance]
            return self._nonpenetration_passed(constraint)

        raise SimulacBaseError(f"Unsupported scene constraint: {constraint!r}")

    def _distance_passed(
        self, constraint: DistanceConstraint
    ) -> Literal[True] | MujocoConstraintFailure:
        a = self._resolve_target_point(constraint.a)
        b = self._resolve_target_point(constraint.b)

        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        distance = sqrt(dx * dx + dy * dy + dz * dz)

        if constraint.min is not None and distance < constraint.min:
            return MujocoConstraintFailure(
                constraint_type="distance",
                message="Distance is below minimum",
                entities=tuple(self.__target_entities(constraint.a, constraint.b)),
                details={
                    "distance": distance,
                    "min": constraint.min,
                    "max": constraint.max,
                },
            )

        if constraint.max is not None and distance > constraint.max:
            return MujocoConstraintFailure(
                constraint_type="distance",
                message="Distance is above maximum",
                entities=self.__target_entities(constraint.a, constraint.b),
                details={
                    "distance": distance,
                    "min": constraint.min,
                    "max": constraint.max,
                },
            )

        return True

    def _bbox_passed(
        self, constraint: BBoxConstraint
    ) -> Literal[True] | MujocoConstraintFailure:
        point = self._resolve_target_point(constraint.target)

        inside = all(
            constraint.lower[i] <= point[i] <= constraint.upper[i] for i in range(3)
        )

        if constraint.mode == "inside":
            if inside is True:
                return inside
            else:
                return MujocoConstraintFailure(
                    constraint_type="bbox",
                    message="entity placed outside the box",
                    entities=self.__target_entities(constraint.target),
                    details={
                        "lower": constraint.lower,
                        "upper": constraint.upper,
                    },
                )

        if constraint.mode == "outside":
            if inside is False:
                return True
            else:
                return MujocoConstraintFailure(
                    constraint_type="bbox",
                    message="entity placed inside the box",
                    entities=self.__target_entities(constraint.target),
                    details={
                        "lower": constraint.lower,
                        "upper": constraint.upper,
                    },
                )
        raise SimulacBaseError(f"Unsupported bbox mode: {constraint.mode}")

    def _nonpenetration_passed(
        self, constraint: NonpenetrationConstraint
    ) -> Literal[True] | MujocoConstraintFailure:
        entity_ids = tuple(target.entity_id for target in constraint.entities)

        for i, a in enumerate(entity_ids):
            for b in entity_ids[i + 1 :]:
                ret = self._pair_nonpenetration_passed(a, b)
                if ret is not True:
                    return ret

        return True

    def _pair_nonpenetration_passed(
        self, a: str, b: str
    ) -> Literal[True] | MujocoConstraintFailure:
        binding_a = self.bindings.get(a)
        binding_b = self.bindings.get(b)

        if binding_a is None:
            raise SimulacBaseError(
                f"Unknown entity in nonpenetration constraint: {a!r}"
            )

        if binding_b is None:
            raise SimulacBaseError(
                f"Unknown entity in nonpenetration constraint: {b!r}"
            )

        if not isinstance(binding_a, (MujocoStuffBinding, MujocoRobotBinding)):
            raise SimulacBaseError(
                f"Unsupported nonpenetration target {a!r}: "
                f"{type(binding_a).__name__}. "
                "Only Stuff and Robot entities can be used in nonpenetration constraints."
            )

        if not isinstance(binding_b, (MujocoStuffBinding, MujocoRobotBinding)):
            raise SimulacBaseError(
                f"Unsupported nonpenetration target {b!r}: "
                f"{type(binding_b).__name__}. "
                "Only Stuff and Robot entities can be used in nonpenetration constraints."
            )

        if not binding_a.geom_ids:
            raise SimulacBaseError(
                f"Nonpenetration target {a!r} has no geoms registered in MuJoCo binding."
            )

        if not binding_b.geom_ids:
            raise SimulacBaseError(
                f"Nonpenetration target {b!r} has no geoms registered in MuJoCo binding."
            )

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # pass if is too small
            penetration_tolerance = 1e-5
            if contact.dist >= -1 * penetration_tolerance:
                continue
            a_geoms = set(binding_a.geom_ids)
            b_geoms = set(binding_b.geom_ids)
            g1, g2 = int(contact.geom1), int(contact.geom2)

            if (g1 in a_geoms and g2 in b_geoms) or (g2 in a_geoms and g1 in b_geoms):
                return MujocoConstraintFailure(
                    constraint_type="nonpenetration",
                    message="Entities are penetrating",
                    entities=(a, b),
                    details={
                        "geom1": g1,
                        "geom2": g2,
                        "dist": contact.dist,
                    },
                )

        return True

    def _resolve_target_point(self, target: EntityTarget | RefTarget):
        if isinstance(target, EntityTarget):
            binding = self.bindings.get(target.entity_id)

            if binding is None:
                raise SimulacBaseError(
                    f"No MuJoCo binding for entity {target.entity_id!r}"
                )

            return tuple(float(x) for x in self.data.xpos[binding.root_body_id])

        if isinstance(target, RefTarget):  # pyright: ignore[reportUnnecessaryIsInstance]
            return self.resolver.resolve_point(target.ref)

        raise SimulacBaseError(f"Unsupported constraint target: {target!r}")

    def __target_entities(
        self,
        *targets: EntityTarget | RefTarget,
    ) -> tuple[str, ...]:
        entity_ids: list[str] = []

        for target in targets:
            if isinstance(target, EntityTarget):
                entity_ids.append(target.entity_id)
                continue

            if isinstance(target, RefTarget):  # pyright: ignore[reportUnnecessaryIsInstance]
                entity_id = getattr(target.ref, "entity_id", None)
                if isinstance(entity_id, str):
                    entity_ids.append(entity_id)

        return tuple(entity_ids)
