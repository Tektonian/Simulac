from __future__ import annotations

from math import sqrt

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
    MujocoLightBinding,
    MujocoRobotBinding,
    MujocoStuffBinding,
)
from simulac.sdk.runner_service.local.mujoco.resolver import MujocoRefResolver


class MujocoConstraintEvaluator:
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
            | MujocoCameraBinding
            | MujocoLightBinding,
        ],
    ) -> None:
        self.model = model
        self.data = data
        self.resolver = resolver
        self.bindings = bindings

    def passed(self, constraints: list[SceneConstraint]) -> bool:
        return all(self._passed(constraint) for constraint in constraints)

    def _passed(self, constraint: SceneConstraint) -> bool:
        if isinstance(constraint, DistanceConstraint):
            return self._distance_passed(constraint)

        if isinstance(constraint, BBoxConstraint):
            return self._bbox_passed(constraint)

        if isinstance(constraint, DistanceConstraint):
            return self._distance_passed(constraint)

        if isinstance(constraint, NonpenetrationConstraint):  # pyright: ignore[reportUnnecessaryIsInstance]
            return self._nonpenetration_passed(constraint)

        raise SimulacBaseError(f"Unsupported scene constraint: {constraint!r}")

    def _distance_passed(self, constraint: DistanceConstraint) -> bool:
        a = self._resolve_target_point(constraint.a)
        b = self._resolve_target_point(constraint.b)

        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        distance = sqrt(dx * dx + dy * dy + dz * dz)

        if constraint.min is not None and distance < constraint.min:
            return False

        if constraint.max is not None and distance > constraint.max:
            return False

        return True

    def _bbox_passed(self, constraint: BBoxConstraint) -> bool:
        point = self._resolve_target_point(constraint.target)

        inside = all(
            constraint.lower[i] <= point[i] <= constraint.upper[i] for i in range(3)
        )

        if constraint.mode == "inside":
            return inside

        if constraint.mode == "outside":
            return not inside

        raise SimulacBaseError(f"Unsupported bbox mode: {constraint.mode}")

    def _nonpenetration_passed(self, constraint: NonpenetrationConstraint) -> bool:
        entity_ids = tuple(target.entity_id for target in constraint.entities)

        for i, a in enumerate(entity_ids):
            for b in entity_ids[i + 1 :]:
                if not self._pair_nonpenetration_passed(a, b):
                    return False

        return True

    def _pair_nonpenetration_passed(self, a: str, b: str) -> bool:
        binding_a = self.bindings.get(a)
        binding_b = self.bindings.get(b)

        if binding_a is None:
            raise SimulacBaseError(f"No MuJoCo binding for entity {a!r}")

        if binding_b is None:
            raise SimulacBaseError(f"No MuJoCo binding for entity {b!r}")

        if not isinstance(
            binding_a, (MujocoRobotBinding, MujocoStuffBinding)
        ) or not isinstance(binding_b, (MujocoStuffBinding, MujocoRobotBinding)):
            raise SimulacBaseError(f"No MuJoCo binding for entity {b!r}")

        body_a = binding_a.root_body_id
        body_b = binding_b.root_body_id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # geom1_body = int(self.model.geom_bodyid[contact.geom1])
            # geom2_body = int(self.model.geom_bodyid[contact.geom2])

            # pass if is too small
            if contact.dist >= -1e-5:
                continue
            a_geoms = binding_a.geom_ids
            b_geoms = binding_b.geom_ids
            g1, g2 = int(contact.geom1), int(contact.geom2)

            if (g1 in a_geoms and g2 in b_geoms) or (g2 in a_geoms and g1 in b_geoms):
                return False

            # if (
            #     self._is_descendant_body(geom1_body, body_a)
            #     and self._is_descendant_body(geom2_body, body_b)
            # ) or (
            #     self._is_descendant_body(geom1_body, body_b)
            #     and self._is_descendant_body(geom2_body, body_a)
            # ):
            #     if float(contact.dist) < 0.0:
            #         return False

        return True

    # def _is_descendant_body(self, child: int, parent: int) -> bool:
    #     body = child
    #     visited: set[int] = set()

    #     while body not in visited:
    #         if body == parent:
    #             return True

    #         visited.add(body)

    #         if body == 0:
    #             break
    #         body = int(self.model.body_parentid[body])

    #     return False

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
