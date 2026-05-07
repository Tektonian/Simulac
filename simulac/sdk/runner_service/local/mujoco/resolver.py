from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.environment_service.common.model.ref import (
    AnchorPosRef,
    BodyPosRef,
    BoundsCenterRef,
    BoundsMaxRef,
    BoundsMinRef,
    BoundsSizeRef,
    CameraPosRef,
    ColliderCenterRef,
    EntityPosRef,
    EntityQuatRef,
    EntityRotRef,
    JointAxisRef,
    JointRef,
    LightPosRef,
    RefBase,
    SupportPointRef,
    SurfaceCenterRef,
    SurfaceNormalRef,
    SurfaceSampleRef,
    WorldPointRef,
)
from simulac.sdk.runner_service.local.mujoco.binding import (
    MujocoRobotBinding,
    MujocoStuffBinding,
)

_AXIS: dict[str, tuple[float, float, float]] = {
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "front": (0.0, 1.0, 0.0),
    "back": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


class MujocoRefResolver:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

    def _id(self, obj_type: mujoco.mjtObj, entity_id: str, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, f"{entity_id}/{name}")
        if idx < 0:
            raise SimulacBaseError(f"No MuJoCo object named {entity_id}/{name}")
        return idx

    def resolve_point(self, ref: RefBase) -> list[float]:
        if isinstance(ref, WorldPointRef):
            if not isinstance(ref.pos, tuple):
                raise SimulacBaseError(
                    f"Resolved ref point must be tuple {ref}/{ref.pos}"
                )
            pos = ref.pos
            return [float(pos[0]), float(pos[1]), float(pos[2])]

        if isinstance(ref, (AnchorRef, AnchorPosRef)):
            sid = self._named_id(mujoco.mjtObj.mjOBJ_SITE, ref.entity_id, ref.name)
            return self.data.site_xpos[sid].copy().tolist()

        if isinstance(ref, (ColliderRef, ColliderCenterRef)):
            gid = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, ref.entity_id, ref.name)
            return self.data.geom_xpos[gid].copy().tolist()

        raise SimulacBaseError(f"Unsupported point ref: {ref}")

    def _joint_dims(self, joint_type: int) -> tuple[int, int]:
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7, 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4, 3
        return 1, 1
