from __future__ import annotations

import mujoco

from simulac.sdk.runner_service.common.model.context import INativeContext


class MujocoNativeContext(INativeContext):
    engine = "mujoco"
    model: mujoco.MjModel
    data: mujoco.MjData

    # TODO: @gangjeuk
    # implement interface
