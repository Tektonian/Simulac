from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, overload

from simulac.base.error.error import SimulacBaseError
from simulac.base.types.geometry import Vec3
from simulac.base.utils.rotation import euler_to_quat
from simulac.sdk import obtain_runtime
from simulac.sdk.environment_service.common.model.entity import TCameraType
from simulac.sdk.environment_service.common.model.ref import (
    ColliderRef,
)
from simulac.sdk.runner_service.common.model.runtime import RuntimeState

from .entity import ActionT
from .object import (
    _CREATE_SENTINAL,
    CameraObject,
    Environment,
    LightObject,
    RobotObject,
    StuffObject,
)

if TYPE_CHECKING:
    from simulac.sdk.runner_service.common.model.context import INativeContext
    from simulac.sdk.runner_service.common.model.runtime import (
        BallJointState,
        FreeJointState,
        HingeJointState,
        LinkState,
        SensorState,
        SiteState,
        SlideJointState,
    )
    from simulac.sdk.runner_service.common.model.runtime import (
        CameraRuntime as SDKCameraRuntime,
    )
    from simulac.sdk.runner_service.common.model.runtime import (
        RobotRuntime as SDKRobotRuntime,
    )
    from simulac.sdk.runner_service.common.model.runtime import (
        StuffRuntime as SDKStuffRuntime,
    )
    from simulac.sdk.runner_service.common.runner import IRunner
    from simulac.sdk.runner_service.local.mujoco.context import MujocoNativeContext


class StuffRuntime:
    def __init__(
        self,
        runtime_object: SDKStuffRuntime,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")
        self._runtime = runtime_object

    def change_mass(self, mass: float) -> None:
        self._runtime.change_mass(mass)

    def change_pos(self, pos: Vec3) -> None:
        self._runtime.change_pos(pos)

    def change_rot(self, rot: Vec3) -> None:
        self._runtime.change_quat(euler_to_quat(*rot))

    def change_friction(self, friction: float) -> None:
        self._runtime.change_friction(friction)

    @property
    def id(self) -> str:
        return self._runtime.id

    def get_pos(self) -> tuple[float, float, float]:
        return self._runtime.get_pos()

    def get_quat(self) -> tuple[float, float, float, float]:
        return self._runtime.get_quat()

    def collider(self, name: str) -> ColliderRef:
        return ColliderRef(self._runtime.id, name)

    def joint(
        self, name: str
    ) -> SlideJointState | HingeJointState | BallJointState | FreeJointState:
        """Runtime joint control
        See object.py:StuffObject
        TODO: @gangjeuk
        implement code

        # common api
        joint = runtime_obj.joint("joint_name")

        joint.get_pos()
        joint.get_vel()
        joint.change_pos(Vec3)
        joint.change_target(value)
        """
        return self._runtime.joint(name)


class RobotRuntime(Generic[ActionT]):
    def __init__(
        self,
        runtime_object: SDKRobotRuntime,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create robot runtime directly")
        self._runtime = runtime_object

    def set_control(self, action: ActionT) -> None:
        self._runtime.set_control(list(action))

    @property
    def id(self) -> str:
        return self._runtime.id

    def get_pos(self) -> Vec3:
        return self._runtime.get_pos()

    def get_quat(self) -> tuple[float, float, float, float]:
        return self._runtime.get_quat()

    def get_joint_pos(self) -> list[float]:
        return self._runtime.get_joint_pos()

    def get_joint_vel(self) -> list[float]:
        return self._runtime.get_joint_vel()

    def site(self, name: str) -> SiteState:
        return self._runtime.site(name)

    def link(self, name: str) -> LinkState:
        return self._runtime.link(name)

    def joint(
        self, name: str
    ) -> HingeJointState | SlideJointState | BallJointState | FreeJointState:
        return self._runtime.joint(name)

    def sensor(self, name: str):
        return self._runtime.sensor(name)

    def collider(self, name: str) -> ColliderRef:
        return ColliderRef(self._runtime.id, name)

    def change_joint_pos(self, joint_pos: list[float]) -> None:
        self._runtime.change_joint_pos(joint_pos)

    def change_joint_vel(self, joint_vel: list[float]) -> None:
        self._runtime.change_joint_vel(joint_vel)

    # NOTE: below two are future use,
    # since our team concluded that we are focuing on `pos` control
    def _change_target_vel(self, vel: float) -> None: ...
    def _change_target_force(self, force: float) -> None: ...


class CameraRuntime(Generic[TCameraType]):
    def __init__(
        self,
        runtime_object: SDKCameraRuntime,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create runtime camera directly")
        self._runtime = runtime_object

    def get_pos(self) -> Vec3:
        return self._runtime.get_pos()

    def get_quat(self) -> tuple[float, float, float, float]:
        return self._runtime.get_quat()

    def change_pos(self, pos: Vec3) -> None:
        self._runtime.change_pos(pos)

    def change_rot(self, rot: Vec3) -> None:
        self._runtime.change_quat(euler_to_quat(*rot))

    def get_fov(self) -> float:
        return self._runtime.get_fov()

    def change_fov(self, fov: float) -> None:
        """for zoom mocking
        Needed?
        """
        self._runtime.change_fov(fov)

    @overload
    def render(
        self: CameraRuntime[Literal["rgb"]],
        *,
        width: int = 640,
        height: int = 480,
    ) -> list[list[tuple[int, int, int]]]: ...

    @overload
    def render(
        self: CameraRuntime[Literal["depth"]],
        *,
        width: int = 640,
        height: int = 480,
    ) -> list[list[float]]: ...

    @overload
    def render(
        self: CameraRuntime[Literal["segmentation"]],
        *,
        width: int = 640,
        height: int = 480,
    ) -> list[list[tuple[int, int]]]: ...

    @overload
    def render(
        self: CameraRuntime[Literal["pointcloud"]],
        *,
        width: int = 640,
        height: int = 480,
    ) -> tuple[
        list[list[tuple[float, float, float]]],
        list[list[bool]],
    ]: ...

    def render(self: CameraRuntime[Any], *, width: int = 640, height: int = 480):
        """_summary_

        Args:
            width (int): _description_
            height (int): _description_

        Returns:
            RGBFrame: rgb[height][width] -> (r, g, b)
            DepthFrame: depth[height][width] -> distance
            SegmentationFrame: segmentation[height][width] -> (object_id, object_type)
            PointCloudFrame:
                pointcloud = (points, mask)
                points[height][width] -> (x, y, z)
                mask[height][width] -> valid/invalid
        """

        return self._runtime.render(width=width, height=height)


class LightRuntime:
    def __init__(
        self,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

    def change_pos(self, pos: Vec3) -> None: ...
    def change_rot(self, rot: Vec3) -> None: ...
    def change_intensity(self, intensity: float) -> None: ...
    def change_color(self, color: tuple[int, int, int]) -> None: ...

    def change_angle(self, angle: float) -> None: ...
    def change_area_size(self, width: float, height: float) -> None: ...
    def look_at(
        self,
        target: Any,
        *,
        up: Vec3 = (0, 0, 1),
    ) -> None: ...


class ParallelRunner:
    def __init__(
        self,
        envs: list[Environment],
        seeds: list[int] | None = None,
        tick: list[int] | None = None,
        record_locations: list[str] | None = None,
        strict: bool = True,
    ) -> None: ...

    def step(self, actions: list[list[float]]) -> None: ...
    def tick(self) -> None: ...

    type State = Any

    def reset(self, seeds: list[int]) -> list[State]: ...

    def close(self) -> None: ...

    # For context manage
    # e.g., `with Runner(env) as runner:`
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...

    def at(self, idx: int) -> Runner: ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Runner: ...


class Runner:
    def __init__(
        self,
        env: Environment,
        seed: int | None = 0,
        tick_dt_ms: int | None = 5,  # 5ms
        # record_location: str
        # | None = None,  # save location of runtime recording data (a.k.a. Lerobot dataset format)
        /,
        *,
        runtime_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
    ):
        self.seed = seed
        self.tick_dt_ms = tick_dt_ms

        self._world_maker = obtain_runtime().world_maker

        self._runner = self._world_maker.create_runner(
            env._env.id, tick_dt_ms=tick_dt_ms, runtime_engine=runtime_engine
        )

        # Freeze and prevent changes in env
        env._freeze()

    # Replaced with RobotRuntime.set_control
    # def step(self, action: list[float]) -> RuntimeState:
    #     return self._runner.step(action)

    def tick(self) -> RuntimeState:
        return self._runner.tick()

    def reset(self, seed: int | None = 0) -> RuntimeState:
        return self._runner.reset(seed)

    def sync(self) -> RuntimeState:
        return self._runner.sync()

    @property
    def state(self) -> RuntimeState:
        return self._runner.get_state()

    @overload
    def get_runtime_object(self, obj: StuffObject) -> StuffRuntime: ...
    @overload
    def get_runtime_object(
        self, obj: RobotObject[ActionT]
    ) -> RobotRuntime[ActionT]: ...
    @overload
    def get_runtime_object(self, obj: LightObject) -> LightRuntime: ...
    @overload
    def get_runtime_object(
        self, obj: CameraObject[TCameraType]
    ) -> CameraRuntime[TCameraType]: ...
    def get_runtime_object(
        self,
        obj: StuffObject | RobotObject[Any] | LightObject | CameraObject[TCameraType],
    ) -> StuffRuntime | RobotRuntime[Any] | LightRuntime | CameraRuntime[TCameraType]:
        if obj._entity.id is None:
            raise SimulacBaseError("Entity should be added before runtime initialized")
        runtime_object = self._runner.get_runtime_object(obj._entity.id)

        if isinstance(obj, StuffObject):
            return StuffRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

        if isinstance(obj, RobotObject):
            return RobotRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

        if isinstance(obj, LightObject):
            return LightRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

        if isinstance(obj, CameraObject):  # pyright: ignore[reportUnnecessaryIsInstance]
            return CameraRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

        raise SimulacBaseError(f"Unsupported runtime object: {type(obj).__name__}")

    def close(self) -> None: ...

    @overload
    def context(self, engine: None) -> INativeContext: ...
    @overload
    def context(self, engine: Literal["mujoco"]) -> MujocoNativeContext: ...
    def context(
        self, engine: None | Literal["mujoco"]
    ) -> INativeContext | MujocoNativeContext:
        """Return physics engine native context

        Args:
            engine (None | Literal[&quot;mujoco&quot;]): Engine name. Ignore it, it's just for typing.

        Returns:
            INativeContext | MujocoNativeContext: _description_
        """
        return self._runner.context(engine or "")

    # For context manage
    # e.g., `with Runner(env) as runner:`
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...

    def _debug_render(self):
        return self._runner._debug_render()
