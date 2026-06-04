from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, overload

from simulac.base.error.error import SimulacBaseError
from simulac.base.types.geometry import ColorRgb, Vec3
from simulac.base.utils.rotation import euler_to_quat
from simulac.sdk import obtain_runtime
from simulac.sdk.environment_service.common.model.entity import TCameraType
from simulac.sdk.environment_service.common.model.ref import (
    ColliderRef,
)
from simulac.sdk.runner_service.common.model.runtime import (
    AmbientLightRuntime as SDKAmbientLightRuntime,
)
from simulac.sdk.runner_service.common.model.runtime import (
    AreaLightRuntime as SDKAreaLightRuntime,
)
from simulac.sdk.runner_service.common.model.runtime import (
    CameraRuntime as SDKCameraRuntime,
)
from simulac.sdk.runner_service.common.model.runtime import (
    LightRuntime as SDKLightRuntime,
)
from simulac.sdk.runner_service.common.model.runtime import (
    PointLightRuntime as SDKPointLightRuntime,
)
from simulac.sdk.runner_service.common.model.runtime import (
    RobotRuntime as SDKRobotRuntime,
)
from simulac.sdk.runner_service.common.model.runtime import RuntimeState
from simulac.sdk.runner_service.common.model.runtime import (
    SpotLightRuntime as SDKSpotLightRuntime,
)
from simulac.sdk.runner_service.common.model.runtime import (
    StuffRuntime as SDKStuffRuntime,
)

from .entity import ActionT
from .object import (
    _CREATE_SENTINAL,
    AmbientLightObject,
    AreaLightObject,
    CameraObject,
    Environment,
    LightObject,
    PointLightObject,
    RobotObject,
    SpotLightObject,
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
    from simulac.sdk.runner_service.common.runner import IRunner
    from simulac.sdk.runner_service.local.mujoco.context import MujocoNativeContext


class StuffRuntime:
    """Runtime handle for reading and mutating a non-robot object."""

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
        """Change the runtime mass of this object."""
        self._runtime.change_mass(mass)

    def change_pos(self, pos: Vec3) -> None:
        """Change the runtime position of this object."""
        self._runtime.change_pos(pos)

    def change_rot(self, rot: Vec3) -> None:
        """Change the runtime Euler rotation of this object."""
        self._runtime.change_quat(euler_to_quat(*rot))

    def change_friction(self, friction: float) -> None:
        """Change the runtime friction of this object."""
        self._runtime.change_friction(friction)

    @property
    def id(self) -> str:
        return self._runtime.id

    def get_pos(self) -> tuple[float, float, float]:
        return self._runtime.get_pos()

    def get_quat(self) -> tuple[float, float, float, float]:
        return self._runtime.get_quat()

    def collider(self, name: str) -> ColliderRef:
        """Return a collider ref scoped to this runtime object."""
        return ColliderRef(self._runtime.id, name)

    def joint(
        self, name: str
    ) -> SlideJointState | HingeJointState | BallJointState | FreeJointState:
        """Return runtime state for a named joint.

        Args:
            name: Asset-authored joint name.

        Returns:
            Runtime joint state matching the engine joint type.
        """
        return self._runtime.joint(name)


class RobotRuntime(Generic[ActionT]):
    """Runtime handle for robot control and articulated state."""

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
        """Write robot control values used by the next Runner.tick() call."""
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
        """Return runtime state for a named robot site."""
        return self._runtime.site(name)

    def link(self, name: str) -> LinkState:
        """Return runtime state for a named robot link."""
        return self._runtime.link(name)

    def joint(
        self, name: str
    ) -> HingeJointState | SlideJointState | BallJointState | FreeJointState:
        """Return runtime state for a named robot joint."""
        return self._runtime.joint(name)

    def sensor(self, name: str):
        """Return runtime state for a named robot sensor."""
        return self._runtime.sensor(name)

    def collider(self, name: str) -> ColliderRef:
        """Return a collider ref scoped to this robot."""
        return ColliderRef(self._runtime.id, name)

    def change_joint_pos(self, joint_pos: list[float]) -> None:
        """Change the robot joint position vector."""
        self._runtime.change_joint_pos(joint_pos)

    def change_joint_vel(self, joint_vel: list[float]) -> None:
        """Change the robot joint velocity vector."""
        self._runtime.change_joint_vel(joint_vel)

    # NOTE: below two are future use,
    # since our team concluded that we are focuing on `pos` control
    def _change_target_vel(self, vel: float) -> None: ...
    def _change_target_force(self, force: float) -> None: ...


class CameraRuntime(Generic[TCameraType]):
    """Runtime handle for camera pose, field of view, and rendering."""

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
        """Change the runtime camera position."""
        self._runtime.change_pos(pos)

    def change_rot(self, rot: Vec3) -> None:
        """Change the runtime camera Euler rotation."""
        self._runtime.change_quat(euler_to_quat(*rot))

    def get_fov(self) -> float:
        return self._runtime.get_fov()

    def change_fov(self, fov: float) -> None:
        """Change the runtime camera field of view."""
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
        """Render one frame using this camera's declared output type.

        Args:
            width: Output frame width.
            height: Output frame height.

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
    """Runtime handle for common light state."""

    def __init__(
        self,
        runtime_object: SDKLightRuntime,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create light runtime directly")
        self._runtime: SDKLightRuntime = runtime_object

    @property
    def id(self) -> str:
        return self._runtime.id

    def get_pos(self) -> Vec3:
        return self._runtime.get_pos()

    def get_quat(self) -> tuple[float, float, float, float]:
        return self._runtime.get_quat()

    def change_pos(self, pos: Vec3) -> None:
        """Change the runtime light position."""
        self._runtime.change_pos(pos)

    def change_rot(self, rot: Vec3) -> None:
        """Change the runtime light Euler rotation."""
        self._runtime.change_quat(euler_to_quat(*rot))

    def get_intensity(self) -> float:
        return self._runtime.get_intensity()

    def change_intensity(self, intensity: float) -> None:
        """Change the runtime light intensity."""
        self._runtime.change_intensity(intensity)

    def get_color(self) -> ColorRgb:
        return self._runtime.get_color()

    def change_color(self, color: ColorRgb) -> None:
        """Change the runtime light color."""
        self._runtime.change_color(color)


class AmbientLightRuntime(LightRuntime):
    """Runtime handle for ambient light state."""

    def __init__(
        self,
        runtime_object: SDKAmbientLightRuntime,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create light runtime directly")
        self._runtime: SDKAmbientLightRuntime = runtime_object


class PointLightRuntime(LightRuntime):
    """Runtime handle for point-light-specific state."""

    def __init__(
        self,
        runtime_object: SDKPointLightRuntime,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create light runtime directly")
        self._runtime: SDKPointLightRuntime = runtime_object

    def get_range(self) -> float:
        return self._runtime.get_range()

    def change_range(self, range: float) -> None:
        """Change the runtime point light range."""
        self._runtime.change_range(range)

    def get_decay(self) -> float:
        return self._runtime.get_decay()

    def change_decay(self, decay: float) -> None:
        """Change the runtime point light decay."""
        self._runtime.change_decay(decay)


class SpotLightRuntime(LightRuntime):
    """Runtime handle for spot-light-specific state."""

    def __init__(
        self,
        runtime_object: SDKSpotLightRuntime,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create light runtime directly")
        self._runtime: SDKSpotLightRuntime = runtime_object

    def get_range(self) -> float:
        return self._runtime.get_range()

    def change_range(self, range: float) -> None:
        """Change the runtime spot light range."""
        self._runtime.change_range(range)

    def get_decay(self) -> float:
        return self._runtime.get_decay()

    def change_decay(self, decay: float) -> None:
        """Change the runtime spot light decay."""
        self._runtime.change_decay(decay)

    def get_angle(self) -> float:
        return self._runtime.get_angle()

    def change_angle(self, angle: float) -> None:
        """Change the runtime spot light cone angle."""
        self._runtime.change_angle(angle)

    def get_direction(self) -> Vec3:
        return self._runtime.get_direction()

    def change_direction(self, direction: Vec3) -> None:
        """Change the runtime spot light direction."""
        self._runtime.change_direction(direction)

    def get_penumbra(self) -> float:
        return self._runtime.get_penumbra()

    def change_penumbra(self, penumbra: float) -> None:
        """Change the runtime spot light penumbra."""
        self._runtime.change_penumbra(penumbra)


class AreaLightRuntime(LightRuntime):
    """Runtime handle for area-light-specific state."""

    def __init__(
        self,
        runtime_object: SDKAreaLightRuntime,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create light runtime directly")
        self._runtime: SDKAreaLightRuntime = runtime_object

    def get_area_size(self) -> tuple[float, float]:
        return self._runtime.get_area_size()

    def change_area_size(self, width: float, height: float) -> None:
        """Change the runtime area light size."""
        self._runtime.change_area_size(width, height)

    def get_direction(self) -> Vec3:
        return self._runtime.get_direction()

    def change_direction(self, direction: Vec3) -> None:
        """Change the runtime area light direction."""
        self._runtime.change_direction(direction)


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
    """Runtime owner for a frozen Environment definition."""

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
        """Advance the runtime by one configured tick duration."""
        return self._runner.tick()

    def reset(self, seed: int | None = 0) -> RuntimeState:
        """Reset the runtime and return the latest state."""
        return self._runner.reset(seed)

    def sync(self) -> RuntimeState:
        """Synchronize and return the latest runtime state."""
        return self._runner.sync()

    @property
    def state(self) -> RuntimeState:
        """Current runtime state."""
        return self._runner.get_state()

    @overload
    def get_runtime_object(self, obj: StuffObject) -> StuffRuntime: ...
    @overload
    def get_runtime_object(
        self, obj: RobotObject[ActionT]
    ) -> RobotRuntime[ActionT]: ...
    @overload
    def get_runtime_object(self, obj: AmbientLightObject) -> AmbientLightRuntime: ...
    @overload
    def get_runtime_object(self, obj: PointLightObject) -> PointLightRuntime: ...
    @overload
    def get_runtime_object(self, obj: SpotLightObject) -> SpotLightRuntime: ...
    @overload
    def get_runtime_object(self, obj: AreaLightObject) -> AreaLightRuntime: ...
    @overload
    def get_runtime_object(self, obj: LightObject) -> LightRuntime: ...
    @overload
    def get_runtime_object(
        self, obj: CameraObject[TCameraType]
    ) -> CameraRuntime[TCameraType]: ...
    def get_runtime_object(
        self,
        obj: StuffObject | RobotObject[Any] | LightObject | CameraObject[TCameraType],
    ) -> (
        StuffRuntime
        | RobotRuntime[Any]
        | AmbientLightRuntime
        | PointLightRuntime
        | SpotLightRuntime
        | AreaLightRuntime
        | LightRuntime
        | CameraRuntime[TCameraType]
    ):
        """Return the typed runtime handle corresponding to a build-time object."""
        if obj._entity.id is None:
            raise SimulacBaseError("Entity should be added before runtime initialized")
        runtime_object = self._runner.get_runtime_object(obj._entity.id)

        if isinstance(obj, StuffObject):
            return StuffRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

        if isinstance(obj, RobotObject):
            return RobotRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

        if isinstance(obj, AmbientLightObject):
            return AmbientLightRuntime(
                runtime_object, _create_sentinal=_CREATE_SENTINAL
            )

        if isinstance(obj, PointLightObject):
            return PointLightRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

        if isinstance(obj, SpotLightObject):
            return SpotLightRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

        if isinstance(obj, AreaLightObject):
            return AreaLightRuntime(runtime_object, _create_sentinal=_CREATE_SENTINAL)

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
        """Return the native engine context for low-level engine-specific access.

        Args:
            engine (None | Literal[&quot;mujoco&quot;]): Optional engine literal used for static typing.

        Returns:
            Native context for the active runtime engine.
        """
        return self._runner.context(engine or "")

    # For context manage
    # e.g., `with Runner(env) as runner:`
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...

    def _debug_render(self):
        return self._runner._debug_render()
