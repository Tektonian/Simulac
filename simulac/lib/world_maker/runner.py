from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, overload

from simulac.base.error.error import SimulacBaseError
from simulac.base.types.geometry import Vec3
from simulac.sdk import obtain_runtime

from .entity import ActionT
from .object import _CREATE_SENTINAL, Environment

if TYPE_CHECKING:
    from .object import CameraObject, LightObject, RobotObject, StuffObject


class Runner:
    def __init__(
        self,
        env: Environment,
        seed: int | None = 0,
        tick: int | None = 5,  # 5ms
        record_location: str
        | None = None,  # save location of runtime recording data (a.k.a. Lerobot dataset format)
        /,
        *,
        runtime_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
    ):
        self.seed = seed
        self.tick_time = tick

        self._world_maker = obtain_runtime().world_maker

        self._runner = self._world_maker.create_runner(env._env.id)

        # Freeze and prevent changes in env
        env._freeze()

    def step(self, action: list[float]):
        self._runner.step(action)

    def tick(self):
        self._runner.tick()

    type State = Any

    def reset(self, seed: int | None = 0) -> State:
        self._runner.reset(seed)

    def get_state(self): ...

    @overload
    def get_runtime_object(self, obj: StuffObject) -> StuffRuntime: ...
    @overload
    def get_runtime_object(
        self, obj: RobotObject[ActionT]
    ) -> RobotRuntime[ActionT]: ...
    @overload
    def get_runtime_object(self, obj: LightObject) -> LightRuntime: ...
    @overload
    def get_runtime_object(self, obj: CameraObject) -> CameraRuntime: ...
    def get_runtime_object(
        self, obj: StuffObject | RobotObject[Any] | LightObject | CameraObject
    ) -> StuffRuntime | RobotRuntime[Any] | LightRuntime | CameraRuntime: ...

    def close(self) -> None: ...

    # For context manage
    # e.g., `with Runner(env) as runner:`
    def __enter__(self): ...
    def __exit__(self, exc_type, exc, tb): ...

    def _debug_render(self):
        return self._runner._debug_render()


class StuffRuntime:
    def __init__(
        self,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

    def change_mass(self, mass: float) -> None: ...
    def change_pos(self, pos: Vec3) -> None: ...
    def change_size(self, size: Vec3) -> None: ...
    def change_fixed(self, is_fixed: bool) -> None: ...
    def change_friction(self, friction: float) -> None: ...
    def change_density(self, density: float) -> None: ...

    def joint(self, name: str) -> None:
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


class RobotRuntime(Generic[ActionT]):
    def __init__(
        self,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

    def step(self, action: ActionT) -> None: ...
    def tick(self) -> None: ...

    def get_pos(self) -> Vec3: ...
    def get_vel(self) -> list[float]: ...


class CameraRuntime:
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
    def change_fov(self, fov: float) -> None:
        """for zoom mocking
        Needed?
        """


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

    def get_state(self) -> object: ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Runner: ...


class RuntimeState:
    def __init__(self):
        """Runtime state returned by `runner.step(action)`
        Remember that Simulac MUST NOT determine the end conditon of runner.
        Finishing runner is reponsible for user and this way is more fitable with philosophy of the Simulac.
        However, we should provide detailed information about running state to users to make them control end conditions.

        Example:
            # Expected usage pattern
            for _ in range(300):
                state = runner.step(action)

                # state SHOULD NOT contain information about `contact`
                # contact info only returned when `state.contacts()` called
                mug_on_table = state.contracts(
                    mug.collider("bottom"),
                    table.collider("top")
                )

                # details
                if mug_on_table:
                    print(mug_on_table.exists)
                    print(mug_on_table.count)
                    print(mug_on_table.max_force)
                    print(mug_on_table.points)
                    print(mug_on_table.normal)

                # Don't get confused, drawer is NOT runtime object, MUST BE StuffObject
                drawer_open = state.joint(drawer.joint("slide")).pos > 0.25

                runtime_drawer = runner.get_runtime_object(drawer)
                # difference between `runtime_drawer` and state.joint(drawer.joint("slide")) is
                # that `state.joint()` is readonly property, while `runtime_drawer` is mutable
                assert state.joint(drawer.joint("slide")).pos == runtime_drawer.pos

                if mug_on_table and drawer_open:
                    print("Happy! Happy! https://upload.wikimedia.org/wikipedia/commons/0/04/So_happy_smiling_cat.jpg")
                    break
        """
