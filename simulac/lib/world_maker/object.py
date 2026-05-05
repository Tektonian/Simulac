from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

from simulac.base.error.error import SimulacBaseError
from simulac.sdk import obtain_runtime
from simulac.sdk.environment_service.common.model.ref import (
    AnchorRef,
    ColliderRef,
    EntityRef,
    JointRef,
    PlaceOp,
    as_place_source,
    as_place_target,
)

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.model.entity import (
        EnvironmentCameraEntity,
        EnvironmentLightEntity,
        EnvironmentMachineEntity,
        EnvironmentStuffEntity,
    )
    from simulac.sdk.environment_service.common.model.ref import (
        ObjectRefType,
        PlaceTargetRefType,
        PointRefType,
    )
    from simulac.sdk.environment_service.common.randomize import (
        Randomizable,
        RandomizableBool,
        RandomizableColor,
        RandomizableFloat,
        RandomizableVec3,
        Vec3,
    )

from .entity import (
    ActionT,
    AmbientLight,
    AreaLight,
    Camera,
    PointLight,
    Robot,
    SpotLight,
    Stuff,
)

if TYPE_CHECKING:
    from .entity import LightType
# Sentinal pattern: https://python-patterns.guide/python/sentinel-object/
_CREATE_SENTINAL = object()


class Environment:
    def __init__(
        self,
        env_uri_or_prebuilt_id: str | None = None,
        default_engine: Literal["mujoco", "newton", "genesis"] = "mujoco",
    ) -> None:
        self._runtime = obtain_runtime()
        self._world_maker = self._runtime.world_maker

        self.default_engine = default_engine
        self._env = self._world_maker.create_environment(
            default_engine, env_uri_or_prebuilt_id
        )
        self.__frozen = False

    def _freeze(self):
        self.__frozen = True

    def _assert_mutable(self):
        if self.__frozen:
            raise SimulacBaseError(
                "\n".join(
                    [
                        "You are trying to change definition of Environment after Runner creation",
                        "Use runner.get_runtime_object(obj).change_*() to mutate runtime state",
                        "It is not illegal, but we intentionally forbidden such actions",
                    ]
                )
            )

    # NOTE: @gangjeuk
    # Should be `place()`?

    # Entity ID pattern
    #   entity_id: lower_snake_case
    #   qualified ref: <entity_id>.<kind>.<name>
    # e.g., entity_id
    #   table
    #   red_cube
    #   panda
    #   front_rgb
    # e.g., qualified_ref
    #   table.collider.top
    #   table.anchor.workspace_center
    #   panda.joint.wrist_1
    #   front_rgb.camera.output

    @overload
    def add_entity(
        self,
        entity: Stuff,
        pos: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> StuffObject: ...
    @overload
    def add_entity(
        self,
        entity: Camera,
        pos: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> CameraObject: ...
    @overload
    def add_entity(
        self,
        entity: LightType,
        pos: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> LightObject: ...
    @overload
    def add_entity(
        self,
        entity: Robot[ActionT],
        pos: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> RobotObject[ActionT]: ...
    def add_entity(
        self,
        entity: Stuff | Robot[ActionT] | Camera | LightType,
        pos: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
    ) -> StuffObject | RobotObject[ActionT] | CameraObject | LightObject:
        description = description or ""

        if isinstance(entity, Stuff):
            env_stuff_obj = self._world_maker.create_stuff_entity(
                entity.obj_uri_or_prebuilt_name, description=description
            )
            self._world_maker.add_entity(
                self._env.id, env_stuff_obj, entity_id, pos=pos, rot=rot
            )
            return StuffObject(env_stuff_obj, _create_sentinal=_CREATE_SENTINAL)
        elif isinstance(entity, Robot):
            env_robot_obj = self._world_maker.create_machine_entity(
                entity.obj_uri_or_prebuilt_name, description=description
            )
            self._world_maker.add_entity(
                self._env.id, env_robot_obj, entity_id, pos=pos, rot=rot
            )
            return cast(
                "RobotObject[ActionT]",
                RobotObject(env_robot_obj, _create_sentinal=_CREATE_SENTINAL),
            )
        elif isinstance(entity, Camera):
            env_camera_obj = self._world_maker.create_camera_entity(
                entity._to_spec(), description=description
            )
            self._world_maker.add_entity(
                self._env.id, env_camera_obj, entity_id, pos=pos, rot=rot
            )
            return CameraObject(env_camera_obj, _create_sentinal=_CREATE_SENTINAL)
        else:
            env_light_obj = self._world_maker.create_light_entity(
                entity._to_spec(), description=description
            )
            self._world_maker.add_entity(
                self._env.id, env_light_obj, entity_id, pos=pos, rot=rot
            )
            return LightObject(env_light_obj, _create_sentinal=_CREATE_SENTINAL)

        # Should not reach
        raise NotImplementedError("Wrong entity")

    def place_object(
        self,
        obj: StuffObject | RobotObject[Any],
        *,
        on: PlaceTargetRefType,
        using: PlaceTargetRefType | None = None,
        margin: RandomizableFloat = 0.0,
    ):
        """
        # When user want to place something on another thing
        env.place(
            mug,
            on=table.collider("top")
            using=mug.collider("bottom"),
            margin=0.04
        )
        """
        self._assert_mutable()
        if obj._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")

        # TODO: @gangjeuk
        # [ ] - verify before place
        # [o] - change `_entity.build_ops` to `_env.relations`

        self._env.relations.append(
            PlaceOp(
                EntityRef(obj._entity.id),
                as_place_target(on, margin=margin),
                as_place_source(using),
                margin,
            )
        )

    @overload
    def remove_object(
        self,
        object_or_object_id: StuffObject
        | RobotObject[Any]
        | CameraObject
        | LightObject,
    ) -> None: ...
    @overload
    def remove_object(self, object_or_object_id: str) -> None: ...
    def remove_object(
        self,
        object_or_object_id: StuffObject
        | RobotObject[Any]
        | CameraObject
        | LightObject
        | str,
    ) -> None:
        pass

    def get_object(
        self, object_id: str
    ) -> StuffObject | RobotObject[Any] | CameraObject | LightObject: ...

    def dump_env(self) -> dict:
        """Return definition of environment.
        Return type `dict` is json format

        Raises:
            SimulacBaseError: _description_

        Returns:
            dict: json format environment definition
        """
        ...


class StuffObject:
    def __init__(
        self,
        entity: EnvironmentStuffEntity,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

        self._entity = entity

    def collider(self, name: str) -> ColliderRef:
        """When user want to customize collision mesh.
        TODO: @gangjeuk
        write codes.

        Example:
            # named collider reference
            # Asset author is responsible for the adequate name of collision mesh
            # Simulac will not support asset editing, edit mjcf, urdf, usd by yourself!

            # named collider reference and set randomization
            table.collider("top").set_friction(Randomize.uniform(0.3, 1.5))

            # geometry derived placement
            cube.set_pos(table.collider("top").surface("up").center)

            # semantic author-defined reference
            # `.anchor` is specific location of an asset defined by an asset author
            # For example
            #   <!--MJCF-->
            #   <body name="table">
            #       <geom name="top" type="box" pos="0 0 0.75" size="0.6 0.4 0.03" />
            #       <site name="workspace_center" pos="0 0 0.79" />
            #       <site name="robot_mount" pos="-0.45 0 0.78" />
            #   </body>
            robot.set_pos(table.anchor("workspace_center").pos)

            # In case of normal 3d asset like .obj and .glb
            # Each node name should be the name of collision mesh
            GLB nodes:
            - top
            - robot_mount

            OBJ groups:
            g top
            g robot_mount

            # ColliderRef type example

            table.anchor("place_area")      # semantic author-defined reference

            top = table.collider("top")     # named collision shape

            top.center      # collider volume center
            top.pose        # collider frame pos
            top.bounds      # world-space bounds (AABB/OBB-ish)
            top.bounds.center
            top.bounds.max
            top.bounds.min
            top.bounds.size

            top.surface("up").center    # center of contact surface
            top.surface("up").normal    # normal vector of contact surface
            top.support((0, 0, 1), frame="world")  # outer contact feature toward world +Z
            top.support((0, 0, 1), frame="local")  # outer contact feature toward local +Z

            top.surface("up").sample(margin=0.04)      # Generate a target point on the table, offset by 4cm from all edges.


        """
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        ref = ColliderRef(self._entity.id, name)
        return ref

    def joint(self, name: str) -> JointRef:
        """When user want to control joint
        TODO: @gangjeuk
        implement code (TOO many TODOs)

        # Same as collision mesh control, we do not provide asset editing

        # named joint reference
        slide = drawer.joint("slide")

        # build-time initial state
        slide.set_pos(Randomize.uniform(0.0, 0.15))

        # optional joint-level randomization
        slide.set_friction(Randomize.uniform(0.1, 0.5))
        slide.set_damping(Randomize.uniform(0.02, 0.2))

        # get articulated state
        pull_pose = drawer.anchor("handle_grasp").pose

        # exposed readonly properties
        joint.pose
        joint.axis
        joint.limit
        joint.type

        """
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        return JointRef(self._entity.id, name, _build_ops=self._entity.build_ops)

    def anchor(self, name: str) -> AnchorRef:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        return AnchorRef(self._entity.id, name)

    def set_mass(self, mass: RandomizableFloat) -> None:
        # do assertion first
        # self._env._assert_mutate()
        ...

    def set_pos(self, pos: RandomizableVec3) -> None: ...
    def set_rot(self, rot: RandomizableVec3) -> None: ...
    def set_size(self, size: RandomizableVec3) -> None: ...
    def set_fixed(self, is_fixed: RandomizableBool) -> None: ...
    def set_friction(self, friction: RandomizableFloat) -> None: ...
    def set_density(self, density: RandomizableFloat) -> None: ...


class RobotObject(Generic[ActionT]):
    def __init__(
        self,
        entity: EnvironmentMachineEntity,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

        self._entity = entity

    def set_pos(self, pos: RandomizableVec3) -> None: ...
    def set_rot(self, rot: RandomizableVec3) -> None: ...
    def set_joint_pos(self, pos: Randomizable[ActionT]) -> None: ...

    def get_joint_min(self) -> ActionT: ...
    def get_joint_max(self) -> ActionT: ...

    """
    See comments on `StuffObject`
    """

    def joint(self, name: str) -> JointRef:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        return JointRef(self._entity.id, name, _build_ops=self._entity.build_ops)

    def collider(self, name: str) -> ColliderRef:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        return ColliderRef(self._entity.id, name, _build_ops=self._entity.build_ops)

    def anchor(self, name: str) -> AnchorRef:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        return AnchorRef(self._entity.id, name)


class CameraObject:
    def __init__(
        self,
        entity: EnvironmentCameraEntity,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")
        self._entity = entity

    def set_pos(self, pos: RandomizableVec3) -> None: ...
    def set_rot(self, rot: RandomizableVec3) -> None: ...
    def set_fov(self, fov: RandomizableFloat) -> None: ...
    def set_aspect(self, aspect: RandomizableFloat) -> None: ...
    def set_near(self, near: RandomizableFloat) -> None: ...
    def set_far(self, far: RandomizableFloat) -> None: ...

    def set_type(
        self,
        type: Literal[
            "rgb", "tactile", "depth", "pointcloud", "normal", "segmentation"
        ],
    ): ...

    # Needed? @gangjeuk
    def _set_resolution(self): ...
    def _set_noise(self): ...
    def _set_exposure(self, exposure: RandomizableFloat): ...

    def look_at(
        self,
        target: Vec3 | AnchorRef | ColliderRef,
        *,
        up: Vec3,
        offset: RandomizableVec3,
    ) -> None: ...

    def attach_to(
        self,
        parent: AnchorRef,
        *,
        offset: RandomizableVec3,
        rot: RandomizableVec3,
    ) -> None: ...
    def follow(
        self,
        target: AnchorRef | ColliderRef | RobotObject[Any] | StuffObject,
        *,
        offset: RandomizableVec3,
        frame: Literal["world", "local"],
    ): ...


class LightObject:
    def __init__(
        self,
        entity: EnvironmentLightEntity,
        /,
        *,
        _create_sentinal: object,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")
        self._entity = entity

    def set_pos(self, pos: RandomizableVec3) -> None: ...
    def set_rot(self, rot: RandomizableVec3) -> None: ...
    def set_intensity(self, intensity: RandomizableFloat) -> None: ...
    def set_type(
        self, type: Literal["ambient", "pointlight", "reactarea", "spot"]
    ) -> None: ...
    def set_color(self, color: RandomizableColor) -> None: ...

    def set_angle(self, angle: RandomizableFloat) -> None:
        """spot only"""

    def set_area_size(
        self, width: RandomizableFloat, height: RandomizableFloat
    ) -> None:
        """area only"""

    # Below two are for headlight
    def look_at(
        self,
        target: Vec3 | AnchorRef | ColliderRef,
        *,
        up: Vec3,
        offset: RandomizableVec3,
    ) -> None: ...

    def attach_to(
        self,
        parent: AnchorRef,
        *,
        offset: RandomizableVec3,
        rot: RandomizableVec3,
    ) -> None: ...
