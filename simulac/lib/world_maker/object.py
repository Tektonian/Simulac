from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

from simulac.base.error.error import SimulacBaseError
from simulac.sdk import obtain_runtime
from simulac.sdk.environment_service.common.model.ref import (
    AnchorRef,
    AttachOp,
    CameraRef,
    ColliderRef,
    EntityRef,
    FollowOp,
    JointRef,
    LookAtOp,
    PlaceOp,
    SetCameraFovOp,
    SetCameraPosOp,
    SetCameraRotOp,
    WorldPointRef,
    as_place_source,
    as_place_target,
)

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.environment import IEnvironment
    from simulac.sdk.environment_service.common.model.constraint import SceneConstraint
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
        RandomConstraint,
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
        pos: RandomizableVec3 | PointRefType = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
        *,
        fixed: bool | None = None,
    ) -> StuffObject: ...
    @overload
    def add_entity(
        self,
        entity: Camera,
        pos: RandomizableVec3 | PointRefType = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
        *,
        fixed: bool | None = None,
    ) -> CameraObject: ...
    @overload
    def add_entity(
        self,
        entity: LightType,
        pos: RandomizableVec3 | PointRefType = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
        *,
        fixed: bool | None = None,
    ) -> LightObject: ...
    @overload
    def add_entity(
        self,
        entity: Robot[ActionT],
        pos: RandomizableVec3 | PointRefType = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
        *,
        fixed: bool | None = None,
    ) -> RobotObject[ActionT]: ...
    def add_entity(
        self,
        entity: Stuff | Robot[ActionT] | Camera | LightType,
        pos: RandomizableVec3 | PointRefType = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
        entity_id: str | None = None,
        description: str | None = None,
        *,
        fixed: bool | None = None,
    ) -> StuffObject | RobotObject[ActionT] | CameraObject | LightObject:
        """_summary_

        Args:
            entity (Stuff | Robot[ActionT] | Camera | LightType): _description_
            pos (RandomizableVec3, optional): _description_. Defaults to (0, 0, 0).
            rot (RandomizableVec3, optional): _description_. Defaults to (0, 0, 0).
            entity_id (str | None, optional): _description_. Defaults to None.
            description (str | None, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            StuffObject | RobotObject[ActionT] | CameraObject | LightObject: _description_
        """

        description = description or ""

        if isinstance(entity, Stuff):
            env_stuff_obj = self._world_maker.create_stuff_entity(
                entity.obj_uri_or_prebuilt_name, description=description
            )
            self._world_maker.add_entity(
                self._env.id,
                env_stuff_obj,
                entity_id,
                pos=pos,
                rot=rot,
                fixed=fixed,
            )
            return StuffObject(
                env_stuff_obj, _create_sentinal=_CREATE_SENTINAL, env=self
            )
        elif isinstance(entity, Robot):
            if fixed is not None:
                raise SimulacBaseError(
                    "\n".join(
                        [
                            "fixed is only supported for Stuff entities.",
                            "Edit your robot asset file if you want to change it",
                        ]
                    )
                )
            env_robot_obj = self._world_maker.create_machine_entity(
                entity.obj_uri_or_prebuilt_name, description=description
            )
            self._world_maker.add_entity(
                self._env.id, env_robot_obj, entity_id, pos=pos, rot=rot
            )
            return cast(
                "RobotObject[ActionT]",
                RobotObject(env_robot_obj, _create_sentinal=_CREATE_SENTINAL, env=self),
            )
        elif isinstance(entity, Camera):
            if fixed is not None:
                raise SimulacBaseError(
                    "\n".join(
                        [
                            "fixed is only supported for Stuff entities.",
                            "If you want to movable camera, use `.attach()`",
                        ]
                    )
                )
            env_camera_obj = self._world_maker.create_camera_entity(
                entity._to_spec(), description=description
            )
            self._world_maker.add_entity(
                self._env.id, env_camera_obj, entity_id, pos=pos, rot=rot
            )
            return CameraObject(
                env_camera_obj, _create_sentinal=_CREATE_SENTINAL, env=self._env
            )
        elif isinstance(entity, (AreaLight, SpotLight, PointLight, AmbientLight)):  # pyright: ignore[reportUnnecessaryIsInstance]
            if fixed is not None:
                raise SimulacBaseError(
                    "\n".join(
                        [
                            "fixed is only supported for Stuff entities.",
                            "If you want to movable light, use `.attach()`",
                        ]
                    )
                )
            env_light_obj = self._world_maker.create_light_entity(
                entity._to_spec(), description=description
            )
            self._world_maker.add_entity(
                self._env.id, env_light_obj, entity_id, pos=pos, rot=rot
            )
            return LightObject(env_light_obj, _create_sentinal=_CREATE_SENTINAL)

        # Should not reach
        raise NotImplementedError("Wrong entity")

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
        # TODO: @gangjeuk
        # [ ] - Remove object
        # [ ] - Remove relations and constraints connected to object
        pass

    def get_object(
        self, object_id: str
    ) -> StuffObject | RobotObject[Any] | CameraObject | LightObject | None:
        env = self._env
        for obj in env.stuffs:
            if obj.id == object_id:
                return StuffObject(obj, _create_sentinal=_CREATE_SENTINAL, env=self)
        for obj in env.machines:
            if obj.id == object_id:
                return RobotObject(obj, _create_sentinal=_CREATE_SENTINAL, env=self)
        for obj in env.lights:
            if obj.id == object_id:
                return LightObject(obj, _create_sentinal=_CREATE_SENTINAL)
        for obj in env.cameras:
            if obj.id == object_id:
                return CameraObject(
                    obj, _create_sentinal=_CREATE_SENTINAL, env=self._env
                )
        return None

    def constraint(self, *constraints: SceneConstraint) -> None:
        self._env.constraints.extend(constraints)

    def dump_env_json(
        self,
        *,
        indent: int = 2,
        include_resolved_assets: bool = False,
        include_runtime_state: bool = False,
        validation: Literal["none", "warn", "raise"] = "warn",
    ) -> str:
        return self._world_maker.dump_env_json(
            self._env.id,
            indent=indent,
            include_resolved_assets=include_resolved_assets,
            include_runtime_state=include_runtime_state,
            validation=validation,
        )

    def save_env(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        indent: int = 2,
        include_resolved_assets: bool = False,
        include_runtime_state: bool = False,
        validation: Literal["none", "warn", "raise"] = "warn",
    ) -> Path:
        return self._world_maker.save_env(
            self._env.id,
            path,
            overwrite=overwrite,
            indent=indent,
            include_resolved_assets=include_resolved_assets,
            include_runtime_state=include_runtime_state,
            validation=validation,
        )


class StuffObject:
    def __init__(
        self,
        entity: EnvironmentStuffEntity,
        /,
        *,
        _create_sentinal: object,
        env: Environment,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")

        self._entity = entity
        self._env = env

    # region TODO: @gangjeuk
    # [ ] - check collider, joint, surface exist
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
        return JointRef(self._entity.id, name)

    def anchor(self, name: str) -> AnchorRef:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        return AnchorRef(self._entity.id, name)

    # end-region

    def set_mass(self, mass: RandomizableFloat) -> None:
        self._env._assert_mutable()
        self._entity.mass = mass

    def set_pos(self, pos: RandomizableVec3) -> None:
        self._env._assert_mutable()
        self._entity.pos = pos

    def set_rot(self, rot: RandomizableVec3) -> None:
        self._env._assert_mutable()
        self._entity.rot = rot

    def set_size(self, size: RandomizableVec3) -> None:
        self._env._assert_mutable()
        self._entity.size = size

    def set_fixed(self, is_fixed: bool) -> None:
        self._env._assert_mutable()
        self._entity.fixed = is_fixed

    def set_friction(self, friction: RandomizableFloat) -> None:
        self._env._assert_mutable()
        self._entity.friction = friction


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
        self._env = env

    def set_pos(self, pos: RandomizableVec3) -> None:
        self._env._assert_mutable()
        self._entity.pos = pos

    def set_rot(self, rot: RandomizableVec3) -> None:
        self._env._assert_mutable()
        self._entity.rot = rot

    def set_joint_pos(self, pos: Randomizable[ActionT]) -> None:
        self._env._assert_mutable()
        self._entity.init_position = pos

    # TODO: @gangjeuk
    # Need discussion
    # For implementation, we need to parse asset a `built-time`, which increases cost and responsibility of `RobotObject`
    def get_joint_min(self) -> ActionT: ...
    def get_joint_max(self) -> ActionT: ...

    """
    See comments on `StuffObject`
    """

    def joint(self, name: str) -> JointRef:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        return JointRef(self._entity.id, name)

    def collider(self, name: str) -> ColliderRef:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        return ColliderRef(self._entity.id, name)

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
        env: IEnvironment,
    ) -> None:
        if _create_sentinal is not _CREATE_SENTINAL:
            raise SimulacBaseError("Please do not create stuff object directly")
        self._entity = entity
        self._env = env

    def set_pos(self, pos: RandomizableVec3) -> None:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        self._env.relations.append(SetCameraPosOp(CameraRef(self._entity.id), pos))

    def set_rot(self, rot: RandomizableVec3) -> None:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        self._env.relations.append(SetCameraRotOp(CameraRef(self._entity.id), rot))

    def set_fov(self, fov: RandomizableFloat) -> None:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")
        self._env.relations.append(SetCameraFovOp(CameraRef(self._entity.id), fov))

    def _set_aspect(self, aspect: RandomizableFloat) -> None: ...
    def _set_near(self, near: RandomizableFloat) -> None: ...
    def _set_far(self, far: RandomizableFloat) -> None: ...

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
        up: Vec3 = (0, 0, 1),
        offset: RandomizableVec3 = (0, 0, 0),
    ) -> None:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")

        target_ref = target
        if isinstance(target, tuple):
            target_ref = WorldPointRef(target)

        self._env.relations.append(
            LookAtOp(
                EntityRef(self._entity.id),
                as_place_target(target_ref),
                up=up,
                offset=offset,
            )
        )

    def attach_to(
        self,
        parent: AnchorRef,
        *,
        offset: RandomizableVec3 = (0, 0, 0),
        rot: RandomizableVec3 = (0, 0, 0),
    ) -> None:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")

        self._env.relations.append(
            AttachOp(
                EntityRef(self._entity.id),
                parent,
                offset=offset,
                rot=rot,
            )
        )

    def follow(
        self,
        target: AnchorRef | ColliderRef | RobotObject[Any] | StuffObject,
        *,
        offset: RandomizableVec3 = (0, 0, 0),
        frame: Literal["world", "local"] = "world",
    ) -> None:
        if self._entity.id is None:
            raise SimulacBaseError("Entity must be added to Environment first")

        if isinstance(target, (RobotObject, StuffObject)):
            if target._entity.id is None:
                raise SimulacBaseError("Follow target entity must be added first")
            target_ref = EntityRef(target._entity.id)
        else:
            target_ref = target

        self._env.relations.append(
            FollowOp(
                EntityRef(self._entity.id),
                target_ref,
                offset=offset,
                frame=frame,
            )
        )


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
