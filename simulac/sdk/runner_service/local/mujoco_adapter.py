from __future__ import annotations

from typing import TYPE_CHECKING, Literal, MutableMapping, cast

import mujoco

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.runner_service.common.physics_engine_adapter import (
    IPhysicsEngineAdapter,
    IPhysicsEngineAdapterState,
)
from simulac.sdk.runner_service.local.mujoco.binding import (
    MujocoActuatorBinding,
    MujocoCameraBinding,
    MujocoGeomBinding,
    MujocoJointBinding,
    MujocoLightBinding,
    MujocoLinkBinding,
    MujocoRobotBinding,
    MujocoSensorBinding,
    MujocoSiteBinding,
    MujocoStuffBinding,
)

from .mujoco.runner import MujocoRunner

if TYPE_CHECKING:
    from simulac.sdk.environment_service.common.environment_service import (
        IEnvironmentManagementService,
    )
    from simulac.sdk.environment_service.common.model.entity import (
        AmbientLightSpec,
        AreaLightSpec,
        EnvironmentCameraEntity,
        EnvironmentLightEntity,
        EnvironmentMachineEntity,
        EnvironmentStuffEntity,
        PointLightSpec,
        SpotLightSpec,
    )
    from simulac.sdk.log_service.common.log_service import ILogService
    from simulac.sdk.runner_service.common.runner import IRunner
    from simulac.sdk.runner_service.common.runner_service import (
        IRunnerManagementService,
    )

MUJOCO_SCENE = """
<mujoco model="scene">
  <statistic center="0.3 0 0.4" extent="1"/>

  <option timestep="0.005" iterations="5" ls_iterations="8" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <custom>
    <numeric data="12" name="max_contact_points"/>
  </custom>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
    <scale contactwidth="0.075" contactheight="0.025" forcewidth="0.05" com="0.05" framewidth="0.01" framelength="0.2"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1"/>
  </worldbody>
</mujoco>
"""


def _subtree_body_ids(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    body_ids: list[int] = []
    for bid in range(model.nbody):
        cur = bid
        while cur != 0:
            if cur == root_body_id:
                body_ids.append(bid)
                break
            cur = int(model.body_parentid[cur])
    return body_ids


class MujocoAdapter(IPhysicsEngineAdapter):
    """_summary_
        ![test](https://picsum.photos/200/300)

    Args:
        PhysicsEngineAdapter (_type_): _description_
    """

    def __init__(
        self,
        env_id: str,
        LogService: ILogService,
        RunnerManagementService: IRunnerManagementService,
        EnvironmentManagementService: IEnvironmentManagementService,
        *,
        tick_dt_ms: int | None = 5,
    ) -> None:

        self.env_id = env_id
        self.LogService = LogService
        self.RunnerManagementService = RunnerManagementService
        self.EnvironmentManagementService = EnvironmentManagementService

        self._runner_count = 0
        self._step_count = 0
        self._step_count_map: MutableMapping[str, int] = dict()

        self.root_spec = mujoco.MjSpec.from_string(MUJOCO_SCENE)
        self.root_spec.option.timestep = (tick_dt_ms or 5) / 1000.0
        self.root_frame = self.root_spec.worldbody.add_frame()

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._stuff_bindings: dict[str, MujocoStuffBinding] = {}
        self._machine_bindings: dict[str, MujocoRobotBinding] = {}
        self._camera_bindings: dict[str, MujocoCameraBinding] = {}
        self._light_bindings: dict[str, MujocoLightBinding] = {}
        env_ret = self.EnvironmentManagementService.get_environment(self.env_id)

        if env_ret[0] is None:
            raise env_ret[1]

        env = env_ret[0]
        self.env = env

        self._stuff_entities = {e.id: e for e in env.stuffs if e.id is not None}
        self._machine_entities = {e.id: e for e in env.machines if e.id is not None}
        self._camera_entities = {e.id: e for e in env.cameras if e.id is not None}
        self._light_entities = {e.id: e for e in env.lights if e.id is not None}
        for stuff in env.stuffs:
            self.__add_stuff(stuff)
        for machine in env.machines:
            self.__add_machine(machine)
        for camera in env.cameras:
            self.__add_camera(camera)
        for light in env.lights:
            self.__add_light(light)
        """
        TODO: @gangjeuk
            1. implement mujoco parallel
            2. [o] - implement robos initialization code (27/04/2026)
            
        """
        self.model = self.root_spec.compile()
        for stuff in env.stuffs:
            self._stuff_bindings[stuff.id] = self.__build_stuff_binding(
                stuff.id,
            )
        for machine in env.machines:
            self._machine_bindings[machine.id] = self.__build_machine_binding(
                machine.id,
            )
        for camera in env.cameras:
            self._camera_bindings[camera.id] = self.__build_camera_binding(camera.id)
        for light in env.lights:
            self._light_bindings[light.id] = self.__build_light_binding(light.id)

    def create_runner(self) -> IRunner:
        if self._step_count != 0:
            raise SimulacBaseError(
                "Cannot create new runner after calling step() function"
            )

        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        def on_after_runner_step(runner_id: str):
            self._step_count_map[runner_id] += 1
            self._step_count += 1

        new_runner_id = f"run_{self._runner_count}"

        runner = MujocoRunner(
            LogService=self.LogService,
            runner_id=new_runner_id,
            env=self.env,
            mj_model=self.model,
            stuff_entities=self._stuff_entities,
            camera_entities=self._camera_entities,
            machine_entities=self._machine_entities,
            light_entities=self._light_entities,
            stuff_bindings=self._stuff_bindings,
            machine_bindings=self._machine_bindings,
            camera_bindins=self._camera_bindings,
            light_bindings=self._light_bindings,
            on_after_call_step=on_after_runner_step,
        )

        self.LogService.debug(
            f"new mujoco runner created env_id: {self.env_id} runner_id: {new_runner_id}"
        )
        self._step_count_map[new_runner_id] = 0
        self._runner_count += 1

        return runner

    def get_state(self) -> IPhysicsEngineAdapterState:
        return IPhysicsEngineAdapterState(
            self.env_id, self._runner_count, self._step_count_map
        )

    def __prepare_stuff_root(
        self,
        child: mujoco.MjSpec,
        entity_id: str,
        *,
        fixed: bool | None,
    ) -> None:
        bodies: list[mujoco.MjsBody] = child.bodies
        roots = [body for body in bodies if body.parent == child.worldbody]

        if len(roots) != 1:
            raise SimulacBaseError(
                f"MJCF asset for {entity_id!r} must have one root body"
            )

        root = roots[0]
        root_name = root.name
        root.name = "__root__"

        # Rename body references
        if root_name and root_name != root.name:
            # flexcomp expands into a flex whose vertices reference generated body
            # names. If the root body is renamed, these references must follow it.
            for flex in child.flexes:
                for body_refs in (flex.vertbody, flex.nodebody):
                    for idx in range(len(body_refs)):
                        if body_refs[idx] == root_name:
                            body_refs[idx] = root.name

        root_freejoints: list[mujoco.MjsJoint] = [
            joint
            for joint in cast(list[mujoco.MjsJoint], root.joints)
            if int(joint.type) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]

        if len(root_freejoints) > 1:
            raise SimulacBaseError(
                f"Stuff asset {entity_id!r} has multiple root freejoints. "
                "Simulac can only manage one root freejoint."
            )

        if fixed is None:
            # use asset setting
            return
        if fixed:
            # force un-fixed stuff to fixed
            for joint in root_freejoints:
                joint_name = joint.name
                if not joint_name:
                    raise SimulacBaseError(
                        f"Stuff asset {entity_id!r} has an unnamed root freejoint. "
                        "Cannot safely remove it for fixed=True."
                    )

                references: list[str] = []

                for actuator in child.actuators:
                    actuator = cast(mujoco.MjsActuator, actuator)
                    actuator_joint: mujoco.MjsJoint = getattr(actuator, "joint", None)
                    if actuator_joint == joint or actuator_joint == joint_name:
                        references.append(f"actuator:{actuator.name or '<unnamed>'}")

                for sensor in child.sensors:
                    sensor = cast(mujoco.MjsSensor, sensor)
                    sensor_joint = getattr(sensor, "joint", None)
                    if sensor_joint == joint or sensor_joint == joint_name:
                        references.append(f"sensor:{sensor.name or '<unnamed>'}")

                for equality in child.equalities:
                    equality = cast(mujoco.MjsEquality, equality)
                    equality_joint1 = getattr(equality, "joint1", None)
                    equality_joint2 = getattr(equality, "joint2", None)
                    if (
                        equality_joint1 == joint
                        or equality_joint2 == joint
                        or equality_joint1 == joint_name
                        or equality_joint2 == joint_name
                    ):
                        references.append(f"equality:{equality.name or '<unnamed>'}")

                if references:
                    raise SimulacBaseError(
                        "\n".join(
                            [
                                (
                                    f"Stuff asset {entity_id!r} was added with fixed=True, "
                                    f"but its root freejoint {joint_name!r} is referenced."
                                ),
                                "Simulac cannot safely remove this freejoint automatically.",
                                f"References: {', '.join(references)}",
                                "Remove the root freejoint references from the asset or use fixed=False.",
                            ]
                        )
                    )
                joint_name = joint.name or "<unnamed>"
                child.delete(joint)
                self.LogService.warn(
                    f"Removed root freejoint {joint_name!r} from Stuff asset {entity_id!r}"
                )

        if not fixed and len(root_freejoints) == 0:
            # force fixed stuff to un-fixed
            root.add_freejoint(name="root_freejoint")

    def __add_stuff(self, stuff: EnvironmentStuffEntity):
        # TODO: URDF file is not handled here. Need handling code
        child = mujoco.MjSpec.from_file(stuff.asset_uri)
        self.__prepare_stuff_root(child, stuff.id, fixed=stuff.fixed)

        self.root_spec.attach(
            child, frame=self.root_frame, prefix=f"{stuff.id}/", suffix=""
        )

    def __add_machine(self, machine: EnvironmentMachineEntity):
        child = mujoco.MjSpec.from_file(machine.asset_uri)
        # TODO: @gangjeuk
        # handle machine.pos, machine.quat
        bodies: list[mujoco.MjsBody] = child.bodies
        roots = [body for body in bodies if body.parent == child.worldbody]

        if len(roots) != 1:
            raise SimulacBaseError(
                f"MJCF asset for Robot {machine.id!r} must have one root body"
            )

        root = roots[0]
        root.name = "__root__"

        self.root_spec.attach(
            child, frame=self.root_frame, prefix=f"{machine.id}/", suffix=""
        )

    def __add_camera(self, camera: EnvironmentCameraEntity) -> None:
        """add camera as mujoco.mjSpec

        If works like adding entity below
        ```xml
        <body name="front_rgb/__root__" pos="0 0 0">
            <camera name="front_rgb/__root__camera" pos="0 0 0" quat="1 0 0 0" fovy="45"/>
        </body>
        ```
        """
        if camera.id is None:
            raise SimulacBaseError("Camera entity id is required")

        body = self.root_spec.worldbody.add_body(
            name=f"{camera.id}/__root__",
            pos=(0.0, 0.0, 0.0),
        )

        body.add_camera(
            name=f"{camera.id}/__root__camera",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            fovy=float(camera.spec.fov),
        )

    def __add_light(self, light: EnvironmentLightEntity) -> None:
        if light.id is None:
            raise SimulacBaseError("Light entity id is required")

        body = self.root_spec.worldbody.add_body(
            name=f"{light.id}/__root__",
            pos=(0.0, 0.0, 0.0),
        )

        spec = light.spec
        rgb = self.__light_rgb(spec.color)
        diffuse, ambient, specular = self.__mujoco_light_colors(spec)

        mj_light = body.add_light(
            name=f"{light.id}/__root__",
            pos=(0.0, 0.0, 0.0),
            dir=(0.0, 0.0, -1.0),
            active=bool(spec.enabled),
            diffuse=diffuse,
            ambient=ambient,
            specular=specular,
        )

        if spec.type == "ambient":
            # Mocking ambient light
            mj_light.type = mujoco.mjtLightType.mjLIGHT_POINT
            mj_light.diffuse = (0.0, 0.0, 0.0)
            mj_light.specular = (0.0, 0.0, 0.0)
            mj_light.ambient = tuple(rgb[i] * float(spec.intensity) for i in range(3))

        elif spec.type == "pointlight":
            mj_light.type = mujoco.mjtLightType.mjLIGHT_POINT
            if spec.range is not None:
                mj_light.range = float(spec.range)
            mj_light.attenuation = self.__mujoco_attenuation(spec.decay)

        elif spec.type == "spot":
            mj_light.type = mujoco.mjtLightType.mjLIGHT_SPOT
            mj_light.cutoff = float(spec.angle)
            mj_light.exponent = max(float(spec.penumbra), 0.0)
            if spec.range is not None:
                mj_light.range = float(spec.range)
            mj_light.attenuation = self.__mujoco_attenuation(spec.decay)

        elif spec.type == "area":
            # MuJoCo has no rectangular area light.
            # Approximate with point light + bulb radius.
            mj_light.type = mujoco.mjtLightType.mjLIGHT_POINT
            mj_light.bulbradius = max(float(spec.width), float(spec.height)) * 0.5

        else:
            raise SimulacBaseError(f"Unsupported light type: {spec.type!r}")

    def __light_rgb(self, color: tuple[int, int, int]) -> tuple[float, float, float]:
        return (
            float(color[0]) / 255.0,
            float(color[1]) / 255.0,
            float(color[2]) / 255.0,
        )

    def __mujoco_light_colors(
        self,
        spec: AmbientLightSpec | PointLightSpec | SpotLightSpec | AreaLightSpec,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        rgb = self.__light_rgb(spec.color)
        intensity = float(spec.intensity)

        diffuse = tuple(channel * intensity for channel in rgb)
        ambient = (0.0, 0.0, 0.0)
        specular = tuple(channel * intensity * 0.3 for channel in rgb)

        return diffuse, ambient, specular

    def __mujoco_attenuation(self, decay: float) -> tuple[float, float, float]:
        if decay <= 0.0:
            return (1.0, 0.0, 0.0)
        if decay <= 1.0:
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def __actuator_target(
        self,
        actuator_id: int,
    ) -> tuple[
        Literal["joint", "tendon", "site", "body", "unknown"], int | None, str | None
    ]:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        model = self.model
        trn_type = int(model.actuator_trntype[actuator_id])
        target_id = int(model.actuator_trnid[actuator_id][0])

        if target_id < 0:
            return "unknown", None, None

        if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
            return (
                "joint",
                target_id,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, target_id),
            )

        if trn_type == mujoco.mjtTrn.mjTRN_TENDON:
            return (
                "tendon",
                target_id,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, target_id),
            )

        if trn_type == mujoco.mjtTrn.mjTRN_SITE:
            return (
                "site",
                target_id,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, target_id),
            )

        if trn_type == mujoco.mjtTrn.mjTRN_BODY:
            return (
                "body",
                target_id,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, target_id),
            )

        return "unknown", target_id, None

    def __build_machine_binding(self, entity_id: str) -> MujocoRobotBinding:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        model = self.model
        root_name = f"{entity_id}/__root__"
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)

        if root_body_id < 0:
            raise SimulacBaseError(f"No MuJoCo root body for machine {entity_id!r}")

        body_ids = _subtree_body_ids(model, root_body_id)
        geom_ids = [
            gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in body_ids
        ]
        joint_ids = [
            jid for jid in range(model.njnt) if int(model.jnt_bodyid[jid]) in body_ids
        ]
        actuator_ids: list[int] = []
        for aid in range(model.nu):
            target_type, target_id, _ = self.__actuator_target(aid)
            if target_type == "joint" and target_id in joint_ids:
                actuator_ids.append(aid)
            elif target_type == "body" and target_id in body_ids:
                actuator_ids.append(aid)
            elif target_type in {"site", "tendon"}:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                if name and name.startswith(f"{entity_id}/"):
                    actuator_ids.append(aid)

        root_freejoint_id = -1

        for jid in joint_ids:
            if (
                int(model.jnt_bodyid[jid]) == root_body_id
                and int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_FREE
            ):
                root_freejoint_id = jid
                break

        links: dict[str, MujocoLinkBinding] = {}
        for body_id in body_ids:
            body_full_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or f"body_{body_id}"
            )
            body_name = (
                body_full_name.split("/", 1)[1]
                if body_full_name.startswith(f"{entity_id}/")
                else body_full_name
            )
            body_geom_ids = [
                gid for gid in geom_ids if int(model.geom_bodyid[gid]) == body_id
            ]
            body_joint_ids = [
                jid for jid in joint_ids if int(model.jnt_bodyid[jid]) == body_id
            ]
            child_body_ids = [
                bid for bid in body_ids if int(model.body_parentid[bid]) == body_id
            ]

            links[body_name] = MujocoLinkBinding(
                entity_id=entity_id,
                full_name=body_full_name,
                name=body_name,
                body_id=body_id,
                parent_body_id=int(model.body_parentid[body_id]),
                child_body_ids=child_body_ids,
                geom_ids=body_geom_ids,
                joint_ids=body_joint_ids,
                mocap_id=int(model.body_mocapid[body_id]),
            )

        joints: dict[str, MujocoJointBinding] = self.__build_joint_bindings(
            entity_id, joint_ids
        )

        actuators: dict[str, MujocoActuatorBinding] = {}
        for actuator_id in actuator_ids:
            actuator_full_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                or f"actuator_{actuator_id}"
            )
            actuator_name = (
                actuator_full_name.split("/", 1)[1]
                if actuator_full_name.startswith(f"{entity_id}/")
                else actuator_full_name
            )

            target_type, target_id, target_name = self.__actuator_target(actuator_id)

            ctrl_range = None
            if bool(model.actuator_ctrllimited[actuator_id]):
                ctrl_range = (
                    float(model.actuator_ctrlrange[actuator_id][0]),
                    float(model.actuator_ctrlrange[actuator_id][1]),
                )

            force_range = None
            if bool(model.actuator_forcelimited[actuator_id]):
                force_range = (
                    float(model.actuator_forcerange[actuator_id][0]),
                    float(model.actuator_forcerange[actuator_id][1]),
                )

            act_range = None
            if bool(model.actuator_actlimited[actuator_id]):
                act_range = (
                    float(model.actuator_actrange[actuator_id][0]),
                    float(model.actuator_actrange[actuator_id][1]),
                )

            actuators[actuator_name] = MujocoActuatorBinding(
                entity_id=entity_id,
                name=actuator_name,
                full_name=actuator_full_name,
                actuator_id=actuator_id,
                target_type=target_type,
                target_id=target_id,
                target_name=target_name,
                ctrl_range=ctrl_range,
                force_range=force_range,
                act_range=act_range,
                group=int(model.actuator_group[actuator_id]),
            )

            if target_type == "joint" and target_id is not None:
                for joint_binding in joints.values():
                    if joint_binding.joint_id == target_id:
                        joint_binding.actuator_ids.append(actuator_id)
                        break
        site_ids, sites = self.__build_site_bindings(entity_id, body_ids)
        geoms = self.__build_geom_bindings(entity_id, body_ids, geom_ids)
        sensors = self.__build_sensor_bindings(
            entity_id,
            body_ids=body_ids,
            geom_ids=geom_ids,
            site_ids=site_ids,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
        )
        return MujocoRobotBinding(
            entity_id=entity_id,
            name=entity_id,
            full_name=entity_id,
            root_body_id=root_body_id,
            root_body_name="__root__",
            root_body_full_name=root_name,
            body_ids=body_ids,
            geom_ids=geom_ids,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
            site_ids=site_ids,
            joints=joints,
            actuators=actuators,
            links=links,
            geoms=geoms,
            sites=sites,
            sensors=sensors,
            root_freejoint_id=root_freejoint_id,
            mocap_id=int(model.body_mocapid[root_body_id]),
        )

    def __build_stuff_binding(
        self,
        entity_id: str,
    ) -> MujocoStuffBinding:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        model = self.model

        root_name = f"{entity_id}/__root__"
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)

        if root_body_id < 0:
            raise SimulacBaseError(f"No MuJoCo root body for entity {entity_id!r}")

        body_ids = _subtree_body_ids(model, root_body_id)
        geom_ids = [
            gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in body_ids
        ]
        joint_ids = [
            jid for jid in range(model.njnt) if int(model.jnt_bodyid[jid]) in body_ids
        ]
        actuator_ids: list[int] = []
        for aid in range(model.nu):
            target_type, target_id, _ = self.__actuator_target(aid)
            if target_type == "joint" and target_id in joint_ids:
                actuator_ids.append(aid)
            elif target_type == "body" and target_id in body_ids:
                actuator_ids.append(aid)
            elif target_type in {"site", "tendon"}:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                if name and name.startswith(f"{entity_id}/"):
                    actuator_ids.append(aid)

        root_freejoint_id = -1
        for jid in joint_ids:
            if (
                int(model.jnt_bodyid[jid]) == root_body_id
                and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
            ):
                root_freejoint_id = jid
                break
        site_ids, sites = self.__build_site_bindings(entity_id, body_ids)
        geoms = self.__build_geom_bindings(entity_id, body_ids, geom_ids)
        sensors = self.__build_sensor_bindings(
            entity_id,
            body_ids=body_ids,
            geom_ids=geom_ids,
            site_ids=site_ids,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
        )
        joints = self.__build_joint_bindings(entity_id, joint_ids)
        return MujocoStuffBinding(
            entity_id=entity_id,
            root_body_id=root_body_id,
            body_ids=body_ids,
            geom_ids=geom_ids,
            joint_ids=joint_ids,
            site_ids=site_ids,
            actuator_ids=actuator_ids,
            geoms=geoms,
            sites=sites,
            sensors=sensors,
            root_freejoint_id=root_freejoint_id,
            joints=joints,
            mocap_id=int(self.model.body_mocapid[root_body_id]),
        )

    def __build_camera_binding(self, entity_id: str) -> MujocoCameraBinding:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        root_body_full_name = f"{entity_id}/__root__"
        camera_full_name = f"{entity_id}/__root__camera"

        root_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            root_body_full_name,
        )
        if root_body_id < 0:
            raise SimulacBaseError(f"No MuJoCo camera root body for {entity_id!r}")

        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_full_name,
        )
        if camera_id < 0:
            raise SimulacBaseError(f"No MuJoCo camera for {entity_id!r}")

        return MujocoCameraBinding(
            entity_id=entity_id,
            root_body_id=root_body_id,
            root_body_name="__root__",
            root_body_full_name=root_body_full_name,
            camera_id=camera_id,
            camera_name="__root__camera",
            camera_full_name=camera_full_name,
            mocap_id=int(self.model.body_mocapid[root_body_id]),
        )

    def __build_light_binding(self, entity_id: str) -> MujocoLightBinding:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        root_body_full_name = f"{entity_id}/__root__"
        light_full_name = f"{entity_id}/__root__"

        root_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            root_body_full_name,
        )
        if root_body_id < 0:
            raise SimulacBaseError(f"No MuJoCo light root body for {entity_id!r}")

        light_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_LIGHT,
            light_full_name,
        )
        if light_id < 0:
            raise SimulacBaseError(f"No MuJoCo light for {entity_id!r}")

        return MujocoLightBinding(
            entity_id=entity_id,
            root_body_id=root_body_id,
            root_body_name="__root__",
            root_body_full_name=root_body_full_name,
            light_id=light_id,
            light_name="__root__",
            light_full_name=light_full_name,
            light_type=self._light_entities[entity_id].spec.type,
            mocap_id=int(self.model.body_mocapid[root_body_id]),
        )

    def __local_name(self, entity_id: str, full_name: str) -> str:
        return (
            full_name.split("/", 1)[1]
            if full_name.startswith(f"{entity_id}/")
            else full_name
        )

    def __build_geom_bindings(
        self,
        entity_id: str,
        body_ids: list[int],
        geom_ids: list[int],
    ) -> dict[str, MujocoGeomBinding]:
        model = self.model
        if model is None:
            raise SimulacBaseError("Adapter not initialized")

        geoms: dict[str, MujocoGeomBinding] = {}
        for geom_id in geom_ids:
            full_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                or f"geom_{geom_id}"
            )
            name = self.__local_name(entity_id, full_name)
            geoms[name] = MujocoGeomBinding(
                entity_id=entity_id,
                full_name=full_name,
                name=name,
                geom_id=geom_id,
                body_id=int(model.geom_bodyid[geom_id]),
            )
        return geoms

    def __build_site_bindings(
        self,
        entity_id: str,
        body_ids: list[int],
    ) -> tuple[list[int], dict[str, MujocoSiteBinding]]:
        model = self.model
        if model is None:
            raise SimulacBaseError("Adapter not initialized")

        site_ids = [
            sid for sid in range(model.nsite) if int(model.site_bodyid[sid]) in body_ids
        ]
        sites: dict[str, MujocoSiteBinding] = {}
        for site_id in site_ids:
            full_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
                or f"site_{site_id}"
            )
            name = self.__local_name(entity_id, full_name)
            sites[name] = MujocoSiteBinding(
                entity_id=entity_id,
                full_name=full_name,
                name=name,
                site_id=site_id,
                body_id=int(model.site_bodyid[site_id]),
            )
        return site_ids, sites

    def __build_sensor_bindings(
        self,
        entity_id: str,
        *,
        body_ids: list[int],
        geom_ids: list[int],
        site_ids: list[int],
        joint_ids: list[int],
        actuator_ids: list[int],
    ) -> dict[str, MujocoSensorBinding]:
        model = self.model
        if model is None:
            raise SimulacBaseError("Adapter not initialized")

        sensors: dict[str, MujocoSensorBinding] = {}
        for sensor_id in range(model.nsensor):
            full_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
                or f"sensor_{sensor_id}"
            )

            obj_type = int(model.sensor_objtype[sensor_id])
            obj_id = int(model.sensor_objid[sensor_id])

            belongs_by_name = full_name.startswith(f"{entity_id}/")
            belongs_by_target = (
                (obj_type == int(mujoco.mjtObj.mjOBJ_BODY) and obj_id in body_ids)
                or (obj_type == int(mujoco.mjtObj.mjOBJ_GEOM) and obj_id in geom_ids)
                or (obj_type == int(mujoco.mjtObj.mjOBJ_SITE) and obj_id in site_ids)
                or (obj_type == int(mujoco.mjtObj.mjOBJ_JOINT) and obj_id in joint_ids)
                or (
                    obj_type == int(mujoco.mjtObj.mjOBJ_ACTUATOR)
                    and obj_id in actuator_ids
                )
            )

            if not belongs_by_name and not belongs_by_target:
                continue

            name = self.__local_name(entity_id, full_name)
            sensors[name] = MujocoSensorBinding(
                entity_id=entity_id,
                full_name=full_name,
                name=name,
                sensor_id=sensor_id,
                sensor_type=int(model.sensor_type[sensor_id]),
                obj_type=obj_type,
                obj_id=obj_id,
                adr=int(model.sensor_adr[sensor_id]),
                dim=int(model.sensor_dim[sensor_id]),
            )

        return sensors

    def __build_joint_bindings(
        self,
        entity_id: str,
        joint_ids: list[int],
    ) -> dict[str, MujocoJointBinding]:
        if self.model is None:
            raise SimulacBaseError("Adapter not initialized")

        joints: dict[str, MujocoJointBinding] = {}

        for joint_id in joint_ids:
            joint_full_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                or f"joint_{joint_id}"
            )
            joint_name = (
                joint_full_name.split("/", 1)[1]
                if joint_full_name.startswith(f"{entity_id}/")
                else joint_full_name
            )

            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                qpos_dim, qvel_dim = 7, 6
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                qpos_dim, qvel_dim = 4, 3
            else:
                qpos_dim, qvel_dim = 1, 1

            limited = bool(self.model.jnt_limited[joint_id])
            joint_range = (
                (
                    float(self.model.jnt_range[joint_id][0]),
                    float(self.model.jnt_range[joint_id][1]),
                )
                if limited
                else None
            )

            joints[joint_name] = MujocoJointBinding(
                entity_id=entity_id,
                full_name=joint_full_name,
                name=joint_name,
                joint_id=joint_id,
                body_id=int(self.model.jnt_bodyid[joint_id]),
                joint_type=joint_type,
                qpos_addr=int(self.model.jnt_qposadr[joint_id]),
                qvel_addr=int(self.model.jnt_dofadr[joint_id]),
                qpos_dim=qpos_dim,
                qvel_dim=qvel_dim,
                axis=(
                    float(self.model.jnt_axis[joint_id][0]),
                    float(self.model.jnt_axis[joint_id][1]),
                    float(self.model.jnt_axis[joint_id][2]),
                ),
                range=joint_range,
                limited=limited,
            )

        return joints
