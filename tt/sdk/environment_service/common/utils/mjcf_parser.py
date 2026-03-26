from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, MutableMapping, Tuple

import mujoco

from tt.base.error.error import TektonianBaseError


@dataclass
class ParsedJoint:
    tid: str
    name: str
    armature: float
    damping: float
    frictionloss: float
    stiffness: float
    type: int
    init_posture: float

    def to_dict(self):
        ret = dict(
            tid=self.tid,
            name=self.name,
            armature=self.armature,
            damping=self.damping,
            frictionloss=self.frictionloss,
            stiffness=self.stiffness,
            type=self.type,
            init_posture=self.init_posture,
        )
        return ret


@dataclass
class ParsedGeom:
    tid: str
    name: str
    solimp: Tuple[float, float, float, float, float]
    solmix: float
    solref: Tuple[float, float]
    friction: Tuple[float, float, float]

    def to_dict(self):
        ret = dict(
            tid=self.tid,
            name=self.name,
            solimp=self.solimp,
            solmix=self.solmix,
            solref=self.solref,
            friction=self.friction,
        )
        return ret


@dataclass
class ParsedBody:
    tid: str
    name: str
    mass: float
    joints: List[ParsedJoint]
    geoms: List[ParsedGeom]
    body: ParsedBody | None

    def to_dict(self):
        ret = dict(
            tid=self.tid,
            name=self.name,
            joints=[joint.to_dict() for joint in self.joints],
            geoms=[geom.to_dict() for geom in self.geoms],
            mass=self.mass,
            body=None if self.body is None else self.body.to_dict(),
        )
        return ret


@dataclass
class ParsedRender:
    tid: str
    geom_tid: str
    material_uri: str | None
    mesh_uri: str

    # TODO: rgbs,metalness, etc...

    def to_dict(self):
        return dict(
            tid=self.tid,
            geom_tid=self.geom_tid,
            material_uri=self.material_uri,
            mesh_uri=self.mesh_uri,
        )


def _get_asset_name_path_map(uri: str) -> dict[str, str]:
    return {}


def parse_mjcf_path(uri: str):
    tree = ET.parse(uri)
    root = tree.getroot()

    if root.tag != "mujoco":
        raise TektonianBaseError(f"File seems like non mjcf: {uri}")

    asset_root_path: str | None = None

    for child in root:
        # check <compiler />
        if child.tag == "compiler":
            if "meshdir" in child.attrib.keys():
                asset_root_path = child.attrib["meshdir"]

    # check <asset />
    assets = root.find("asset")
    materials_map: MutableMapping[str, str] = dict()
    mesh_map: MutableMapping[str, str] = dict()
    if assets:
        textures_map: MutableMapping[str, str] = dict()
        for asset in assets:
            if asset.tag == "texture":
                if "file" in asset.attrib.keys():
                    name = (
                        asset.attrib["name"]
                        if "name" in asset.attrib.keys()
                        else asset.attrib["file"].split(".")[-2]
                    )
                    textures_map[name] = asset.attrib["file"]
            elif asset.tag == "mesh":
                name = (
                    asset.attrib["name"]
                    if "name" in asset.attrib.keys()
                    else asset.attrib["file"].split(".")[-2]
                )
                mesh_map[name] = asset.attrib["file"]
        for asset in assets:
            if asset.tag == "material":
                name = asset.attrib["name"]
                if "texture" in asset.attrib.keys():
                    texture_name = asset.attrib["texture"]
                    materials_map[name] = textures_map[texture_name]
                else:
                    # TODO: add rgba, metalness parsing
                    materials_map[name] = "todo"

    # check <worldbody />
    model = mujoco.MjModel.from_xml_path(uri)
    num_body = model.nbody

    bodies_map: MutableMapping[int, ParsedBody] = {}

    render_ret = []

    for body_idx in range(num_body):
        if body_idx == 0:
            # pass n==0. idx==0 means <worldbody /> tag
            continue

        body: mujoco._structs._MjModelBodyViews = model.body(body_idx)

        parsed_body = ParsedBody(
            f"body.{body_idx}", body.name, body.mass[0], [], [], None
        )

        jnt_idx = body.jntadr[0]
        jnt_cnt = body.jntnum[0]

        if jnt_idx != -1:
            # -1 means no joint
            for cnt in range(jnt_cnt):
                joint: mujoco._structs._MjModelJointViews = model.joint(jnt_idx + cnt)
                parsed_joint = ParsedJoint(
                    f"body.{body_idx}.joints.{cnt}",
                    joint.name,
                    joint.armature[0],
                    joint.damping[0],
                    joint.frictionloss[0],
                    joint.stiffness[0],
                    int(joint.type[0]),
                    joint.qpos0[0],
                )
                parsed_body.joints.append(parsed_joint)

        geom_idx = body.geomadr[0]
        geom_cnt = body.geomnum[0]

        if geom_idx != -1:
            # Also, -1 means no geom
            for cnt in range(geom_cnt):
                geom: mujoco._structs._MjModelGeomViews = model.geom(geom_idx + cnt)
                geom_id = f"body.{body_idx}.geoms.{cnt}"
                parsed_geom = ParsedGeom(
                    geom_id,
                    geom.name,
                    tuple(geom.solimp.tolist()),
                    geom.solmix[0],
                    tuple(geom.solref.tolist()),
                    tuple(geom.friction.tolist()),
                )
                parsed_body.geoms.append(parsed_geom)

                if geom.type[0] == mujoco._enums.mjtGeom.mjGEOM_HFIELD:
                    continue

                parsed_render = ParsedRender(
                    f"render.{len(render_ret)}", geom_id, None, ""
                )

                mesh_id = geom.dataid[0]
                if mesh_id != -1:
                    mesh: mujoco._structs._MjModelMeshViews = model.mesh(mesh_id)
                    parsed_render.mesh_uri = mesh_map[mesh.name]

                material_id = geom.matid[0]
                if material_id != -1:
                    material: mujoco._structs._MjModelMaterialViews = model.mat(
                        material_id
                    )
                    parsed_render.material_uri = materials_map[material.name]

                render_ret.append(parsed_render.to_dict())

        bodies_map[body_idx] = parsed_body

        if body.parentid[0] != 0:
            # == 0 means <worldbody /> tag
            parent_id = body.parentid[0]
            bodies_map[parent_id].body = parsed_body

    # Get first body. Id of the body should be 1
    ret = bodies_map[1]

    return ret, render_ret
