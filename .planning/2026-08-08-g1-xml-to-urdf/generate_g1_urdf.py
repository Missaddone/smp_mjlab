from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[3]
ROBOT_REPO = TASK_ROOT / "robot_retargeter"
DEFAULT_SOURCE = ROBOT_REPO / "asset/robot/g1_description/mjcf/g1.xml"
DEFAULT_OUTPUT = ROBOT_REPO / "asset/robot/g1_description/g1.urdf"
SUPPORTED_GEOMETRIES = {"box", "cylinder", "mesh", "sphere"}


def parse_floats(value: str | None, default: str) -> list[float]:
    return [float(part) for part in (value if value is not None else default).split()]


def format_floats(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def quat_wxyz_to_rpy(quat: list[float]) -> list[float]:
    if len(quat) != 4:
        raise ValueError(f"expected WXYZ quaternion, got {quat}")
    norm = math.sqrt(sum(value * value for value in quat))
    if norm == 0:
        raise ValueError("zero quaternion is invalid")
    w, x, y, z = (value / norm for value in quat)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return [roll, pitch, yaw]


def quat_wxyz_to_matrix(quat: list[float]) -> list[list[float]]:
    if len(quat) != 4:
        raise ValueError(f"expected WXYZ quaternion, got {quat}")
    norm = math.sqrt(sum(value * value for value in quat))
    if norm == 0:
        raise ValueError("zero quaternion is invalid")
    w, x, y, z = (value / norm for value in quat)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def rotate_diagonal_inertia(diagonal: list[float], quat: list[float]) -> list[list[float]]:
    rotation = quat_wxyz_to_matrix(quat)
    return [
        [sum(rotation[row][axis] * diagonal[axis] * rotation[column][axis] for axis in range(3)) for column in range(3)]
        for row in range(3)
    ]


def add_origin(parent: ET.Element, element: ET.Element) -> None:
    xyz = parse_floats(element.get("pos"), "0 0 0")
    quat = parse_floats(element.get("quat"), "1 0 0 0")
    ET.SubElement(parent, "origin", xyz=format_floats(xyz), rpy=format_floats(quat_wxyz_to_rpy(quat)))


def add_geometry(parent: ET.Element, geom: ET.Element, mesh_files: dict[str, str]) -> None:
    geometry = ET.SubElement(parent, "geometry")
    if geom.get("mesh") is not None:
        geometry_type = "mesh"
    else:
        geometry_type = geom.get("type", "sphere")
    if geometry_type not in SUPPORTED_GEOMETRIES:
        raise ValueError(f"unsupported active geometry type: {geometry_type}")

    if geometry_type == "mesh":
        mesh_name = geom.get("mesh")
        if mesh_name not in mesh_files:
            raise ValueError(f"geom references unknown mesh: {mesh_name}")
        mesh = ET.SubElement(geometry, "mesh", filename=mesh_files[mesh_name])
        if geom.get("scale") is not None:
            mesh.set("scale", format_floats(parse_floats(geom.get("scale"), "1 1 1")))
        return

    size = parse_floats(geom.get("size"), "")
    if geometry_type == "sphere":
        if len(size) != 1:
            raise ValueError(f"sphere requires one size value: {ET.tostring(geom, encoding='unicode')}")
        ET.SubElement(geometry, "sphere", radius=f"{size[0]:.12g}")
    elif geometry_type == "cylinder":
        if len(size) != 2:
            raise ValueError(f"cylinder requires radius and half-length: {ET.tostring(geom, encoding='unicode')}")
        ET.SubElement(geometry, "cylinder", radius=f"{size[0]:.12g}", length=f"{2.0 * size[1]:.12g}")
    elif geometry_type == "box":
        if len(size) != 3:
            raise ValueError(f"box requires three half-extents: {ET.tostring(geom, encoding='unicode')}")
        ET.SubElement(geometry, "box", size=format_floats([2.0 * value for value in size]))


def is_visual_only(geom: ET.Element) -> bool:
    return geom.get("contype") == "0" and geom.get("conaffinity") == "0"


def add_link(robot: ET.Element, body: ET.Element, mesh_files: dict[str, str], material_counter: list[int]) -> None:
    body_name = body.get("name")
    if not body_name:
        raise ValueError("every active body must be named")
    link = ET.SubElement(robot, "link", name=body_name)

    inertial = body.find("inertial")
    if inertial is not None:
        urdf_inertial = ET.SubElement(link, "inertial")
        inertial_xyz = parse_floats(inertial.get("pos"), "0 0 0")
        ET.SubElement(urdf_inertial, "origin", xyz=format_floats(inertial_xyz), rpy="0 0 0")
        ET.SubElement(urdf_inertial, "mass", value=f"{float(inertial.get('mass')):.12g}")
        diagonal = parse_floats(inertial.get("diaginertia"), "")
        if len(diagonal) != 3:
            raise ValueError(f"body {body_name} does not have diagonal principal inertia")
        tensor = rotate_diagonal_inertia(
            diagonal,
            parse_floats(inertial.get("quat"), "1 0 0 0"),
        )
        ET.SubElement(
            urdf_inertial,
            "inertia",
            ixx=f"{tensor[0][0]:.12g}",
            ixy=f"{tensor[0][1]:.12g}",
            ixz=f"{tensor[0][2]:.12g}",
            iyy=f"{tensor[1][1]:.12g}",
            iyz=f"{tensor[1][2]:.12g}",
            izz=f"{tensor[2][2]:.12g}",
        )

    for geom in body.findall("geom"):
        visual_only = is_visual_only(geom)
        tag = "visual" if visual_only else "collision"
        urdf_geom = ET.SubElement(link, tag)
        add_origin(urdf_geom, geom)
        add_geometry(urdf_geom, geom, mesh_files)
        if visual_only:
            rgba = parse_floats(geom.get("rgba"), "0.7 0.7 0.7 1")
            material_counter[0] += 1
            material = ET.SubElement(urdf_geom, "material", name=f"g1_material_{material_counter[0]}")
            ET.SubElement(material, "color", rgba=format_floats(rgba))


def add_joint(
    robot: ET.Element,
    parent_name: str,
    body: ET.Element,
    default_damping: float,
    default_friction: float,
) -> None:
    body_name = body.get("name")
    source_joints = body.findall("joint")
    if len(source_joints) > 1:
        raise ValueError(f"body {body_name} has multiple joints, which cannot map directly to URDF")

    if source_joints:
        source_joint = source_joints[0]
        joint_type = source_joint.get("type", "hinge")
        if joint_type == "free":
            urdf_type = "floating"
        elif joint_type == "hinge":
            urdf_type = "revolute"
        else:
            raise ValueError(f"unsupported joint type {joint_type} on {source_joint.get('name')}")
        joint_name = source_joint.get("name")
    else:
        source_joint = None
        urdf_type = "fixed"
        joint_name = f"{body_name}_fixed_joint"

    joint = ET.SubElement(robot, "joint", name=joint_name, type=urdf_type)
    add_origin(joint, body)
    ET.SubElement(joint, "parent", link=parent_name)
    ET.SubElement(joint, "child", link=body_name)

    if source_joint is None:
        return

    if urdf_type == "floating":
        ET.SubElement(
            joint,
            "dynamics",
            damping=f"{float(source_joint.get('damping', default_damping)):.12g}",
            friction=f"{float(source_joint.get('frictionloss', default_friction)):.12g}",
        )
        return

    joint_position = parse_floats(source_joint.get("pos"), "0 0 0")
    if any(abs(value) > 1e-12 for value in joint_position):
        raise ValueError(f"joint {joint_name} has a nonzero local position and needs frame splitting")
    ET.SubElement(joint, "axis", xyz=format_floats(parse_floats(source_joint.get("axis"), "0 0 1")))
    joint_range = parse_floats(source_joint.get("range"), "")
    effort_range = parse_floats(source_joint.get("actuatorfrcrange"), "")
    if len(joint_range) != 2 or len(effort_range) != 2:
        raise ValueError(f"joint {joint_name} lacks range or actuator force range")
    ET.SubElement(
        joint,
        "limit",
        lower=f"{joint_range[0]:.12g}",
        upper=f"{joint_range[1]:.12g}",
        effort=f"{max(abs(value) for value in effort_range):.12g}",
        velocity="100",
    )
    ET.SubElement(
        joint,
        "dynamics",
        damping=f"{float(source_joint.get('damping', default_damping)):.12g}",
        friction=f"{float(source_joint.get('frictionloss', default_friction)):.12g}",
    )


def convert(source: Path, output: Path) -> None:
    source_root = ET.parse(source).getroot()
    if source_root.tag != "mujoco":
        raise ValueError(f"expected a MuJoCo model, got {source_root.tag}")
    worldbody = source_root.find("worldbody")
    if worldbody is None:
        raise ValueError("source MJCF has no worldbody")
    root_bodies = worldbody.findall("body")
    if len(root_bodies) != 1 or root_bodies[0].get("name") != "pelvis":
        raise ValueError("expected pelvis to be the only source root body")

    mesh_files: dict[str, str] = {}
    for mesh in source_root.findall("./asset/mesh"):
        name = mesh.get("name")
        filename = mesh.get("file")
        if not name or not filename:
            raise ValueError("all meshes must have names and files")
        mesh_files[name] = Path(filename).name

    source_bodies = worldbody.findall(".//body")
    source_joints = worldbody.findall(".//joint")
    source_inertials = worldbody.findall(".//inertial")
    hinge_joints = [joint for joint in source_joints if joint.get("type", "hinge") == "hinge"]
    if (len(source_bodies), len(source_joints), len(source_inertials), len(hinge_joints)) != (37, 30, 30, 29):
        raise ValueError("source G1 structure changed; review conversion assumptions")

    default_joint = source_root.find("./default/joint")
    if default_joint is None:
        raise ValueError("source MJCF has no default joint dynamics")
    default_damping = float(default_joint.get("damping", "0"))
    default_friction = float(default_joint.get("frictionloss", "0"))

    robot = ET.Element("robot", name=source_root.get("model", "g1_29dof_rev_1_0"))
    mujoco_extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        mujoco_extension,
        "compiler",
        meshdir="meshes",
        discardvisual="false",
        fusestatic="false",
    )
    robot.append(ET.Comment("Source MJCF has no velocity limits; 100 rad/s is intentionally non-binding."))
    robot.append(ET.Comment("MJCF motor actuators and joint armature must be configured by the training environment."))
    ET.SubElement(robot, "link", name="world")

    material_counter = [0]

    def emit_body(body: ET.Element, parent_name: str) -> None:
        add_link(robot, body, mesh_files, material_counter)
        add_joint(robot, parent_name, body, default_damping, default_friction)
        for child in body.findall("body"):
            emit_body(child, body.get("name"))

    emit_body(root_bodies[0], "world")
    ET.indent(robot, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    with output.open("a", encoding="utf-8") as file:
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the repository G1 MJCF to a MuJoCo-oriented URDF.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.source.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
