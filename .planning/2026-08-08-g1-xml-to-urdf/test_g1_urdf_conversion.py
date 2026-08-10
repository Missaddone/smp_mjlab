from __future__ import annotations

import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


TEST_ROOT = Path(__file__).resolve().parents[3]
ROBOT_REPO = TEST_ROOT / "robot_retargeter"
SOURCE = ROBOT_REPO / "asset/robot/g1_description/mjcf/g1.xml"
OUTPUT = ROBOT_REPO / "asset/robot/g1_description/g1.urdf"


def _floats(value: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.fromstring(value if value is not None else default, sep=" ")


def _body_inertia_tensor(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T


class G1UrdfConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not OUTPUT.exists():
            raise AssertionError(f"generated URDF is missing: {OUTPUT}")
        cls.source_tree = ET.parse(SOURCE)
        cls.urdf_tree = ET.parse(OUTPUT)
        cls.source_root = cls.source_tree.getroot()
        cls.urdf_root = cls.urdf_tree.getroot()

    def test_graph_and_joint_limits_match_source(self) -> None:
        source_bodies = self.source_root.findall(".//worldbody//body")
        source_hinges = [
            joint
            for joint in self.source_root.findall(".//worldbody//joint")
            if joint.get("type", "hinge") != "free"
        ]
        links = self.urdf_root.findall("link")
        joints = self.urdf_root.findall("joint")
        revolute = [joint for joint in joints if joint.get("type") == "revolute"]
        fixed = [joint for joint in joints if joint.get("type") == "fixed"]
        floating = [joint for joint in joints if joint.get("type") == "floating"]

        self.assertEqual(len(source_bodies), 37)
        self.assertEqual(len(links), 38)
        self.assertEqual(len(joints), 37)
        self.assertEqual(len(revolute), 29)
        self.assertEqual(len(fixed), 7)
        self.assertEqual([joint.get("name") for joint in floating], ["floating_base_joint"])
        floating_dynamics = floating[0].find("dynamics")
        self.assertIsNotNone(floating_dynamics)
        self.assertEqual(float(floating_dynamics.get("damping")), 0.001)
        self.assertEqual(float(floating_dynamics.get("friction")), 0.1)
        self.assertEqual({joint.get("name") for joint in source_hinges}, {joint.get("name") for joint in revolute})

        source_by_name = {joint.get("name"): joint for joint in source_hinges}
        for joint in revolute:
            source_joint = source_by_name[joint.get("name")]
            limit = joint.find("limit")
            dynamics = joint.find("dynamics")
            self.assertIsNotNone(limit)
            self.assertIsNotNone(dynamics)
            np.testing.assert_allclose(
                [float(limit.get("lower")), float(limit.get("upper"))],
                _floats(source_joint.get("range")),
                rtol=0,
                atol=1e-10,
            )
            effort_range = _floats(source_joint.get("actuatorfrcrange"), "0 0")
            self.assertAlmostEqual(float(limit.get("effort")), float(np.max(np.abs(effort_range))))
            self.assertEqual(float(limit.get("velocity")), 100.0)
            self.assertEqual(float(dynamics.get("damping")), 0.001)
            self.assertEqual(float(dynamics.get("friction")), 0.1)

    def test_mesh_references_resolve_once(self) -> None:
        compiler = self.urdf_root.find("./mujoco/compiler")
        self.assertIsNotNone(compiler)
        self.assertEqual(compiler.get("meshdir"), "meshes")
        self.assertEqual(compiler.get("fusestatic"), "false")
        mesh_dir = (OUTPUT.parent / compiler.get("meshdir")).resolve()
        meshes = self.urdf_root.findall(".//mesh")
        self.assertGreater(len(meshes), 0)
        for mesh in meshes:
            filename = mesh.get("filename")
            self.assertEqual(Path(filename).name, filename)
            self.assertTrue((mesh_dir / filename).is_file(), filename)

    def test_mujoco_compiled_models_match(self) -> None:
        source_model = mujoco.MjModel.from_xml_path(str(SOURCE))
        urdf_model = mujoco.MjModel.from_xml_path(str(OUTPUT))

        self.assertEqual((source_model.nbody, source_model.njnt, source_model.nq, source_model.nv), (38, 30, 36, 35))
        self.assertEqual(
            (urdf_model.nbody, urdf_model.njnt, urdf_model.nq, urdf_model.nv),
            (source_model.nbody, source_model.njnt, source_model.nq, source_model.nv),
        )

        source_joint_names = [source_model.joint(index).name for index in range(source_model.njnt)]
        urdf_joint_names = [urdf_model.joint(index).name for index in range(urdf_model.njnt)]
        self.assertEqual(urdf_joint_names, source_joint_names)
        np.testing.assert_allclose(urdf_model.jnt_range, source_model.jnt_range, rtol=0, atol=1e-9)
        np.testing.assert_allclose(urdf_model.dof_damping, source_model.dof_damping, rtol=0, atol=1e-12)
        np.testing.assert_allclose(urdf_model.dof_frictionloss, source_model.dof_frictionloss, rtol=0, atol=1e-12)
        self.assertEqual((urdf_model.ngeom, urdf_model.nmesh), (source_model.ngeom, source_model.nmesh))
        np.testing.assert_array_equal(urdf_model.geom_type, source_model.geom_type)
        np.testing.assert_array_equal(urdf_model.geom_bodyid, source_model.geom_bodyid)
        np.testing.assert_array_equal(urdf_model.geom_contype, source_model.geom_contype)
        np.testing.assert_array_equal(urdf_model.geom_conaffinity, source_model.geom_conaffinity)
        np.testing.assert_allclose(urdf_model.geom_pos, source_model.geom_pos, rtol=0, atol=1e-10)
        np.testing.assert_allclose(urdf_model.geom_size, source_model.geom_size, rtol=0, atol=1e-10)
        for geom_id in range(source_model.ngeom):
            quat_dot = abs(float(np.dot(urdf_model.geom_quat[geom_id], source_model.geom_quat[geom_id])))
            self.assertTrue(math.isclose(quat_dot, 1.0, abs_tol=2e-10), f"geom {geom_id}")

        source_body_names = [source_model.body(index).name for index in range(1, source_model.nbody)]
        urdf_body_names = [urdf_model.body(index).name for index in range(1, urdf_model.nbody)]
        self.assertEqual(urdf_body_names, source_body_names)
        np.testing.assert_allclose(urdf_model.body_mass, source_model.body_mass, rtol=0, atol=1e-9)
        np.testing.assert_allclose(urdf_model.body_ipos, source_model.body_ipos, rtol=0, atol=1e-9)
        for body_name in source_body_names:
            source_id = source_model.body(body_name).id
            urdf_id = urdf_model.body(body_name).id
            np.testing.assert_allclose(
                _body_inertia_tensor(urdf_model, urdf_id),
                _body_inertia_tensor(source_model, source_id),
                rtol=0,
                # MuJoCo diagonalizes a full URDF tensor on import. The torso's
                # near-adjacent principal values produce ~3.8e-8 eigensolver noise.
                atol=5e-8,
                err_msg=body_name,
            )

        source_data = mujoco.MjData(source_model)
        urdf_data = mujoco.MjData(urdf_model)
        mujoco.mj_forward(source_model, source_data)
        mujoco.mj_forward(urdf_model, urdf_data)
        for body_name in source_body_names:
            source_id = source_model.body(body_name).id
            urdf_id = urdf_model.body(body_name).id
            np.testing.assert_allclose(urdf_data.xpos[urdf_id], source_data.xpos[source_id], rtol=0, atol=2e-8)
            quat_dot = abs(float(np.dot(urdf_data.xquat[urdf_id], source_data.xquat[source_id])))
            self.assertTrue(math.isclose(quat_dot, 1.0, abs_tol=2e-8), body_name)

        # Core URDF has no MuJoCo motor or armature representation. Keep this
        # limitation explicit so training configurations restore both values.
        self.assertEqual((source_model.nu, urdf_model.nu), (29, 0))
        np.testing.assert_allclose(source_model.dof_armature, 0.01, rtol=0, atol=1e-12)
        np.testing.assert_allclose(urdf_model.dof_armature, 0.0, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
