# G1 MJCF-to-URDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a MuJoCo-loadable floating-base G1 URDF that faithfully preserves the source MJCF kinematics, inertials, limits, and geometry classification.

**Architecture:** A task-local Python converter parses the fixed source MJCF with `xml.etree.ElementTree`, recursively emits URDF links/joints, and converts MuJoCo WXYZ quaternions to URDF RPY. A task-local `unittest` validator compares both raw XML structure and compiled MuJoCo models.

**Tech Stack:** Python 3.13 standard library, MuJoCo Python bindings, `unittest`.

---

### Task 1: Define conversion acceptance tests

**Files:**
- Create: `.planning/2026-08-08-g1-xml-to-urdf/test_g1_urdf_conversion.py`

- [x] Write tests for output existence, URDF graph counts, mesh resolution, source joint limits/dynamics, and compiled MuJoCo model parity.
- [x] Run `.venv/bin/python .planning/2026-08-08-g1-xml-to-urdf/test_g1_urdf_conversion.py`.
- [x] Confirm failure is caused by the absent `g1.urdf` output.

### Task 2: Implement deterministic conversion

**Files:**
- Create: `.planning/2026-08-08-g1-xml-to-urdf/generate_g1_urdf.py`
- Create: `../robot_retargeter/asset/robot/g1_description/g1.urdf`

- [x] Parse the source asset map and body tree.
- [x] Emit `world`, the floating pelvis joint, dynamic links, fixed marker links, inertials, visuals, collisions, limits, and dynamics.
- [x] Assert the source shape: 29 hinge joints, 30 inertials, joint positions at body origins, and only supported geometry types.
- [x] Generate formatted XML without changing the source MJCF.

### Task 3: Verify model parity

**Files:**
- Validate: `../robot_retargeter/asset/robot/g1_description/g1.urdf`

- [x] Run the validator and require zero failures.
- [x] Load the generated URDF directly with `mujoco.MjModel.from_xml_path`.
- [x] Compare compiled dimensions, joint properties, inertials, geometry, and zero-pose world transforms against the source MJCF.
- [x] Run `git -C ../robot_retargeter diff --check` and inspect `git status --short` to confirm only the requested URDF path was added.
