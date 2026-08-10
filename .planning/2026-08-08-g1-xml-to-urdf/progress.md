# Progress: G1 MJCF/XML to URDF

## 2026-08-08
- Accepted the revised task: convert the G1 XML/MJCF description into URDF.
- Read the required workflow skills.
- Confirmed the expected directory is not directly under the repository root.
- Created isolated planning files to avoid changing unrelated root-level planning state.
- Located and began inspecting the installed `mjlab` Unitree G1 MJCF and its STL assets.
- Counted the complete G1 kinematic/inertial model and identified MJCF-only semantics that require explicit URDF conversion choices.
- Corrected the target to sibling `../robot_retargeter` and found its actual G1 MJCF path.
- Verified the target repository is clean and compared its G1 MJCF with an existing paired URDF convention.
- Audited all source bodies/joints/actuators and found a doubled mesh-path problem in existing reference URDFs that the G1 output must avoid.
- Verified MuJoCo's actual URDF floating-base behavior with minimal in-memory models.
- Began checking official Unitree descriptions for velocity limits; the first web search failed at the tool layer with HTTP 404.
- User approved the MuJoCo-oriented Scheme A without further design review.
- Wrote the design specification, implementation plan, and acceptance test.
- Confirmed the acceptance test fails only because the target URDF is absent.
- Confirmed `robot_retargeter` is a clean normal checkout on `main`; user explicitly requested the asset be generated in that existing directory, so work proceeds in place without a separate worktree.
- Generated the first URDF and passed graph/limit/mesh tests; compiled-model parity exposed MuJoCo fixed-link fusion for the seven marker bodies.
- Setting `fusestatic=false` preserved all marker bodies; the next compiled comparison exposed missing floating-base damping/friction in the URDF.
- Floating-base dynamics now match. A stricter physical-inertia audit found that diagonal inertia plus inertial RPY does not survive MuJoCo URDF import correctly, so a full-tensor representation is required.
- Full link-frame inertia tensors now agree within `3.73e-8 kg·m²`; the remaining difference is MuJoCo eigendecomposition noise, so the acceptance tolerance is `5e-8`.
- Final acceptance tests pass (3/3). Direct MuJoCo load reports `nbody=38`, `njnt=30`, `nq=36`, `nv=35`, `ngeom=84`, and `nmesh=35`.
- Confirmed only the requested URDF directory was added to `robot_retargeter`; left it uncommitted.
- User requested an hhtools-compatible layout matching R1: move `g1.urdf` directly under `g1_description/`, change `meshdir` to `meshes`, and delete the old `urdf/` directory.
- Updated the acceptance test first; it fails for the expected reason that the new root-level URDF has not yet been generated.
- Updated the generator to emit `g1_description/g1.urdf` with `meshdir="meshes"`, regenerated the model, and removed the superseded empty `g1_description/urdf/` directory.
- Re-ran the complete acceptance suite after relocation: all 3 tests passed, including mesh-path resolution and compiled-model parity.
- Independently compiled the final root-level URDF with MuJoCo: `nbody=38`, `njnt=30`, `nq=36`, `nv=35`, `nmesh=35`, `ngeom=84`.
- Confirmed the final layout contains `g1.urdf`, `meshes/`, and `mjcf/` directly under `g1_description/`, with no remaining `urdf/` directory.
