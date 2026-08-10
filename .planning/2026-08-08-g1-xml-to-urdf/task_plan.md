# G1 MJCF/XML to URDF

## Goal
Locate the G1 model XML, design a faithful URDF conversion, generate the URDF after user approval, and verify structure, paths, joints, inertials, and geometry references.

## Phases
- [x] Phase 1: Locate and inspect G1 model assets and repository conventions.
- [x] Phase 2: Clarify target compatibility and present conversion approaches/design for approval.
- [x] Phase 3: Write the approved design and implementation plan.
- [x] Phase 4: Generate the URDF and any narrowly required conversion tooling.
- [x] Phase 5: Validate the output and document known MJCF-to-URDF semantic differences.
- [x] Phase 6: Relocate the URDF beside `meshes/` for hhtools-compatible ingestion and remove the old `urdf/` directory.

## Current Phase
Complete.

## Next Step
None. All requested conversion and relocation work is complete.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| `g1_description` not found at repository root | 1 | Search the full repository for matching paths and model XML files. |
| Target repository initially interpreted as the current repository | 1 | User clarified it is sibling `../robot_retargeter`; all further source inspection uses that repository. |
| Loading existing R1 URDF with MuJoCo failed: `meshes/meshes/left_knee_link.STL` | 1 | Existing files combine `compiler meshdir="meshes"` with `filename="meshes/..."`; do not copy this doubled-path convention into G1. Validate the generated G1 URDF directly with MuJoCo. |
| Official Unitree web search returned HTTP 404 from the search tool | 1 | Retry with a narrower official GitHub query; if unavailable, avoid inventing source-absent velocity data and document the fallback. |
| Generated URDF compiled to `nbody=31` instead of source `nbody=38` | 1 | MuJoCo fused seven fixed marker links; explicitly set `fusestatic="false"` and regenerate before deciding whether this is an unavoidable URDF-import boundary. |
| Inertial principal values matched but link-frame inertia tensors differed by up to `0.00374 kg·m²` | 1 | MuJoCo's URDF import did not preserve inertial-origin RPY as expected; add a failing tensor-parity test, then emit the full rotated inertia tensor with zero inertial RPY. |
