# G1 MuJoCo-Oriented URDF Design

## Scope

Convert `../robot_retargeter/asset/robot/g1_description/mjcf/g1.xml` into
`../robot_retargeter/asset/robot/g1_description/g1.urdf`. Do not change the
retargeting configuration or the source MJCF.

## Model structure

- Add a synthetic massless `world` link and a `floating_base_joint` to `pelvis`.
- Preserve all 29 actuated hinge joints as URDF `revolute` joints.
- Preserve the seven source marker bodies as massless fixed links so frame names remain available.
- Preserve every body pose and joint axis in the source local frame.

## Dynamics

- Preserve link mass and center of mass. Rotate each MJCF principal-axis diagonal inertia into the link frame and write the full URDF inertia tensor with zero inertial RPY; this avoids MuJoCo URDF import losing inertial-frame rotation.
- Preserve hinge lower/upper limits, damping, and friction loss.
- Map `actuatorfrcrange` to the URDF effort limit.
- Use a documented non-binding `100 rad/s` velocity limit because the source MJCF contains no velocity limits; training actuator configuration must provide the real limit.
- MJCF armature and motor declarations have no portable core-URDF equivalent and are not fabricated in the URDF.

## Geometry

- Emit source geoms with `contype=0` and `conaffinity=0` as visuals.
- Emit contact-enabled source geoms as collisions.
- Preserve mesh, sphere, cylinder, and box poses and dimensions.
- Place the URDF directly under `g1_description/`, set MuJoCo `meshdir="meshes"`, and use bare STL filenames so upload tools preserve the same URDF/mesh relationship.

## Verification

- XML parse and graph checks: 38 links total, 37 URDF joints, 29 revolute joints, one floating base, and seven fixed marker joints.
- All mesh references resolve.
- MuJoCo loads the source MJCF and generated URDF.
- Both compiled models have `nbody=38`, `njnt=30`, `nq=36`, and `nv=35`.
- Joint names, ranges, damping, friction loss, body masses, inertial positions, and link-frame inertia tensors match within numerical tolerance.

## Known semantic boundary

URDF cannot encode MJCF actuators, armature, sensors, contact exclusions, or every MuJoCo-specific compiler option. This source has motors and armature but no active sensors/contact exclusions. Training code must add actuators after loading the URDF.
