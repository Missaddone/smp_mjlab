"""G1 goalkeeper task with SMP guidance.

The robot faces incoming shots, with the goal behind it.  Every five seconds a
dynamic football is fired from randomized shooter positions toward the goal.  The policy
observes the facing direction plus the ball's relative position and linear
velocity in the robot heading frame.
"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import root_height_below_minimum
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.catchball import mdp


def get_football_spec(
  radius: float = 0.11,
  mass: float = 0.43,
  rgba: tuple[float, float, float, float] = (0.95, 0.72, 0.18, 1.0),
) -> mujoco.MjSpec:
  """Create a free-floating sphere that collides with terrain and the robot."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="football")
  body.add_freejoint(name="football_joint")
  geom = body.add_geom(
    name="football_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius,),
    mass=mass,
    rgba=rgba,
  )
  geom.friction = (0.8, 0.02, 0.001)
  geom.condim = 3
  return spec


def get_goal_spec(
  goal_width: float = 2.4,
  goal_height: float = 1.6,
  post_radius: float = 0.04,
  rgba: tuple[float, float, float, float] = (0.95, 0.95, 0.95, 1.0),
) -> mujoco.MjSpec:
  """Create a simple fixed goal frame."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal_frame")
  left_y = -goal_width * 0.5
  right_y = goal_width * 0.5
  crossbar_z = goal_height

  body.add_geom(
    name="goal_left_post",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.0, left_y, 0.0, 0.0, left_y, crossbar_z),
    size=(post_radius,),
    rgba=rgba,
    contype=0,
    conaffinity=0,
  )
  body.add_geom(
    name="goal_right_post",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.0, right_y, 0.0, 0.0, right_y, crossbar_z),
    size=(post_radius,),
    rgba=rgba,
    contype=0,
    conaffinity=0,
  )
  body.add_geom(
    name="goal_crossbar",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(0.0, left_y, crossbar_z, 0.0, right_y, crossbar_z),
    size=(post_radius,),
    rgba=rgba,
    contype=0,
    conaffinity=0,
  )
  body.add_geom(
    name="goal_line_vis",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.01),
    size=(0.01, goal_width * 0.5, 0.005),
    rgba=(0.2, 0.9, 0.2, 0.6),
    contype=0,
    conaffinity=0,
  )
  return spec


def g1_catchball_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 catch-ball env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Scene ---------------------------------------------------------------
  cfg.scene.entities["football"] = EntityCfg(spec_fn=get_football_spec)
  cfg.scene.entities["goal"] = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    spec_fn=lambda: get_goal_spec(goal_width=2.4, goal_height=1.6),
  )
  football_contact_cfg = ContactSensorCfg(
    name="football_contact",
    primary=ContactMatch(mode="geom", pattern="football_geom", entity="football"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=16,
    secondary_policy="any",
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (football_contact_cfg,)
  cfg.sim.nconmax = max(cfg.sim.nconmax or 35, 80)
  cfg.sim.njmax = max(cfg.sim.njmax or 1500, 1600)

  # --- Commands ------------------------------------------------------------
  cfg.commands["catchball"] = mdp.CatchBallCommandCfg(
    entity_name="robot",
    ball_entity_name="football",
    contact_sensor_name="football_contact",
    resampling_time_range=(5.0, 5.0),
    align_to_robot_heading=True,
    goal_direction=(-1.0, 0.0),
    spawn_distance_min=3.0,
    spawn_distance_max=5.0,
    spawn_height_min=0.7,
    spawn_height_max=1.4,
    shooter_lateral_range=1.2,
    launch_angle_jitter=0.15,
    goal_line_distance=0.8,
    goal_width=2.4,
    goal_height=1.6,
    ball_radius=0.11,
    ball_speed_min=4.0,
    ball_speed_max=8.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "catchball"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  # task = sparse block reward + face shots - position/concede penalties.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.goalkeeper_block_shot,
          3.0,
          {
            "command_name": "catchball",
            "catch_distance": 0.22,
          },
        ),
        (
          mdp.goalkeeper_position_penalty,
          -1.0,
          {
            "command_name": "catchball",
            "distance_from_goal": 0.5,
            "forward_scale": 6.0,
            "lateral_scale": 1.0,
          },
        ),
        (mdp.catch_ball_face_direction, 0.15, {"command_name": "catchball"}),
        (mdp.goalkeeper_goal_allowed_penalty, -2.0, {"command_name": "catchball"}),
      ),
      "ws": 4,
    },
  )

  # --- Metrics -------------------------------------------------------------
  # Mirror the reward sub-terms inside ``task_smp_product`` for TensorBoard.
  cfg.metrics["goalkeeper_block_shot"] = MetricsTermCfg(
    func=mdp.goalkeeper_block_shot,
    params={
      "command_name": "catchball",
      "catch_distance": 0.22,
    },
  )
  cfg.metrics["goalkeeper_position_penalty"] = MetricsTermCfg(
    func=mdp.goalkeeper_position_penalty,
    params={
      "command_name": "catchball",
      "distance_from_goal": 0.5,
      "forward_scale": 6.0,
      "lateral_scale": 1.0,
    },
  )
  cfg.metrics["goalkeeper_face_direction"] = MetricsTermCfg(
    func=mdp.catch_ball_face_direction,
    params={"command_name": "catchball"},
  )
  cfg.metrics["goalkeeper_goal_allowed_penalty"] = MetricsTermCfg(
    func=mdp.goalkeeper_goal_allowed_penalty,
    params={"command_name": "catchball"},
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_lafan_run.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  cfg.episode_length_s = 20.0

  return cfg
