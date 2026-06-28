"""G1 upstairs task -- fixed +x stair climbing with SMP guidance.

This uses a rough-G1-style twist command on an inverted open-stairs terrain.  The
robot spawns on the low center platform; walking forward in the body frame moves
it outward and upward over the stair rings.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg, mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
  BoxOpenStairsTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import smp_reward_component_log, task_smp_product
from smp.rl.tasks.steering import mdp


def _add_smp_reward_component_logs(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.rewards["task_reward"] = RewardTermCfg(
    func=smp_reward_component_log,
    weight=1.0,
    params={"component": "task_reward", "log_name": "task_reward"},
  )
  cfg.rewards["style_reward"] = RewardTermCfg(
    func=smp_reward_component_log,
    weight=1.0,
    params={"component": "style_reward", "log_name": "style_reward"},
  )


def g1_upstairs_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 upstairs env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Terrain -------------------------------------------------------------
  cfg.scene.extent = 12.0
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=17,
      curriculum=True,
      size=(12.0, 12.0),
      border_width=0.5,
      num_rows=8,
      num_cols=1,
      difficulty_range=(0.0, 1.0),
      sub_terrains={
        "open_upstairs": BoxOpenStairsTerrainCfg(
          proportion=1.0,
          step_height_range=(0.04, 0.18),
          step_width_range=(0.45, 0.80),
          platform_width=1.8,
          border_width=0.3,
          step_thickness=0.08,
          inverted=True,
        ),
      },
    ),
    env_spacing=None,
    max_init_terrain_level=2,
    debug_vis=play,
  )

  foot_site_names = ("left_foot", "right_foot")
  feet_ground_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    RayCastSensorCfg(
      name="terrain_scan",
      frame=ObjRef(type="body", name="pelvis", entity="robot"),
      ray_alignment="yaw",
      pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
      max_distance=5.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=play,
    ),
    TerrainHeightSensorCfg(
      name="foot_height_scan",
      frame=tuple(
        ObjRef(type="site", name=site_name, entity="robot")
        for site_name in foot_site_names
      ),
      ray_alignment="yaw",
      pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
      max_distance=1.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=play,
    ),
    feet_ground_contact,
  )

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # --- Commands ------------------------------------------------------------
  cfg.commands["twist"] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 8.0),
    heading_command=False,
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.4, 1.0),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-1.0, 1.0),
      heading=None,
    ),
    viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15),
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "twist"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs
  cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    scale=0.2,
  )
  cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    scale=0.2,
  )
  cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
    func=velocity_mdp.foot_height,
    params={"sensor_name": "foot_height_scan"},
  )
  cfg.observations["critic"].terms["foot_air_time"] = ObservationTermCfg(
    func=velocity_mdp.foot_air_time,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.observations["critic"].terms["foot_contact"] = ObservationTermCfg(
    func=velocity_mdp.foot_contact,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.observations["critic"].terms["foot_contact_forces"] = ObservationTermCfg(
    func=velocity_mdp.foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
  )

  # --- Rewards -------------------------------------------------------------
  # task = rough-G1 style twist tracking on perceived terrain,
  # plus a small climb-progress term, gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          velocity_mdp.track_linear_velocity,
          0.75,
          {"command_name": "twist", "std": 0.5},
        ),
        (
          velocity_mdp.track_angular_velocity,
          0.10,
          {"command_name": "twist", "std": 0.5},
        ),
        (
          mdp.upstairs_height_progress,
          0.15,
          {"target_climb_height": 0.9, "nominal_root_height": 0.75},
        ),
      ),
    },
  )
  _add_smp_reward_component_logs(cfg)
  cfg.rewards["feet_air_time"] = RewardTermCfg(
    func=velocity_mdp.feet_air_time,
    weight=0.20,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.08,
      "threshold_max": 0.55,
      "command_name": "twist",
      "command_threshold": 0.3,
    },
  )
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=velocity_mdp.feet_clearance,
    weight=-1.0,
    params={
      "target_height": 0.16,
      "height_sensor_name": "foot_height_scan",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
    },
  )
  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=velocity_mdp.feet_swing_height,
    weight=-0.15,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.16,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=velocity_mdp.feet_slip,
    weight=-0.10,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
    },
  )
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=velocity_mdp.soft_landing,
    weight=-1e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["upright"] = RewardTermCfg(
    func=velocity_mdp.upright,
    weight=0.25,
    params={
      "std": 0.45,
      "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    },
  )
  cfg.rewards["ankle_pos_limits"] = RewardTermCfg(
    func=velocity_mdp.joint_pos_limits,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(".*ankle_pitch.*", ".*ankle_roll.*"),
      ),
    },
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=velocity_mdp.action_rate_l2,
    weight=-0.005,
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_loco.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_origin_offset,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
    func=mdp.out_of_terrain_bounds,
    time_out=True,
    params={"margin": 0.3, "asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.terminations["terrain_edge_reached"] = TerminationTermCfg(
    func=mdp.terrain_edge_reached,
    time_out=True,
    params={
      "threshold_fraction": 0.45,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # --- Curriculum ----------------------------------------------------------
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.upstairs_terrain_levels,
    params={
      "command_name": "twist",
      "success_fraction": 0.45,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  if play:
    cfg.curriculum = {}
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
