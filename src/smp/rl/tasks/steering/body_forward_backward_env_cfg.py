"""Body-frame forward/backward velocity task with SMP guidance."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import smp_reward_component_log
from smp.rl.tasks.steering import mdp

PRETRAIN_CKPT_DIR = "datasets/pretrain_ckpt"
FOOT_SITE_NAMES = ("left_foot", "right_foot")
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


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


def _body_x_command(
  lin_vel_x: tuple[float, float],
) -> mdp.BodyVelocityCommandCfg:
  return mdp.BodyVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    lin_vel_x_min=lin_vel_x[0],
    lin_vel_x_max=lin_vel_x[1],
    lin_vel_y_min=0.0,
    lin_vel_y_max=0.0,
    yaw_rate_min=0.0,
    yaw_rate_max=0.0,
  )


def _set_command_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "steering"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs
  cfg.observations["actor"].terms.pop("base_lin_vel", None)


def _add_foot_sensors_and_critic_obs(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
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
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(
      ObjRef(type="site", name=site_name, entity="robot")
      for site_name in FOOT_SITE_NAMES
    ),
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=play,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_contact,
    foot_height_scan,
  )
  cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
    func=velocity_mdp.foot_height,
    params={"sensor_name": "foot_height_scan"},
  )
  cfg.observations["critic"].terms["foot_contact"] = ObservationTermCfg(
    func=velocity_mdp.foot_contact,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.observations["critic"].terms["foot_contact_forces"] = ObservationTermCfg(
    func=velocity_mdp.foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
  )


def _posture_std_slow() -> dict[str, float]:
  return {
    r".*hip_pitch.*": 0.65,
    r".*knee.*": 0.65,
    r".*ankle_pitch.*": 0.30,
    r".*hip_roll.*": 0.18,
    r".*hip_yaw.*": 0.18,
    r".*ankle_roll.*": 0.12,
    r".*waist_yaw.*": 0.18,
    r".*waist_roll.*": 0.12,
    r".*waist_pitch.*": 0.12,
    r".*shoulder_pitch.*": 0.30,
    r".*shoulder_roll.*": 0.20,
    r".*shoulder_yaw.*": 0.20,
    r".*elbow.*": 0.25,
    r".*wrist_roll.*": 0.30,
    r".*wrist_pitch.*": 0.30,
    r".*wrist_yaw.*": 0.30,
  }


def _posture_std_fast() -> dict[str, float]:
  return {
    r".*hip_pitch.*": 0.80,
    r".*knee.*": 0.80,
    r".*ankle_pitch.*": 0.40,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*ankle_roll.*": 0.14,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.14,
    r".*waist_pitch.*": 0.14,
    r".*shoulder_pitch.*": 0.40,
    r".*shoulder_roll.*": 0.25,
    r".*shoulder_yaw.*": 0.25,
    r".*elbow.*": 0.35,
    r".*wrist_roll.*": 0.35,
    r".*wrist_pitch.*": 0.35,
    r".*wrist_yaw.*": 0.35,
  }


def g1_body_forward_backward_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the body-frame x velocity task.

  The command is ``[v_x, 0, 0]`` in the robot heading frame, so positive values
  ask the policy to walk forward and negative values ask it to step backward.
  """
  cfg = g1_smp_env_cfg(play=play)

  initial_range = (-0.5, 1.0)
  final_range = (-1.0, 3.0)
  cfg.commands["steering"] = _body_x_command(final_range if play else initial_range)
  cfg.commands["steering"].debug_vis = play

  _set_command_obs(cfg)
  _add_foot_sensors_and_critic_obs(cfg, play=play)

  cfg.events["init_smp_state"].params["ckpt_path"] = (
    f"{PRETRAIN_CKPT_DIR}/amp_loco_clips2_mirrored_lafan_norm_12000.pt"
  )

  cfg.rewards.clear()
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=mdp.body_velocity_task_smp_product,
    weight=1.0,
    params={
      "command_name": "steering",
      "lin_vel_err_scale": 1.5,
      "yaw_rate_err_scale": 1.0,
      "lin_vel_weight": 0.85,
      "yaw_rate_weight": 0.15,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )
  _add_smp_reward_component_logs(cfg)
  cfg.rewards["posture"] = RewardTermCfg(
    func=mdp.speed_dependent_joint_posture_reward,
    weight=0.15,
    params={
      "command_name": "steering",
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      "std_slow": _posture_std_slow(),
      "std_fast": _posture_std_fast(),
      "fast_threshold": 1.5,
    },
  )
  cfg.rewards["feet_slip"] = RewardTermCfg(
    func=velocity_mdp.feet_slip,
    weight=-0.10,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "steering",
      "command_threshold": 0.15,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  )
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=velocity_mdp.feet_clearance,
    weight=-0.25,
    params={
      "target_height": 0.10,
      "height_sensor_name": "foot_height_scan",
      "command_name": "steering",
      "command_threshold": 0.15,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  )
  cfg.rewards["support_foot_tilt"] = RewardTermCfg(
    func=mdp.support_foot_tilt_penalty,
    weight=-0.08,
    params={
      "sensor_cfg": SceneEntityCfg("feet_ground_contact"),
      "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
      "contact_threshold": 1.0,
    },
  )
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=velocity_mdp.soft_landing,
    weight=-1.0e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "steering",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["action_rate"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005)

  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  if play:
    cfg.curriculum = {}
  else:
    cfg.curriculum["command_velocity"] = CurriculumTermCfg(
      func=mdp.body_velocity_command_ranges,
      params={
        "command_name": "steering",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (-0.5, 1.0),
            "lin_vel_y": (0.0, 0.0),
            "yaw_rate": (0.0, 0.0),
          },
          {
            "step": 5000 * 24,
            "lin_vel_x": (-1.0, 2.0),
            "lin_vel_y": (0.0, 0.0),
            "yaw_rate": (0.0, 0.0),
          },
          {
            "step": 10000 * 24,
            "lin_vel_x": final_range,
            "lin_vel_y": (0.0, 0.0),
            "yaw_rate": (0.0, 0.0),
          },
        ],
      },
    )

  return cfg
