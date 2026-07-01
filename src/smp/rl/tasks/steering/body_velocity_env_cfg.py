"""Body-velocity and steering variants migrated from the IsaacLab SMP tasks."""

from __future__ import annotations

from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.terminations import bad_orientation
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from smp.rl.rewards import smp_reward_component_log
from smp.rl.tasks.steering import mdp
from smp.rl.tasks.steering.steering_env_cfg import g1_steering_smp_env_cfg

PRETRAIN_CKPT_DIR = "datasets/pretrain_ckpt"
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SITE_NAMES = ("left_foot", "right_foot")
CORE_AND_LEG_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
)


def _command_obs() -> ObservationTermCfg:
  return ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "steering"})


def _set_command_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  command_obs = _command_obs()
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs


def _set_body_velocity_actor_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  _set_command_obs(cfg)
  cfg.observations["actor"].terms.pop("base_lin_vel", None)


def _body_velocity_command(
  lin_vel_x_min: float,
  lin_vel_x_max: float,
  lin_vel_y_min: float,
  lin_vel_y_max: float,
  yaw_rate_min: float,
  yaw_rate_max: float,
) -> mdp.BodyVelocityCommandCfg:
  return mdp.BodyVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    lin_vel_x_min=lin_vel_x_min,
    lin_vel_x_max=lin_vel_x_max,
    lin_vel_y_min=lin_vel_y_min,
    lin_vel_y_max=lin_vel_y_max,
    yaw_rate_min=yaw_rate_min,
    yaw_rate_max=yaw_rate_max,
  )


def _body_velocity_reward(
  style_floor: float = 0.0,
  lin_vel_err_scale: float = 2.0,
  yaw_rate_err_scale: float = 1.0,
) -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.body_velocity_task_smp_product,
    weight=1.0,
    params={
      "command_name": "steering",
      "lin_vel_err_scale": lin_vel_err_scale,
      "yaw_rate_err_scale": yaw_rate_err_scale,
      "lin_vel_weight": 0.75,
      "yaw_rate_weight": 0.25,
      "style_floor": style_floor,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )


def _body_velocity_sum_reward(
  task_weight: float = 1.0,
  style_weight: float = 1.0,
) -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.body_velocity_task_smp_sum,
    weight=1.0,
    params={
      "command_name": "steering",
      "lin_vel_err_scale": 2.0,
      "yaw_rate_err_scale": 1.0,
      "lin_vel_weight": 0.75,
      "yaw_rate_weight": 0.25,
      "task_weight": task_weight,
      "style_weight": style_weight,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )


def _body_velocity_poly_task_reward() -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.body_velocity_task_component,
    weight=0.6,
    params={
      "command_name": "steering",
      "lin_vel_err_scale": 2.0,
      "yaw_rate_err_scale": 1.0,
      "lin_vel_weight": 0.75,
      "yaw_rate_weight": 0.25,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )


def _cached_smp_component_reward(
  component: str,
  weight: float,
) -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.smp_cached_component,
    weight=weight,
    params={"component": component},
  )


def _set_body_velocity_poly_rewards(
  cfg: ManagerBasedRlEnvCfg,
  task_weight: float = 0.6,
  style_weight: float = 0.2,
  product_weight: float = 0.3,
) -> None:
  cfg.rewards.clear()
  cfg.rewards["task_reward"] = _body_velocity_poly_task_reward()
  cfg.rewards["task_reward"].weight = task_weight
  cfg.rewards["style_reward"] = _cached_smp_component_reward("style_reward", style_weight)
  cfg.rewards["task_style_product"] = _cached_smp_component_reward(
    "task_smp_product",
    product_weight,
  )


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


def _body_velocity_base_cfg(
  play: bool,
  ckpt_name: str,
  command_cfg: mdp.BodyVelocityCommandCfg,
  reward_cfg: RewardTermCfg | None = None,
  reward_name: str = "task_smp_product",
) -> ManagerBasedRlEnvCfg:
  cfg = g1_steering_smp_env_cfg(play=play)
  command_cfg.debug_vis = play
  cfg.commands["steering"] = command_cfg
  _set_body_velocity_actor_obs(cfg)
  cfg.events["init_smp_state"].params["ckpt_path"] = f"{PRETRAIN_CKPT_DIR}/{ckpt_name}"
  cfg.terminations["bad_orientation"] = TerminationTermCfg(
    func=bad_orientation,
    params={
      "limit_angle": 1.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards.clear()
  cfg.rewards[reward_name] = reward_cfg or _body_velocity_reward(style_floor=0.0)
  _add_smp_reward_component_logs(cfg)
  return cfg


def g1_body_velocity_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    # "pretrained_lafan_run.pt",
    # "lafan_run_clips_lafan_norm_3600.pt",
    # "lafan_walk_clips_lafan_norm_10800.pt",
    # "amp_lafan_walk_clips2_mirrored_lafan_norm_128.pt",
    # "amp_lafan_walk_clips2_mirrored_lafan_norm_128.pt",
    "amp_walk_clips2_mirrored_lafan_norm_128.pt",
    _body_velocity_command(-1.0, 2.0, -1.0, 1.0, -1.0, 1.0),
  )


def g1_body_velocity_unitree_ref_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  command_cfg = _body_velocity_command(-1.0, 2.0, -1.0, 1.0, -1.0, 1.0)
  command_cfg.heading_command = True
  command_cfg.rel_standing_envs = 0.05
  command_cfg.rel_single_axis_envs = 0.1
  command_cfg.command_deadzone = 0.1
  command_cfg.single_axis_min_abs = 0.1
  command_cfg.heading_control_stiffness = 0.0
  command_cfg.rel_heading_envs = 0.5
  cfg = _body_velocity_base_cfg(
    play,
    "amp_walk_clips2_mirrored_lafan_norm_128.pt",
    command_cfg,
    reward_cfg=_body_velocity_reward(lin_vel_err_scale=2.0, yaw_rate_err_scale=1.0),
    reward_name="task_smp_product_unitree_ref",
  )
  _add_sensor(
    cfg,
    ContactSensorCfg(
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
    ),
  )
  cfg.observations["critic"].terms["foot_contact"] = ObservationTermCfg(
    func=velocity_mdp.foot_contact,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.observations["critic"].terms["foot_contact_forces"] = ObservationTermCfg(
    func=velocity_mdp.foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=velocity_mdp.soft_landing,
    weight=-1.0e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "steering",
      "command_threshold": 0.1,
    },
  )
  cfg.rewards["action_rate"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.000)
  return cfg


def g1_body_velocity_sum_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "amp_walk_clips2_mirrored_lafan_norm_128.pt",
    _body_velocity_command(-1.0, 2.0, -1.0, 1.0, -1.0, 1.0),
    reward_cfg=_body_velocity_sum_reward(task_weight=1.0, style_weight=5.0),
    reward_name="task_smp_sum",
  )


def g1_body_velocity_poly_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _body_velocity_base_cfg(
    play,
    "amp_walk_clips2_mirrored_lafan_norm_128.pt",
    _body_velocity_command(-1.0, 2.0, -1.0, 1.0, -1.0, 1.0),
  )
  _set_body_velocity_poly_rewards(
    cfg,
    task_weight=0.6,
    style_weight=0.1,
    product_weight=0.3,
  )
  return cfg


def g1_body_velocity_run_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "amp_run.pt",
    _body_velocity_command(-1.0, 3.0, -1.0, 1.0, -2.0, 2.0),
  )


def g1_body_velocity_walk_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "lafan_walk_clips2_lafan_norm_12000.pt",
    _body_velocity_command(
      -1.0,
      3.0,
      -1.0,
      1.0,
      -2.0,
      2.0,
    ),
  )


def g1_steering_modified_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_steering_smp_env_cfg(play=play)
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=True,
    rand_face_dir=False,
    tar_speed_min=-0.2,
    tar_speed_max=5.0,
    debug_vis=True,
  )
  cfg.rewards.clear()
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=mdp.steering_task_smp_product,
    weight=1.0,
    params={
      "command_name": "steering",
      "vel_err_scale": 1.0,
      "velocity_weight": 1.5,
      "face_weight": 0.5,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )
  _add_smp_reward_component_logs(cfg)
  return cfg


def _add_sensor(cfg: ManagerBasedRlEnvCfg, sensor: ContactSensorCfg) -> None:
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (sensor,)


def g1_body_velocity_foot_regularized_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_body_velocity_smp_env_cfg(play=play)
  _add_sensor(
    cfg,
    ContactSensorCfg(
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
    ),
  )
  sensor_name = "feet_ground_contact"
  sensor_cfg = SceneEntityCfg(sensor_name)
  foot_asset_cfg = SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES)
  cfg.rewards["feet_air_time"] = RewardTermCfg(
    func=velocity_mdp.feet_air_time,
    weight=0.25,
    params={
      "sensor_name": sensor_name,
      "threshold_min": 0.05,
      "threshold_max": 0.4,
      "command_name": "steering",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["feet_slide"] = RewardTermCfg(
    func=velocity_mdp.feet_slip,
    weight=-0.1,
    params={
      "sensor_name": sensor_name,
      "command_name": "steering",
      "command_threshold": 0.15,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  )
  cfg.rewards["support_foot_tilt"] = RewardTermCfg(
    func=mdp.support_foot_tilt_penalty,
    weight=-0.1,
    params={"sensor_cfg": sensor_cfg, "asset_cfg": foot_asset_cfg, "contact_threshold": 1.0},
  )
  cfg.rewards["persistent_single_support"] = RewardTermCfg(
    func=mdp.persistent_single_support_penalty,
    weight=-0.2,
    params={
      "command_name": "steering",
      "sensor_cfg": sensor_cfg,
      "max_single_support_time": 0.6,
      "max_excess_time": 1.0,
      "command_speed_threshold": 0.15,
    },
  )
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=velocity_mdp.soft_landing,
    weight=-1.0e-5,
    params={
      "sensor_name": sensor_name,
      "command_name": "steering",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["action_rate"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005)
  return cfg


BODY_VELOCITY_TASKS: dict[str, Callable[[bool], ManagerBasedRlEnvCfg]] = {
  "BodyVelocity": g1_body_velocity_smp_env_cfg,
  "BodyVelocity-UnitreeRef": g1_body_velocity_unitree_ref_smp_env_cfg,
  "BodyVelocity-Sum": g1_body_velocity_sum_smp_env_cfg,
  "BodyVelocity-Poly": g1_body_velocity_poly_smp_env_cfg,
  "BodyVelocity-Walk": g1_body_velocity_walk_smp_env_cfg,
  "BodyVelocity-Run": g1_body_velocity_run_smp_env_cfg,
  "BodyVelocity-FootRegularized": g1_body_velocity_foot_regularized_smp_env_cfg,
}
