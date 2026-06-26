"""Body-velocity and steering variants migrated from the IsaacLab SMP tasks."""

from __future__ import annotations

from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from smp.rl.events import init_smp_double_prior_state
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
  stand_sample_prob: float = 0.0,
  stand_lin_vel_x_min: float = -0.15,
  stand_lin_vel_x_max: float = 0.15,
  stand_lin_vel_y_min: float = -0.15,
  stand_lin_vel_y_max: float = 0.15,
  stand_yaw_rate_max: float = 0.2,
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
    stand_sample_prob=stand_sample_prob,
    stand_lin_vel_x_min=stand_lin_vel_x_min,
    stand_lin_vel_x_max=stand_lin_vel_x_max,
    stand_lin_vel_y_min=stand_lin_vel_y_min,
    stand_lin_vel_y_max=stand_lin_vel_y_max,
    stand_yaw_rate_max=stand_yaw_rate_max,
  )


def _body_velocity_reward(style_floor: float = 0.0, use_stand_branch: bool = False) -> RewardTermCfg:
  return RewardTermCfg(
    func=mdp.body_velocity_task_smp_product,
    weight=1.0,
    params={
      "command_name": "steering",
      "lin_vel_err_scale": 2.0,
      "yaw_rate_err_scale": 1.0,
      "lin_vel_weight": 0.75,
      "yaw_rate_weight": 0.25,
      "use_stand_branch": use_stand_branch,
      "style_floor": style_floor,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )


def _body_velocity_base_cfg(
  play: bool,
  ckpt_name: str,
  command_cfg: mdp.BodyVelocityCommandCfg,
  reward_cfg: RewardTermCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  cfg = g1_steering_smp_env_cfg(play=play)
  cfg.commands["steering"] = command_cfg
  _set_body_velocity_actor_obs(cfg)
  cfg.events["init_smp_state"].params["ckpt_path"] = f"{PRETRAIN_CKPT_DIR}/{ckpt_name}"
  cfg.rewards.clear()
  cfg.rewards["task_smp_blend"] = reward_cfg or _body_velocity_reward()
  return cfg


def g1_body_velocity_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "pretrained_lafan_run.pt",
    _body_velocity_command(0.0, 3.0, -3.0, 3.0, -2.0, 2.0),
  )


def g1_body_velocity_lafan_run_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "lafan_run_all_norm.pt",
    # "pretrained_lafan_run.pt",
    _body_velocity_command(
      1.0,
      5.0,
      0.0,
      0.0,
      -0.0,
      0.0,
    ),
  )


def g1_body_velocity_amp_run_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "amp_run.pt",
    _body_velocity_command(0.0, 3.0, -3.0, 3.0, -2.0, 2.0),
  )


def g1_body_velocity_lafan_walk_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "lafan_walk_all_norm.pt",
    _body_velocity_command(
      1.0,
      3.0,
      0.0,
      0.0,
      -0.0,
      0.0,
    ),
  )


def g1_body_velocity_lafan_walk_matched_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return g1_body_velocity_lafan_walk_smp_env_cfg(play=play)


def g1_body_velocity_lafan_walk_run_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "lafan_walk_run_local_nrom.pt",
    _body_velocity_command(0.25, 5.0, -5.0, 5.0, -2.0, 2.0),
  )


def g1_body_velocity_amp_all_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _body_velocity_base_cfg(
    play,
    "amp_all.pt",
    _body_velocity_command(0.0, 5.0, -5.0, 5.0, -2.0, 2.0),
  )


def g1_body_velocity_lafan_walk_posture_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_body_velocity_lafan_walk_smp_env_cfg(play=play)
  cfg.commands["steering"] = _body_velocity_command(0.15, 3.0, -3.0, 3.0, -1.0, 1.0)
  cfg.rewards["base_upright"] = RewardTermCfg(func=mdp.base_upright_penalty, weight=-0.2)
  cfg.rewards["root_height"] = RewardTermCfg(func=mdp.root_height_below_target_penalty, weight=-0.2)
  cfg.rewards["action_rate"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005)
  return cfg


def g1_steering_modified_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_steering_smp_env_cfg(play=play)
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=True,
    rand_face_dir=False,
    tar_speed_min=-0.2,
    tar_speed_max=5.0,
    stand_sample_prob=0.1,
    stand_speed=0.0,
    run_speed_min=0.4,
    run_speed_max=5.0,
    debug_vis=True,
  )
  cfg.rewards.clear()
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=mdp.steering_modified_stand_branch_reward,
    weight=1.0,
    params={
      "command_name": "steering",
      "vel_err_scale": 1.0,
      "velocity_weight": 1.5,
      "face_weight": 0.5,
      "deadzone_face_weight": 0.0,
      "deadzone_stand_weight": 0.7,
      "deadzone_lin_vel_penalty_weight": 2.0,
      "deadzone_joint_vel_penalty_weight": 0.07,
      "deadzone_action_penalty_weight": 0.07,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )
  return cfg


def g1_steering_with_stand_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _body_velocity_base_cfg(
    play,
    "pretrained_lafan_run.pt",
    _body_velocity_command(0.3, 3.0, -3.0, 3.0, -1.0, 1.0, stand_sample_prob=0.2),
    RewardTermCfg(
      func=mdp.body_velocity_task_smp_product,
      weight=1.0,
      params={
        "command_name": "steering",
        "lin_vel_err_scale": 2.0,
        "yaw_rate_err_scale": 1.0,
        "lin_vel_weight": 0.75,
        "yaw_rate_weight": 0.25,
        "stand_speed_threshold": 0.2,
        "stand_yaw_rate_threshold": 0.2,
        "fixed_timesteps": (8, 15, 22),
        "ws": 6.0,
      },
    ),
  )
  return cfg


def g1_zero_velocity_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _body_velocity_base_cfg(
    play,
    "pretrained_jushen_stand.pt",
    _body_velocity_command(0.0, 0.5, -0.5, 0.5, 0.0, 0.0),
    RewardTermCfg(
      func=mdp.body_velocity_task_smp_product,
      weight=1.0,
      params={
        "command_name": "steering",
        "lin_vel_err_scale": 2.0,
        "yaw_rate_err_scale": 1.0,
        "lin_vel_weight": 1.0,
        "yaw_rate_weight": 0.0,
        "zero_lin_vel_target": True,
        "use_stand_branch": False,
        "fixed_timesteps": (8, 15, 22),
        "ws": 6.0,
      },
    ),
  )
  cfg.rewards["action_rate"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
  return cfg


def g1_steering_double_prior_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_steering_smp_env_cfg(play=play)
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=True,
    rand_face_dir=False,
    tar_speed_min=-0.5,
    tar_speed_max=5.0,
    speed_deadzone=0.1,
    deadzone_sample_prob=0.4,
    debug_vis=True,
  )
  cfg.events["init_smp_state"] = EventTermCfg(
    func=init_smp_double_prior_state,
    mode="startup",
    params={
      "moving_ckpt_path": f"{PRETRAIN_CKPT_DIR}/pretrained_lafan_run.pt",
      "stand_ckpt_path": f"{PRETRAIN_CKPT_DIR}/pretrained_jushen_stand.pt",
      "gsi_buffer_size": 4096,
      "gsi_batch_size": 1024,
      "compile_model": True,
      "compile_mode": "max-autotune",
    },
  )
  cfg.rewards.clear()
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=mdp.steering_doubleprior_task_reward,
    weight=1.0,
    params={
      "command_name": "steering",
      "vel_err_scale": 1.0,
      "velocity_weight": 1.5,
      "face_weight": 0.5,
      "deadzone_face_weight": 0.0,
      "deadzone_stand_weight": 1.0,
      "deadzone_lin_vel_penalty_weight": 2.0,
      "deadzone_joint_vel_penalty_weight": 0.1,
      "deadzone_action_penalty_weight": 0.1,
      "fixed_timesteps": (8, 15, 22),
      "moving_ws": 6.0,
      "stand_ws": 6.0,
      "moving_prior_name": "moving",
      "stand_prior_name": "stand",
    },
  )
  return cfg


def _add_sensor(cfg: ManagerBasedRlEnvCfg, sensor: ContactSensorCfg) -> None:
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (sensor,)


def g1_body_velocity_hand_collision_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_body_velocity_smp_env_cfg(play=play)
  for side in ("left", "right"):
    _add_sensor(
      cfg,
      ContactSensorCfg(
        name=f"{side}_hand_body_contact",
        primary=ContactMatch(mode="body", pattern=f"{side}_wrist_yaw_link", entity="robot"),
        # mjlab's ContactSensor supports a single secondary filter. The original
        # IsaacLab task used a one-to-many filter list, so this mjlab port uses
        # the native broader "hand contacts anything" signal.
        secondary=None,
        fields=("force",),
        reduce="maxforce",
        history_length=3,
      ),
    )
    cfg.rewards[f"{side}_hand_body_collision"] = RewardTermCfg(
      func=mdp.filtered_contact_force_penalty,
      weight=-0.2,
      params={
        "sensor_cfg": SceneEntityCfg(f"{side}_hand_body_contact"),
        "threshold": 2.0,
        "saturation_force": 20.0,
      },
    )
  return cfg


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
  "BodyVelocity-LafanRun": g1_body_velocity_lafan_run_smp_env_cfg,
  "BodyVelocity-AmpRun": g1_body_velocity_amp_run_smp_env_cfg,
  "BodyVelocity-LafanWalk": g1_body_velocity_lafan_walk_smp_env_cfg,
  "BodyVelocity-LafanWalk-Matched": g1_body_velocity_lafan_walk_matched_smp_env_cfg,
  "BodyVelocity-LafanWalk-Posture": g1_body_velocity_lafan_walk_posture_smp_env_cfg,
  "BodyVelocity-LafanWalkRun": g1_body_velocity_lafan_walk_run_smp_env_cfg,
  "BodyVelocity-AmpAll": g1_body_velocity_amp_all_smp_env_cfg,
  "BodyVelocity-HandCollision": g1_body_velocity_hand_collision_smp_env_cfg,
  "BodyVelocity-FootRegularized": g1_body_velocity_foot_regularized_smp_env_cfg,
  "ZeroVelocity": g1_zero_velocity_smp_env_cfg,
}
