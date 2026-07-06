"""G1 body-velocity task with SMP guidance.

Each env gets a body-frame xy velocity and yaw-rate target, periodically
resampled.  Reward = body-velocity tracking task reward, gated by SMP guidance.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import body_velocity_task_smp_mix, body_velocity_task_smp_product
from smp.rl.tasks.body_velocity import mdp


def g1_body_velocity_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 body-velocity env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["body_velocity"] = mdp.BodyVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    lin_vel_x_min=-1.0,
    lin_vel_x_max=2.0,
    lin_vel_y_min=-1.0,
    lin_vel_y_max=1.0,
    yaw_rate_min=-1.0,
    yaw_rate_max=1.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "body_velocity"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs
  cfg.observations["actor"].terms.pop("base_lin_vel", None)

  # --- Rewards -------------------------------------------------------------
  # task = 0.75·body linear velocity tracking + 0.25·yaw-rate tracking,
  # gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=body_velocity_task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.body_velocity_linear_tracking,
          0.75,
          {"command_name": "body_velocity", "lin_vel_err_scale": 2.0},
        ),
        (
          mdp.body_velocity_yaw_tracking,
          0.25,
          {"command_name": "body_velocity", "yaw_rate_err_scale": 1.0},
        ),
      ),
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_lafan_run.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg


def g1_body_velocity_exp4_smp_env_cfg(
  play: bool = False,
  *,
  lin_vel_err_scale: float,
  yaw_rate_err_scale: float,
  product_weight: float,
  linear_weight: float,
  yaw_weight: float,
  linear_reward_func=mdp.body_velocity_linear_tracking,
) -> ManagerBasedRlEnvCfg:
  """Build an Experiment 4 body-velocity cfg with a specified reward mixture."""
  cfg = g1_body_velocity_smp_env_cfg(play=play)
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=body_velocity_task_smp_mix,
    weight=1.0,
    params={
      "linear_reward_func": linear_reward_func,
      "yaw_reward_func": mdp.body_velocity_yaw_tracking,
      "command_name": "body_velocity",
      "lin_vel_err_scale": lin_vel_err_scale,
      "yaw_rate_err_scale": yaw_rate_err_scale,
      "product_weight": product_weight,
      "linear_weight": linear_weight,
      "yaw_weight": yaw_weight,
      "fixed_timesteps": (8, 15, 22),
      "ws": 6.0,
    },
  )
  return cfg
