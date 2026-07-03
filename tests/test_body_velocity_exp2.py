import math
import unittest

import torch

from smp.rl.tasks.steering import mdp
from smp.rl.tasks.steering.body_velocity_env_cfg import (
  g1_body_velocity_exp2_smp_env_cfg,
  g1_body_velocity_static_exp2_smp_env_cfg,
)


class BodyVelocityExp2Test(unittest.TestCase):
  def test_exp2_reward_terms_use_standard_task_smp_product_name(self):
    for env_cfg_builder in (
      g1_body_velocity_exp2_smp_env_cfg,
      g1_body_velocity_static_exp2_smp_env_cfg,
    ):
      cfg = env_cfg_builder(play=False)
      self.assertIn("task_smp_product", cfg.rewards)
      self.assertNotIn("task_smp_product_exp2", cfg.rewards)
      self.assertNotIn("task_smp_product_static_exp2", cfg.rewards)

  def test_body_velocity_command_cfg_accepts_static_probability(self):
    cfg = mdp.BodyVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      static_prob=0.25,
    )

    self.assertEqual(cfg.static_prob, 0.25)

  def test_static_mask_requires_each_component_under_threshold(self):
    lin_vel = torch.tensor(
      [
        [0.049, -0.049],
        [0.050, 0.000],
        [0.000, -0.051],
      ]
    )
    yaw_rate = torch.tensor([0.049, 0.000, 0.049])

    mask = mdp.body_velocity_static_command_mask(
      lin_vel,
      yaw_rate,
      lin_vel_threshold=0.05,
      yaw_rate_threshold=0.05,
    )

    torch.testing.assert_close(mask, torch.tensor([True, False, False]))

  def test_blended_linear_yaw_task_formula(self):
    lin_err = torch.tensor([0.25])
    yaw_err = torch.tensor([0.04])

    actual = mdp.body_velocity_linear_yaw_task_reward(
      lin_err,
      yaw_err,
      lin_vel_err_scale=2.0,
      yaw_rate_err_scale=1.0,
    )

    r_l = math.exp(-2.0 * 0.25)
    r_y = math.exp(-1.0 * 0.04)
    expected = 0.7 * r_l * r_y + 0.15 * r_l + 0.15 * r_y
    torch.testing.assert_close(actual, torch.tensor([expected], dtype=actual.dtype))

  def test_static_task_formula_multiplies_per_foot_rewards(self):
    root_lin_err = torch.tensor([0.01])
    root_yaw_err = torch.tensor([0.04])
    foot_vel_xy = torch.tensor([[[0.10, 0.00], [0.00, 0.20]]])

    actual = mdp.body_velocity_static_task_reward(
      root_lin_err,
      root_yaw_err,
      foot_vel_xy,
      foot_vel_err_scale=2.0,
    )

    r_l = math.exp(-2.0 * 0.01)
    r_y = math.exp(-1.0 * 0.04)
    r_foot_l = math.exp(-2.0 * 0.01)
    r_foot_r = math.exp(-2.0 * 0.04)
    expected = (0.7 * r_l * r_y + 0.15 * r_l + 0.15 * r_y) * r_foot_l * r_foot_r
    torch.testing.assert_close(actual, torch.tensor([expected], dtype=actual.dtype))


if __name__ == "__main__":
  unittest.main()
