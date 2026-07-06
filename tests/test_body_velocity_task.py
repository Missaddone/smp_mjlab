import importlib
import unittest


class BodyVelocityTaskCfgTest(unittest.TestCase):
  def test_env_cfg_uses_body_velocity_command_obs_and_reward_split(self):
    from smp.rl import rewards as smp_rewards
    from smp.rl.tasks.body_velocity import mdp
    from smp.rl.tasks.body_velocity.body_velocity_env_cfg import (
      g1_body_velocity_smp_env_cfg,
    )

    cfg = g1_body_velocity_smp_env_cfg(play=False)

    self.assertIn("body_velocity", cfg.commands)
    self.assertIsInstance(cfg.commands["body_velocity"], mdp.BodyVelocityCommandCfg)
    self.assertEqual(cfg.commands["body_velocity"].lin_vel_x_min, -1.0)
    self.assertEqual(cfg.commands["body_velocity"].lin_vel_x_max, 2.0)
    self.assertEqual(cfg.commands["body_velocity"].lin_vel_y_min, -1.0)
    self.assertEqual(cfg.commands["body_velocity"].lin_vel_y_max, 1.0)
    self.assertEqual(cfg.commands["body_velocity"].yaw_rate_min, -1.0)
    self.assertEqual(cfg.commands["body_velocity"].yaw_rate_max, 1.0)

    self.assertEqual(
      cfg.observations["actor"].terms["command"].params["command_name"],
      "body_velocity",
    )
    self.assertEqual(
      cfg.observations["critic"].terms["command"].params["command_name"],
      "body_velocity",
    )
    self.assertNotIn("base_lin_vel", cfg.observations["actor"].terms)
    self.assertIn("base_lin_vel", cfg.observations["critic"].terms)

    reward = cfg.rewards["task_smp_product"]
    self.assertIs(reward.func, smp_rewards.body_velocity_task_smp_product)
    self.assertEqual(reward.params["fixed_timesteps"], (8, 15, 22))
    self.assertEqual(reward.params["ws"], 6.0)
    self.assertEqual(len(reward.params["task_terms"]), 2)
    linear_term, yaw_term = reward.params["task_terms"]
    self.assertIs(linear_term[0], mdp.body_velocity_linear_tracking)
    self.assertEqual(linear_term[1], 0.75)
    self.assertEqual(linear_term[2]["command_name"], "body_velocity")
    self.assertEqual(linear_term[2]["lin_vel_err_scale"], 2.0)
    self.assertIs(yaw_term[0], mdp.body_velocity_yaw_tracking)
    self.assertEqual(yaw_term[1], 0.25)
    self.assertEqual(yaw_term[2]["command_name"], "body_velocity")
    self.assertEqual(yaw_term[2]["yaw_rate_err_scale"], 1.0)
    self.assertEqual(
      cfg.events["init_smp_state"].params["ckpt_path"],
      "datasets/pretrain_ckpt/pretrained_lafan_run.pt",
    )

  def test_body_velocity_task_is_registered_from_tasks_package(self):
    import mjlab.tasks.registry as registry

    importlib.import_module("smp.rl.tasks")

    self.assertIn("Smp-BodyVelocity-G1", registry.list_tasks())

  def test_exp4_group_env_cfgs_encode_ablation_table(self):
    from smp.rl import rewards as smp_rewards
    from smp.rl.tasks.body_velocity import mdp

    cases = (
      (
        "smp.rl.tasks.body_velocity.body_velocity_exp4_group1_env_cfg",
        "g1_body_velocity_exp4_group1_smp_env_cfg",
        1.0,
        1.0,
        0.7,
        0.15,
        0.15,
        mdp.body_velocity_linear_tracking,
      ),
      (
        "smp.rl.tasks.body_velocity.body_velocity_exp4_group2_env_cfg",
        "g1_body_velocity_exp4_group2_smp_env_cfg",
        2.0,
        1.0,
        0.7,
        0.15,
        0.15,
        mdp.body_velocity_linear_tracking,
      ),
      (
        "smp.rl.tasks.body_velocity.body_velocity_exp4_group3_env_cfg",
        "g1_body_velocity_exp4_group3_smp_env_cfg",
        1.0,
        0.5,
        0.7,
        0.15,
        0.15,
        mdp.body_velocity_linear_tracking,
      ),
      (
        "smp.rl.tasks.body_velocity.body_velocity_exp4_group4_env_cfg",
        "g1_body_velocity_exp4_group4_smp_env_cfg",
        1.0,
        0.5,
        0.0,
        0.5,
        0.5,
        mdp.body_velocity_linear_tracking,
      ),
      (
        "smp.rl.tasks.body_velocity.body_velocity_exp4_group5_env_cfg",
        "g1_body_velocity_exp4_group5_smp_env_cfg",
        1.0,
        0.5,
        0.0,
        0.5,
        0.5,
        mdp.body_velocity_linear_tracking_zero_negative_projection,
      ),
      (
        "smp.rl.tasks.body_velocity.body_velocity_exp4_group6_env_cfg",
        "g1_body_velocity_exp4_group6_smp_env_cfg",
        1.0,
        1.0,
        0.6,
        0.2,
        0.2,
        mdp.body_velocity_linear_tracking,
      ),
      (
        "smp.rl.tasks.body_velocity.body_velocity_exp4_group7_env_cfg",
        "g1_body_velocity_exp4_group7_smp_env_cfg",
        1.0,
        1.0,
        0.8,
        0.1,
        0.1,
        mdp.body_velocity_linear_tracking,
      ),
    )

    for (
      module_name,
      builder_name,
      k_xy,
      k_yaw,
      product_weight,
      linear_weight,
      yaw_weight,
      linear_reward_func,
    ) in cases:
      with self.subTest(builder=builder_name):
        module = importlib.import_module(module_name)
        cfg = getattr(module, builder_name)(play=False)
        reward = cfg.rewards["task_smp_product"]
        self.assertIs(reward.func, smp_rewards.body_velocity_task_smp_mix)
        self.assertIs(reward.params["linear_reward_func"], linear_reward_func)
        self.assertIs(reward.params["yaw_reward_func"], mdp.body_velocity_yaw_tracking)
        self.assertEqual(reward.params["lin_vel_err_scale"], k_xy)
        self.assertEqual(reward.params["yaw_rate_err_scale"], k_yaw)
        self.assertEqual(reward.params["product_weight"], product_weight)
        self.assertEqual(reward.params["linear_weight"], linear_weight)
        self.assertEqual(reward.params["yaw_weight"], yaw_weight)
        self.assertEqual(
          cfg.events["init_smp_state"].params["ckpt_path"],
          "datasets/pretrain_ckpt/pretrained_lafan_run.pt",
        )

  def test_exp4_group_tasks_are_registered(self):
    import mjlab.tasks.registry as registry

    importlib.import_module("smp.rl.tasks")

    tasks = registry.list_tasks()
    for group in range(1, 8):
      self.assertIn(f"Smp-BodyVelocity-Exp4-Group{group}-G1", tasks)


if __name__ == "__main__":
  unittest.main()
