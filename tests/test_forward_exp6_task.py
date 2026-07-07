import importlib
import unittest

import torch


class ForwardExp6CfgTest(unittest.TestCase):
  def test_original_forward_command_is_unchanged(self):
    from smp.rl.tasks.steering.forward_env_cfg import g1_forward_smp_env_cfg

    cfg = g1_forward_smp_env_cfg(play=False)
    command = cfg.commands["steering"]

    self.assertEqual(command.tar_speed_min, 0.5)
    self.assertEqual(command.tar_speed_max, 5.0)
    self.assertEqual(command.zero_speed_prob, 0.0)
    self.assertEqual(command.low_speed_prob, 0.0)
    self.assertEqual(command.dead_zone_speed, 0.0)

  def test_exp6_forward_group_command_configs(self):
    cases = (
      (
        "smp.rl.tasks.steering.forward_exp6_group4_env_cfg",
        "g1_forward_exp6_group4_smp_env_cfg",
        0.0,
        5.0,
        0.2,
        0.0,
        0.0,
        1.5,
        0.0,
      ),
      (
        "smp.rl.tasks.steering.forward_exp6_group5_env_cfg",
        "g1_forward_exp6_group5_smp_env_cfg",
        1.5,
        5.0,
        0.2,
        0.4,
        0.0,
        1.5,
        0.0,
      ),
      (
        "smp.rl.tasks.steering.forward_exp6_group6_env_cfg",
        "g1_forward_exp6_group6_smp_env_cfg",
        0.0,
        5.0,
        0.0,
        0.0,
        0.0,
        1.5,
        1.0,
      ),
    )

    for (
      module_name,
      builder_name,
      tar_speed_min,
      tar_speed_max,
      zero_speed_prob,
      low_speed_prob,
      low_speed_min,
      low_speed_max,
      dead_zone_speed,
    ) in cases:
      with self.subTest(builder=builder_name):
        module = importlib.import_module(module_name)
        cfg = getattr(module, builder_name)(play=False)
        command = cfg.commands["steering"]
        self.assertEqual(command.tar_speed_min, tar_speed_min)
        self.assertEqual(command.tar_speed_max, tar_speed_max)
        self.assertEqual(command.zero_speed_prob, zero_speed_prob)
        self.assertEqual(command.low_speed_prob, low_speed_prob)
        self.assertEqual(command.low_speed_min, low_speed_min)
        self.assertEqual(command.low_speed_max, low_speed_max)
        self.assertEqual(command.dead_zone_speed, dead_zone_speed)
        self.assertEqual(
          cfg.events["init_smp_state"].params["ckpt_path"],
          "datasets/pretrain_ckpt/pretrained_loco.pt",
        )

  def test_exp6_forward_tasks_are_registered(self):
    import mjlab.tasks.registry as registry

    importlib.import_module("smp.rl.tasks")

    tasks = registry.list_tasks()
    for group in range(4, 7):
      self.assertIn(f"Smp-Forward-Exp6-Group{group}-G1", tasks)

  def test_speed_sampler_supports_zero_low_and_dead_zone(self):
    from smp.rl.tasks.steering.mdp.commands import sample_target_speeds

    random_values = torch.tensor([0.1, 0.3, 0.7, 0.95])
    uniform_values = torch.tensor([0.0, 0.5, 0.5, 1.0])
    speeds = sample_target_speeds(
      random_values=random_values,
      uniform_values=uniform_values,
      zero_speed_prob=0.2,
      low_speed_prob=0.4,
      low_speed_min=0.0,
      low_speed_max=1.5,
      tar_speed_min=1.5,
      tar_speed_max=5.0,
      dead_zone_speed=0.0,
    )
    expected = torch.tensor([0.0, 0.75, 3.25, 5.0])
    self.assertTrue(torch.allclose(speeds, expected))

    speeds = sample_target_speeds(
      random_values=torch.tensor([0.0, 0.0, 0.0]),
      uniform_values=torch.tensor([0.1, 0.2, 0.9]),
      zero_speed_prob=0.0,
      low_speed_prob=0.0,
      low_speed_min=0.0,
      low_speed_max=1.5,
      tar_speed_min=0.0,
      tar_speed_max=5.0,
      dead_zone_speed=1.0,
    )
    expected = torch.tensor([0.0, 1.0, 4.5])
    self.assertTrue(torch.allclose(speeds, expected))


if __name__ == "__main__":
  unittest.main()
