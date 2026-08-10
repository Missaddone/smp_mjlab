import unittest
from pathlib import Path

import torch
from mjlab.tasks.registry import load_runner_cls

from smp.rl import SmpOnPolicyRunner, StrictSmpOnPolicyRunner
from smp.rl.tasks.body_velocity.body_velocity_exp14_new_dev_env_cfg import (
  EXP14_NEW_DEV_BODY_VELOCITY_GROUP_SPECS,
  _build_exp14_new_dev_body_velocity_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_exp15_env_cfg import (
  EXP15_BODY_VELOCITY_GROUP_SPECS,
)
from smp.rl.tasks.body_velocity.body_velocity_exp16_env_cfg import (
  EXP16_BODY_VELOCITY_GROUP_SPECS,
  _build_exp16_body_velocity_cfg,
)
from smp.rl.tasks.body_velocity.body_velocity_new_dev_exp13_env_cfg import (
  EXP13_NEW_DEV_BODY_VELOCITY_GROUP_SPECS,
  _build_new_dev_exp13_body_velocity_cfg,
)
from smp.rl.tasks.body_velocity.mdp.diagnostics import action_diagnostic_values


class Exp15SpecificationTest(unittest.TestCase):
  def test_group_specs_cover_the_approved_thirteen_groups(self):
    specs = EXP15_BODY_VELOCITY_GROUP_SPECS
    self.assertEqual([spec.group for spec in specs], list(range(1, 14)))
    self.assertEqual([spec.iterations for spec in specs[:2]], [1000, 3000])
    self.assertTrue(all(spec.static_tilt_weight == 0.0 for spec in specs[:2]))
    self.assertEqual(
      [spec.static_tilt_weight for spec in specs[2:8]],
      [0.05, 0.1, 0.2, 0.05, 0.1, 0.2],
    )
    self.assertEqual(
      [spec.moving_tilt_weight for spec in specs[9:]],
      [0.02, 0.05, 0.02, 0.05],
    )
    self.assertTrue(
      all(spec.static_tilt_weight == 0.1 for spec in specs[8:]),
    )


class NewDevExp13Exp14SpecificationTest(unittest.TestCase):
  def test_new_dev_exp13_uses_the_fully_mirrored_retrained_prior(self):
    self.assertEqual(
      [spec.group for spec in EXP13_NEW_DEV_BODY_VELOCITY_GROUP_SPECS],
      [4, 5, 6],
    )
    for spec in EXP13_NEW_DEV_BODY_VELOCITY_GROUP_SPECS:
      with self.subTest(group=spec.group):
        cfg = _build_new_dev_exp13_body_velocity_cfg(spec)
        self.assertEqual(
          cfg.events["init_smp_state"].params["ckpt_path"],
          "datasets/pretrain_ckpt/new_dev_exp13_loco_stop_static_mirrored.pt",
        )

  def test_exp14_new_dev_specs_keep_parent_reward_and_isolate_flatfoot_mode(self):
    specs = EXP14_NEW_DEV_BODY_VELOCITY_GROUP_SPECS
    self.assertEqual([spec.group for spec in specs], list(range(1, 10)))
    self.assertEqual([spec.parent_group for spec in specs], [4, 5, 6] * 3)
    self.assertEqual(
      [spec.kind for spec in specs],
      ["control"] * 3 + ["moving"] * 3 + ["static"] * 3,
    )
    self.assertTrue(all(spec.iterations == 100 for spec in specs))
    self.assertTrue(
      all(spec.moving_tilt_weight == 0.0 and spec.static_tilt_weight == 0.0
          for spec in specs[:3])
    )
    self.assertTrue(
      all(spec.moving_tilt_weight == 0.1 and spec.static_tilt_weight == 0.0
          for spec in specs[3:6])
    )
    self.assertTrue(
      all(spec.moving_tilt_weight == 0.0 and spec.static_tilt_weight == 0.1
          for spec in specs[6:])
    )

  def test_new_dev_registry_uses_resumable_and_strict_smp_runners(self):
    for parent_group in (4, 5, 6):
      self.assertIs(
        load_runner_cls(f"Smp-BodyVelocity-NewDev-Exp13-Group{parent_group}-G1"),
        SmpOnPolicyRunner,
      )

  def test_exp14_new_dev_builders_add_only_the_requested_reward_terms(self):
    for spec in EXP14_NEW_DEV_BODY_VELOCITY_GROUP_SPECS:
      with self.subTest(group=spec.group):
        cfg = _build_exp14_new_dev_body_velocity_cfg(spec)
        extras = {
          key: value
          for key, value in cfg.rewards.items()
          if key.startswith("new_dev_exp14_")
        }
        if spec.kind == "control":
          self.assertEqual(extras, {})
        elif spec.kind == "moving":
          self.assertEqual(set(extras), {"new_dev_exp14_moving_support_foot_tilt"})
          term = extras["new_dev_exp14_moving_support_foot_tilt"]
          self.assertEqual(term.weight, -0.1)
          self.assertEqual(term.params["contact_threshold"], 1.0)
          self.assertEqual(term.params["min_contact_time"], 0.06)
        else:
          self.assertEqual(
            set(extras),
            {
              "new_dev_exp14_static_left_foot_tilt",
              "new_dev_exp14_static_right_foot_tilt",
            },
          )
          self.assertTrue(all(term.weight == -0.1 for term in extras.values()))
    for group in range(1, 10):
      self.assertIs(
        load_runner_cls(f"Smp-BodyVelocity-NewDev-Exp14-Group{group}-G1"),
        StrictSmpOnPolicyRunner,
      )


class Exp16SpecificationTest(unittest.TestCase):
  def test_exp16_builders_use_motebu_prior_exp13_reward_mix_and_reduced_commands(self):
    specs = EXP16_BODY_VELOCITY_GROUP_SPECS
    self.assertEqual([spec.group for spec in specs], [1, 2, 3])
    self.assertEqual([spec.exp13_reward_group for spec in specs], [4, 5, 6])
    self.assertTrue(all(spec.parent_group is None for spec in specs))

    expected_mixes = ((0.5, 0.25, 0.25), (0.6, 0.2, 0.2), (0.7, 0.15, 0.15))
    for spec, expected_mix in zip(specs, expected_mixes, strict=True):
      with self.subTest(group=spec.group):
        cfg = _build_exp16_body_velocity_cfg(spec)
        command = cfg.commands["body_velocity"]
        self.assertEqual(
          (command.lin_vel_x_min, command.lin_vel_x_max), (-1.5, 1.5)
        )
        self.assertEqual(
          (command.lin_vel_y_min, command.lin_vel_y_max), (-1.5, 1.5)
        )
        self.assertEqual((command.yaw_rate_min, command.yaw_rate_max), (-2.0, 2.0))
        self.assertEqual(command.zero_command_prob, 0.3)
        self.assertEqual(
          cfg.events["init_smp_state"].params["ckpt_path"],
          "datasets/pretrain_ckpt/exp16_motebu_stop_static.pt",
        )
        task_terms = cfg.rewards["task_smp_product"].params["task_terms"]
        moving_params = next(
          params for _, _, params in task_terms if "moving_product_weight" in params
        )
        self.assertEqual(
          (
            moving_params["moving_product_weight"],
            moving_params["moving_linear_weight"],
            moving_params["moving_yaw_weight"],
          ),
          expected_mix,
        )
        self.assertFalse(any("tilt" in name for name in cfg.rewards))
        self.assertIs(
          load_runner_cls(f"Smp-BodyVelocity-Exp16-Group{spec.group}-G1"),
          SmpOnPolicyRunner,
        )

  def test_exp16_scripts_stage_exactly_two_sources_and_two_mirrors(self):
    root = Path(__file__).resolve().parents[1]
    prepare_script = (root / "scripts/run_exp16_prepare_motebu_stop_static_prior.sh").read_text()
    train_script = (root / "scripts/run_exp16_groups1_3.sh").read_text()
    play_script = (root / "scripts/play_exp16_groups1_3_wandb.sh").read_text()

    self.assertIn("datasets/csv/theme/motebu.csv", prepare_script)
    self.assertIn("datasets/csv/theme/stop_static.csv", prepare_script)
    self.assertIn("motebu_mirror.csv", prepare_script)
    self.assertIn("stop_static_mirror.csv", prepare_script)
    self.assertIn("expected_raw_files", prepare_script)
    self.assertIn('INPUT_FPS="${INPUT_FPS:-30}"', prepare_script)
    self.assertIn('OUTPUT_FPS="${OUTPUT_FPS:-50}"', prepare_script)
    self.assertIn("datasets/pretrain_ckpt/exp16_motebu_stop_static.pt", prepare_script)
    self.assertIn("--agent.max-iterations=10000", train_script)
    self.assertNotIn("--agent.resume=True", train_script)
    self.assertIn("Smp-BodyVelocity-Exp16-Group${GROUP}-G1", train_script)
    self.assertIn("VIEWER=viser", play_script)
    self.assertIn("Smp-BodyVelocity-Exp16-Group${GROUP}-G1", play_script)

  def test_new_dev_scripts_require_strict_exp14_source_and_preserve_exp13_scratch(self):
    root = Path(__file__).resolve().parents[1]
    prepare_script = (root / "scripts/run_new_dev_exp13_prepare_prior.sh").read_text()
    exp13_script = (root / "scripts/run_new_dev_exp13_groups4_6.sh").read_text()
    exp14_script = (root / "scripts/run_new_dev_exp14_groups1_9.sh").read_text()
    play_script = (root / "scripts/play_new_dev_exp14_groups1_9_wandb.sh").read_text()

    self.assertIn("Groups 4-6 train from scratch for 10000 iterations", exp13_script)
    self.assertIn("loco", prepare_script)
    self.assertIn("stop_static", prepare_script)
    self.assertIn("ensure_missing_mirrors", prepare_script)
    self.assertIn(
      "datasets/pretrain_ckpt/new_dev_exp13_loco_stop_static_mirrored.pt",
      prepare_script,
    )
    self.assertNotIn("--agent.resume=True", exp13_script)
    self.assertIn('SMP_CHECKPOINT_IDENTITY="new_dev_exp13_g${GROUP}"', exp13_script)
    self.assertIn(
      "<group> <gpu> <source_wandb_run_path> [checkpoint]",
      exp14_script,
    )
    self.assertNotIn("<parent_group>", exp14_script)
    self.assertNotIn("PARENT_GROUP", exp14_script)
    self.assertIn("EXPECTED_CHECKPOINT_IDENTITY", exp14_script)
    self.assertIn("SMP_EXPECTED_CHECKPOINT_IDENTITY", exp14_script)
    self.assertIn("--agent.resume=True", exp14_script)
    self.assertIn("--agent.max-iterations=100", exp14_script)
    self.assertIn("--agent.algorithm.learning-rate=1e-4", exp14_script)
    self.assertIn("--agent.algorithm.desired-kl=0.002", exp14_script)
    self.assertIn("--agent.algorithm.num-learning-epochs=1", exp14_script)
    self.assertIn("Smp-BodyVelocity-NewDev-Exp14-Group${GROUP}-G1", play_script)

class DiagnosticTensorTest(unittest.TestCase):
  def test_action_diagnostics_measure_first_and_second_differences(self):
    action = torch.tensor([[1.0, -1.0, 0.0]])
    previous = torch.zeros_like(action)
    previous_previous = torch.zeros_like(action)
    joint_vel = torch.tensor([[3.0, 4.0, 0.0]])

    values = action_diagnostic_values(
      action,
      previous,
      previous_previous,
      joint_vel,
    )

    self.assertAlmostEqual(values["action_delta_rms"].item(), (2.0 / 3.0) ** 0.5)
    self.assertAlmostEqual(values["action_delta_peak"].item(), 1.0)
    self.assertAlmostEqual(values["action_accel_proxy_peak"].item(), 1.0)
    self.assertAlmostEqual(
      values["joint_vel_rms"].item(), (25.0 / 3.0) ** 0.5, places=6
    )
    self.assertAlmostEqual(values["joint_vel_peak"].item(), 4.0)

  def test_leg_and_foot_diagnostics_detect_alternating_policy_outputs(self):
    action = torch.tensor([[1.0, -1.0, 0.1, -0.1]])
    previous = torch.zeros_like(action)
    previous_previous = torch.tensor([[1.0, -1.0, 0.1, -0.1]])
    joint_vel = torch.zeros_like(action)

    values = action_diagnostic_values(
      action,
      previous,
      previous_previous,
      joint_vel,
      action_groups={
        "leg": torch.tensor([0, 1]),
        "foot": torch.tensor([2, 3]),
      },
    )

    self.assertAlmostEqual(values["leg_action_sign_flip_rate"].item(), 1.0)
    self.assertAlmostEqual(values["foot_action_sign_flip_rate"].item(), 1.0)
    self.assertAlmostEqual(values["leg_action_accel_proxy_peak"].item(), 2.0)
    self.assertAlmostEqual(values["foot_action_accel_proxy_peak"].item(), 0.2)
    self.assertAlmostEqual(values["leg_joint_vel_peak"].item(), 0.0)
    self.assertAlmostEqual(values["foot_joint_vel_peak"].item(), 0.0)


if __name__ == "__main__":
  unittest.main()
