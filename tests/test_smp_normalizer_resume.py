import os
import unittest
import warnings
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner

from smp.rl import SmpOnPolicyRunner, StrictSmpOnPolicyRunner
from smp.rl.utils import DiffNormalizer


class DiffNormalizerResumeTest(unittest.TestCase):
  def _make_normalizer(self) -> DiffNormalizer:
    return DiffNormalizer(num_timesteps=3, device=torch.device("cpu"))

  def test_round_trip_preserves_the_next_normalized_output(self):
    source = self._make_normalizer()
    source.update_and_normalize(1, torch.tensor([2.0, 4.0]))

    state = source.state_dict()
    restored = self._make_normalizer()
    restored.load_state_dict(state)

    next_mse = torch.tensor([3.0, 9.0])
    self.assertTrue(
      torch.allclose(
        source.update_and_normalize(1, next_mse.clone()),
        restored.update_and_normalize(1, next_mse.clone()),
      )
    )
    self.assertEqual(state["version"], 1)
    self.assertEqual(state["num_timesteps"], 3)
    self.assertEqual(state["mean"].device.type, "cpu")
    self.assertEqual(state["count"].device.type, "cpu")

  def test_load_rejects_invalid_state(self):
    valid = self._make_normalizer().state_dict()
    cases = {
      "missing_schema": {key: value for key, value in valid.items() if key != "mean"},
      "wrong_shape": {**valid, "mean": torch.ones(4)},
      "nonfinite_mean": {**valid, "mean": torch.tensor([1.0, float("nan"), 1.0])},
      "nonpositive_mean": {**valid, "mean": torch.tensor([1.0, 0.0, 1.0])},
      "negative_count": {**valid, "count": torch.tensor([0, -1, 0])},
      "float_count": {**valid, "count": torch.tensor([0.0, 1.0, 0.0])},
      "nonfinite_count": {**valid, "count": torch.tensor([0.0, float("inf"), 0.0])},
      "bool_min_value": {**valid, "min_value": True},
    }

    for name, state in cases.items():
      with self.subTest(name=name):
        with self.assertRaises(ValueError):
          self._make_normalizer().load_state_dict(state)


class SmpOnPolicyRunnerResumeTest(unittest.TestCase):
  @staticmethod
  def _runner(normalizer: DiffNormalizer, *, strict: bool = False) -> SmpOnPolicyRunner:
    runner = object.__new__(SmpOnPolicyRunner)
    runner.env = type(
      "FakeWrappedEnv",
      (),
      {"unwrapped": type("FakeEnv", (), {"_smp_normalizer": normalizer})()},
    )()
    runner.strict_smp_normalizer_resume = strict
    return runner

  @staticmethod
  def _normalizer() -> DiffNormalizer:
    return DiffNormalizer(num_timesteps=3, device=torch.device("cpu"))

  def _assert_normalizer_states_equal(self, actual, expected) -> None:
    self.assertEqual(actual["version"], expected["version"])
    self.assertEqual(actual["num_timesteps"], expected["num_timesteps"])
    self.assertEqual(actual["min_value"], expected["min_value"])
    self.assertEqual(actual["max_count"], expected["max_count"])
    self.assertTrue(torch.equal(actual["mean"], expected["mean"]))
    self.assertTrue(torch.equal(actual["count"], expected["count"]))

  def test_save_places_normalizer_state_at_infos_top_level(self):
    normalizer = self._normalizer()
    normalizer.update_and_normalize(1, torch.tensor([2.0, 4.0]))
    runner = self._runner(normalizer)

    with patch.object(MjlabOnPolicyRunner, "save") as parent_save:
      runner.save("checkpoint.pt", infos={"caller": "kept"})

    _, saved_infos = parent_save.call_args.args
    self.assertEqual(saved_infos["caller"], "kept")
    self._assert_normalizer_states_equal(saved_infos["smp_normalizer"], normalizer.state_dict())
    self.assertNotIn("env_state", saved_infos["smp_normalizer"])

  def test_save_records_configured_checkpoint_identity(self):
    runner = self._runner(self._normalizer())

    with patch.dict(os.environ, {"SMP_CHECKPOINT_IDENTITY": "new_dev_exp13_g4"}):
      with patch.object(MjlabOnPolicyRunner, "save") as parent_save:
        runner.save("checkpoint.pt")

    _, saved_infos = parent_save.call_args.args
    self.assertEqual(saved_infos["smp_checkpoint_identity"], "new_dev_exp13_g4")

  def test_load_restores_normalizer_from_parent_infos(self):
    source = self._normalizer()
    source.update_and_normalize(2, torch.tensor([3.0, 9.0]))
    target = self._normalizer()
    runner = self._runner(target)

    with patch.object(
      MjlabOnPolicyRunner,
      "load",
      return_value={"smp_normalizer": source.state_dict()},
    ):
      infos = runner.load("checkpoint.pt", strict=False)

    self._assert_normalizer_states_equal(infos["smp_normalizer"], source.state_dict())
    self.assertTrue(torch.equal(target.mean, source.mean))
    self.assertTrue(torch.equal(target.count, source.count))

  def test_load_legacy_checkpoint_raises_in_strict_mode(self):
    runner = self._runner(self._normalizer(), strict=True)

    with TemporaryDirectory() as directory:
      checkpoint_path = f"{directory}/legacy.pt"
      torch.save({"infos": {"env_state": {}}}, checkpoint_path)
      with patch.object(MjlabOnPolicyRunner, "load") as parent_load:
        with self.assertRaisesRegex(RuntimeError, "smp_normalizer"):
          runner.load(checkpoint_path)

    parent_load.assert_not_called()

  def test_load_legacy_checkpoint_warns_and_keeps_startup_normalizer(self):
    normalizer = self._normalizer()
    expected_mean = normalizer.mean.clone()
    expected_count = normalizer.count.clone()
    runner = self._runner(normalizer)

    with patch.object(MjlabOnPolicyRunner, "load", return_value={"env_state": {}}):
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.load("legacy.pt", strict=False)

    self.assertTrue(any("smp_normalizer" in str(item.message) for item in caught))
    self.assertTrue(torch.equal(normalizer.mean, expected_mean))
    self.assertTrue(torch.equal(normalizer.count, expected_count))

  def test_strict_load_rejects_malformed_normalizer_before_parent_load(self):
    runner = self._runner(self._normalizer(), strict=True)
    malformed = {
      **self._normalizer().state_dict(),
      "mean": torch.ones(4),
    }

    with TemporaryDirectory() as directory:
      checkpoint_path = f"{directory}/malformed.pt"
      torch.save({"infos": {"smp_normalizer": malformed}}, checkpoint_path)
      with patch.object(MjlabOnPolicyRunner, "load") as parent_load:
        with self.assertRaisesRegex(ValueError, "invalid shape"):
          runner.load(checkpoint_path)

    parent_load.assert_not_called()

  def test_loaded_checkpoint_identity_is_preserved_on_the_next_save(self):
    runner = self._runner(self._normalizer(), strict=True)
    source = self._normalizer()
    infos = {
      "smp_normalizer": source.state_dict(),
      "smp_checkpoint_identity": "new_dev_exp13_g4",
    }

    with TemporaryDirectory() as directory:
      checkpoint_path = f"{directory}/parent.pt"
      torch.save({"infos": infos}, checkpoint_path)
      with patch.dict(
        os.environ,
        {"SMP_EXPECTED_CHECKPOINT_IDENTITY": "new_dev_exp13_g4"},
        clear=True,
      ):
        with patch.object(MjlabOnPolicyRunner, "load", return_value=infos):
          runner.load(checkpoint_path)

      with patch.dict(os.environ, {}, clear=True):
        with patch.object(MjlabOnPolicyRunner, "save") as parent_save:
          runner.save("child.pt")

    _, saved_infos = parent_save.call_args.args
    self.assertEqual(
      saved_infos["smp_checkpoint_identity"],
      "new_dev_exp13_g4",
    )

  def test_strict_load_rejects_mismatched_checkpoint_identity_before_parent_load(self):
    runner = self._runner(self._normalizer(), strict=True)
    source = self._normalizer()

    with TemporaryDirectory() as directory:
      checkpoint_path = f"{directory}/wrong-parent.pt"
      torch.save(
        {
          "infos": {
            "smp_normalizer": source.state_dict(),
            "smp_checkpoint_identity": "new_dev_exp13_g4",
          }
        },
        checkpoint_path,
      )
      with patch.dict(os.environ, {"SMP_EXPECTED_CHECKPOINT_IDENTITY": "new_dev_exp13_g5"}):
        with patch.object(MjlabOnPolicyRunner, "load") as parent_load:
          with self.assertRaisesRegex(RuntimeError, "checkpoint identity"):
            runner.load(checkpoint_path)

    parent_load.assert_not_called()

  def test_strict_runner_enables_normalizer_continuity_without_registry_kwargs(self):
    env = object()
    cfg = {}
    with patch.object(SmpOnPolicyRunner, "__init__", return_value=None) as parent_init:
      StrictSmpOnPolicyRunner(env, cfg, "logs", "cpu")

    parent_init.assert_called_once_with(
      env,
      cfg,
      "logs",
      "cpu",
      strict_smp_normalizer_resume=True,
    )


if __name__ == "__main__":
  unittest.main()
