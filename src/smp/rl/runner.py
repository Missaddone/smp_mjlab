"""SMP-specific RSL-RL runner extensions."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv


class SmpOnPolicyRunner(MjlabOnPolicyRunner):
  """Persist SMP reward calibration alongside policy checkpoints.

  ``DiffNormalizer`` is environment state rather than part of the policy or
  optimizer.  Keeping it at the top level of ``infos`` makes strict-resume
  checkpoints self-contained while preserving mjlab's ``env_state`` handling.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    *,
    strict_smp_normalizer_resume: bool = False,
  ) -> None:
    self.strict_smp_normalizer_resume = strict_smp_normalizer_resume
    super().__init__(env, train_cfg, log_dir, device)

  def save(self, path: str, infos=None) -> None:
    """Save the normalizer state as ``infos['smp_normalizer']``."""
    normalizer = self.env.unwrapped._smp_normalizer
    checkpoint_infos = {
      **(infos or {}),
      "smp_normalizer": normalizer.state_dict(),
    }
    identity = os.environ.get("SMP_CHECKPOINT_IDENTITY")
    if identity:
      checkpoint_infos["smp_checkpoint_identity"] = identity
    super().save(path, checkpoint_infos)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Restore the normalizer after the parent has restored the policy."""
    if self.strict_smp_normalizer_resume:
      checkpoint = torch.load(path, map_location=map_location, weights_only=False)
      checkpoint_infos = checkpoint.get("infos")
      if not isinstance(checkpoint_infos, Mapping) or "smp_normalizer" not in checkpoint_infos:
        raise RuntimeError(self._missing_normalizer_message(path))
      expected_identity = os.environ.get("SMP_EXPECTED_CHECKPOINT_IDENTITY")
      if expected_identity and checkpoint_infos.get("smp_checkpoint_identity") != expected_identity:
        raise RuntimeError(self._identity_mismatch_message(path, expected_identity, checkpoint_infos))

    infos = super().load(path, load_cfg, strict, map_location)
    normalizer_state = infos.get("smp_normalizer") if isinstance(infos, Mapping) else None
    if normalizer_state is not None:
      self.env.unwrapped._smp_normalizer.load_state_dict(normalizer_state)
      return infos

    message = self._missing_normalizer_message(path)
    if self.strict_smp_normalizer_resume:
      raise RuntimeError(message)
    warnings.warn(message, UserWarning, stacklevel=2)
    return infos

  @staticmethod
  def _missing_normalizer_message(path: str) -> str:
    return (
      f"Checkpoint '{path}' has no smp_normalizer state; retaining the startup "
      "DiffNormalizer. It cannot provide a continuous SMP reward scale."
    )

  @staticmethod
  def _identity_mismatch_message(
    path: str,
    expected_identity: str,
    checkpoint_infos: Mapping,
  ) -> str:
    return (
      f"checkpoint identity mismatch for '{path}': expected "
      f"'{expected_identity}', got '{checkpoint_infos.get('smp_checkpoint_identity')}'."
    )


class StrictSmpOnPolicyRunner(SmpOnPolicyRunner):
  """Require SMP normalizer continuity for fine-tuning tasks.

  The task registry instantiates runner classes without SMP-specific keyword
  arguments, so strict mode belongs in this small concrete subclass rather
  than in each training launcher.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(
      env,
      train_cfg,
      log_dir,
      device,
      strict_smp_normalizer_resume=True,
    )
