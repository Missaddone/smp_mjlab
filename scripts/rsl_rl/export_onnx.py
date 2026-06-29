"""Export an RSL-RL actor checkpoint to ONNX without launching simulation.

This direct exporter reads the actor MLP and observation normalizer from a
checkpoint produced by the mjlab RSL-RL runner. It exports the deterministic
actor mean used for deployment.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


class NormalizedActor(nn.Module):
  """Observation normalizer followed by the actor MLP."""

  def __init__(
    self,
    mean: torch.Tensor,
    std: torch.Tensor,
    mlp: nn.Sequential,
    eps: float = 1.0e-2,
  ):
    super().__init__()
    self.register_buffer("mean", mean)
    self.register_buffer("std", std)
    self.eps = eps
    self.mlp = mlp

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    obs = (obs - self.mean) / (self.std + self.eps)
    return self.mlp(obs)


def _activation(name: str) -> nn.Module:
  name = name.lower()
  if name == "elu":
    return nn.ELU()
  if name == "relu":
    return nn.ReLU()
  if name == "tanh":
    return nn.Tanh()
  raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(actor_state: dict[str, torch.Tensor], activation: str) -> nn.Sequential:
  layers: OrderedDict[str, nn.Module] = OrderedDict()
  linear_ids = sorted(
    int(key.split(".")[1])
    for key in actor_state
    if key.startswith("mlp.") and key.endswith(".weight")
  )
  if not linear_ids:
    raise RuntimeError("Could not find actor MLP weights in checkpoint.")

  for i, layer_id in enumerate(linear_ids):
    weight = actor_state[f"mlp.{layer_id}.weight"]
    bias = actor_state[f"mlp.{layer_id}.bias"]
    linear = nn.Linear(weight.shape[1], weight.shape[0])
    linear.weight.data.copy_(weight)
    linear.bias.data.copy_(bias)
    layers[f"linear_{i}"] = linear
    if i != len(linear_ids) - 1:
      layers[f"{activation}_{i}"] = _activation(activation)
  return nn.Sequential(layers)


def _normalizer_tensors(actor_state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
  first_weight = next(
    value for key, value in actor_state.items() if key.startswith("mlp.") and key.endswith(".weight")
  )
  obs_dim = first_weight.shape[1]
  mean = actor_state.get("obs_normalizer._mean")
  std = actor_state.get("obs_normalizer._std")
  if mean is None:
    mean = torch.zeros(obs_dim, dtype=first_weight.dtype)
  if std is None:
    std = torch.ones(obs_dim, dtype=first_weight.dtype)
  return mean.clone(), std.clone()


def load_actor(checkpoint_path: str | os.PathLike[str], activation: str = "elu") -> NormalizedActor:
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  if "actor_state_dict" not in ckpt:
    raise RuntimeError(f"Checkpoint does not contain `actor_state_dict`: {checkpoint_path}")

  actor_state = ckpt["actor_state_dict"]
  mean, std = _normalizer_tensors(actor_state)
  mlp = _build_mlp(actor_state, activation=activation)
  model = NormalizedActor(mean=mean, std=std, mlp=mlp)
  model.eval()
  return model


def main() -> None:
  parser = argparse.ArgumentParser(description="Export an RSL-RL actor checkpoint to ONNX.")
  parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt.")
  parser.add_argument("--output-dir", default=None, help="Directory for the ONNX file.")
  parser.add_argument("--output-name", default="policy.onnx", help="ONNX filename.")
  parser.add_argument(
    "--activation",
    default="elu",
    choices=("elu", "relu", "tanh"),
    help="Actor hidden activation.",
  )
  parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
  parser.add_argument("--verbose", action="store_true", help="Enable verbose torch.onnx export.")
  args = parser.parse_args()

  checkpoint_path = Path(args.checkpoint).expanduser().resolve()
  output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint_path.parent / "exported"
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / args.output_name

  model = load_actor(checkpoint_path, activation=args.activation)
  obs_dim = int(model.mean.shape[-1])
  action_dim = int(model.mlp[-1].out_features)
  dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

  print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
  print(f"[INFO] Actor input dim: {obs_dim}, action dim: {action_dim}")
  print(f"[INFO] Exporting ONNX to: {output_path}")

  torch.onnx.export(
    model,
    dummy_obs,
    str(output_path),
    export_params=True,
    opset_version=args.opset,
    dynamo=False,
    external_data=False,
    verbose=args.verbose,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
  )

  print("[INFO] ONNX export complete.")


if __name__ == "__main__":
  main()
