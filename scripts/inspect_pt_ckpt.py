"""Inspect training-related metadata in local PyTorch checkpoint files.

Usage:
  uv run scripts/inspect_pt_ckpt.py
  uv run scripts/inspect_pt_ckpt.py datasets/pretrain_ckpt
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_CKPT_DIR = "datasets/pretrained-ckpt"
FALLBACK_CKPT_DIR = "datasets/pretrain_ckpt"
MODEL_KEYS = {
  "model",
  "model_ema",
  "module",
  "net",
  "network",
  "state_dict",
  "model_state_dict",
}
TRAINING_KEYS = {
  "args",
  "cfg",
  "config",
  "epoch",
  "global_step",
  "hparams",
  "hyper_parameters",
  "iteration",
  "lr_scheduler",
  "optimizer",
  "scheduler",
  "step",
  "train_loss",
  "val_loss",
}


def _is_tensor_like(value: Any) -> bool:
  cls = type(value)
  return hasattr(value, "shape") and hasattr(value, "dtype") and (
    (cls.__module__.startswith("torch") and cls.__name__ == "Tensor")
    or cls.__module__.startswith("numpy")
  )


def _tensor_summary(value: Any) -> str:
  shape = tuple(value.shape)
  cls = type(value)
  if cls.__module__.startswith("torch"):
    return f"Tensor(shape={shape}, dtype={value.dtype})"
  return f"{cls.__name__}(shape={shape}, dtype={value.dtype})"


def _looks_like_state_dict(value: Any) -> bool:
  if not isinstance(value, Mapping) or not value:
    return False
  values = list(value.values())
  tensor_count = sum(_is_tensor_like(v) for v in values)
  return tensor_count >= max(1, len(values) // 2)


def _state_dict_summary(value: Mapping[Any, Any], *, max_items: int) -> list[str]:
  lines = [f"parameter/state dict: {len(value)} entries"]
  shown = 0
  for key, item in value.items():
    if _is_tensor_like(item):
      lines.append(f"  - {key}: {_tensor_summary(item)}")
      shown += 1
    if shown >= max_items:
      break
  if len(value) > shown:
    lines.append(f"  ... {len(value) - shown} more entries hidden")
  return lines


def _optimizer_summary(value: Any) -> list[str]:
  if not isinstance(value, Mapping):
    return [f"optimizer: {type(value).__name__}"]

  lines = ["optimizer:"]
  state = value.get("state")
  param_groups = value.get("param_groups")
  if isinstance(state, Mapping):
    lines.append(f"  state entries: {len(state)}")
  if isinstance(param_groups, Sequence):
    lines.append(f"  param_groups: {len(param_groups)}")
    for idx, group in enumerate(param_groups):
      if not isinstance(group, Mapping):
        continue
      brief = {
        key: group[key]
        for key in ("lr", "weight_decay", "betas", "eps", "momentum")
        if key in group
      }
      lines.append(f"    [{idx}] {brief}")
  return lines


def _summarize_value(
  name: str,
  value: Any,
  *,
  depth: int,
  max_depth: int,
  max_items: int,
) -> list[str]:
  indent = "  " * depth

  if name in MODEL_KEYS or _looks_like_state_dict(value):
    return [f"{indent}{line}" for line in _state_dict_summary(value, max_items=max_items)]

  if name == "optimizer":
    return [f"{indent}{line}" for line in _optimizer_summary(value)]

  if _is_tensor_like(value):
    return [f"{indent}{name}: {_tensor_summary(value)}"]

  if isinstance(value, Mapping):
    lines = [f"{indent}{name}: dict({len(value)})"]
    if depth >= max_depth:
      return lines
    for idx, (key, item) in enumerate(value.items()):
      if idx >= max_items:
        lines.append(f"{indent}  ... {len(value) - max_items} more keys hidden")
        break
      lines.extend(
        _summarize_value(
          str(key),
          item,
          depth=depth + 1,
          max_depth=max_depth,
          max_items=max_items,
        )
      )
    return lines

  if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
    if len(value) > max_items:
      return [f"{indent}{name}: {type(value).__name__}(len={len(value)})"]
    return [f"{indent}{name}: {value!r}"]

  return [f"{indent}{name}: {value!r} ({type(value).__name__})"]


def inspect_file(path: Path, *, max_items: int, max_depth: int) -> None:
  size_mb = path.stat().st_size / 1024 / 1024
  print(f"\n== {path} ({size_mb:.1f} MB) ==")

  try:
    import torch
  except ModuleNotFoundError as exc:
    raise RuntimeError(
      "PyTorch is required to read .pt files. Run this with the project's "
      "training environment, for example: uv run scripts/inspect_pt_ckpt.py"
    ) from exc

  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(ckpt, Mapping):
    print(f"root object: {type(ckpt).__name__}")
    for line in _summarize_value(
      "root", ckpt, depth=0, max_depth=max_depth, max_items=max_items
    ):
      print(line)
    return

  keys = list(ckpt.keys())
  print(f"top-level keys ({len(keys)}): {keys}")

  printed = set()
  for key in keys:
    if str(key) in TRAINING_KEYS:
      printed.add(key)
      for line in _summarize_value(
        str(key),
        ckpt[key],
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
      ):
        print(line)

  for key in keys:
    if key in printed:
      continue
    for line in _summarize_value(
      str(key),
      ckpt[key],
      depth=0,
      max_depth=max_depth,
      max_items=max_items,
    ):
      print(line)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Show keys and training metadata from PyTorch .pt checkpoints."
  )
  parser.add_argument(
    "ckpt_dir",
    nargs="?",
    default=DEFAULT_CKPT_DIR,
    help=f"checkpoint directory to scan (default: {DEFAULT_CKPT_DIR})",
  )
  parser.add_argument("--pattern", default="*.pt", help="file pattern to scan")
  parser.add_argument("--max-items", type=int, default=8, help="max nested items to show")
  parser.add_argument("--max-depth", type=int, default=2, help="max dict nesting to show")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  ckpt_dir = Path(args.ckpt_dir)
  if (
    args.ckpt_dir == DEFAULT_CKPT_DIR
    and not ckpt_dir.exists()
    and Path(FALLBACK_CKPT_DIR).exists()
  ):
    print(f"[WARN] {DEFAULT_CKPT_DIR} not found; using {FALLBACK_CKPT_DIR}")
    ckpt_dir = Path(FALLBACK_CKPT_DIR)

  if not ckpt_dir.exists():
    raise FileNotFoundError(f"checkpoint directory does not exist: {ckpt_dir}")

  paths = sorted(ckpt_dir.rglob(args.pattern))
  if not paths:
    raise FileNotFoundError(f"no files matching {args.pattern!r} under {ckpt_dir}")

  for path in paths:
    inspect_file(path, max_items=args.max_items, max_depth=args.max_depth)


if __name__ == "__main__":
  main()
