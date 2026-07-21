"""Analyze body-frame command ranges from theme CSV motions."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from mjlab.utils.lab_api.math import quat_apply_inverse


@dataclass(frozen=True)
class MotionRange:
  lin_x_min: float
  lin_x_max: float
  lin_y_min: float
  lin_y_max: float
  yaw_min: float
  yaw_max: float


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Compute min/max body-frame vx, vy, and yaw rate from theme CSV files. "
      "CSV quaternion columns are interpreted as xyzw, matching mjlab MotionLoader."
    )
  )
  parser.add_argument("--input-dir", default="datasets/csv/theme")
  parser.add_argument("--styles", nargs="+", default=["male", "female", "children"])
  parser.add_argument("--input-fps", type=int, default=30)
  parser.add_argument("--output-fps", type=int, default=50)
  parser.add_argument("--device", default="cpu")
  return parser.parse_args()


def _style_files(input_dir: Path, style: str) -> list[Path]:
  style_pattern = re.compile(rf"(^|[_\-.]){re.escape(style.lower())}([_\-.]|$)")
  style_files = [
    path
    for path in input_dir.glob("*.csv")
    if style_pattern.search(path.stem.lower())
  ]
  common_static = [
    path
    for path in input_dir.glob("*.csv")
    if any(token in path.stem.lower() for token in ("stop", "static", "stand"))
    and path not in style_files
  ]
  return sorted(style_files + common_static)


def _range_for_file(
  csv_path: Path, input_fps: int, output_fps: int, device: str
) -> MotionRange:
  motion = CsvMotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
  )
  lin_vel_b = quat_apply_inverse(motion.motion_base_rots, motion.motion_base_lin_vels)
  ang_vel_b = quat_apply_inverse(motion.motion_base_rots, motion.motion_base_ang_vels)
  return MotionRange(
    lin_x_min=float(torch.min(lin_vel_b[:, 0]).item()),
    lin_x_max=float(torch.max(lin_vel_b[:, 0]).item()),
    lin_y_min=float(torch.min(lin_vel_b[:, 1]).item()),
    lin_y_max=float(torch.max(lin_vel_b[:, 1]).item()),
    yaw_min=float(torch.min(ang_vel_b[:, 2]).item()),
    yaw_max=float(torch.max(ang_vel_b[:, 2]).item()),
  )


def _merge(ranges: list[MotionRange]) -> MotionRange:
  return MotionRange(
    lin_x_min=min(item.lin_x_min for item in ranges),
    lin_x_max=max(item.lin_x_max for item in ranges),
    lin_y_min=min(item.lin_y_min for item in ranges),
    lin_y_max=max(item.lin_y_max for item in ranges),
    yaw_min=min(item.yaw_min for item in ranges),
    yaw_max=max(item.yaw_max for item in ranges),
  )


def _format_range(item: MotionRange) -> str:
  return (
    f"x=[{item.lin_x_min:.4f}, {item.lin_x_max:.4f}], "
    f"y=[{item.lin_y_min:.4f}, {item.lin_y_max:.4f}], "
    f"yaw=[{item.yaw_min:.4f}, {item.yaw_max:.4f}]"
  )


def main() -> None:
  args = _parse_args()
  input_dir = Path(args.input_dir)
  if not input_dir.is_dir():
    raise FileNotFoundError(f"input dir not found: {input_dir}")

  for style in args.styles:
    files = _style_files(input_dir, style)
    if not files:
      print(f"[WARN] style={style}: no CSV files matched")
      continue

    file_ranges = []
    print(f"\n## style={style}")
    for csv_path in files:
      item = _range_for_file(csv_path, args.input_fps, args.output_fps, args.device)
      file_ranges.append(item)
      print(f"- {csv_path.name}: {_format_range(item)}")

    merged = _merge(file_ranges)
    print(f"[SUMMARY] style={style}: {_format_range(merged)}")


if __name__ == "__main__":
  main()
