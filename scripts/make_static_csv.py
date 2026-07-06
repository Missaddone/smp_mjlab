"""Build a long static G1 CSV clip from a short pose CSV.

The output keeps one selected source row fixed for every frame, so root,
orientation, and joints are constant.  This produces zero velocities after the
normal CSV -> NPZ processing path.

Usage:
  uv run scripts/make_static_csv.py
  uv run scripts/make_static_csv.py --input datasets/csv/forward/stop.csv --seconds 30
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Cfg:
  input: str = "datasets/csv/forward/stop.csv"
  """Short source CSV containing the stop pose."""

  output: str = "datasets/csv/forward/stop_static.csv"
  """Output CSV path."""

  source_row: int = -1
  """Source row to repeat. Negative values follow Python indexing; -1 = last row."""

  seconds: float = 30.0
  """Static clip duration in seconds."""

  fps: int = 30
  """Output CSV frame rate. LAFAN/G1 CSVs are normally 30 fps."""

  min_rows: int = 300
  """Minimum number of rows to write, regardless of seconds."""

  expected_cols: int = 36
  """Expected G1 CSV columns: root_pos(3), quat_xyzw(4), joint_pos(29)."""

  target_output_fps: int = 50
  """Only used to print the expected resampled frame/window count."""

  window_size: int = 10
  """Only used to print the expected window count after csv_to_npz.py."""


def main(cfg: Cfg) -> None:
  input_path = Path(cfg.input)
  if not input_path.exists():
    raise FileNotFoundError(f"Input CSV not found: {input_path}")
  if cfg.seconds <= 0:
    raise ValueError("--seconds must be positive")
  if cfg.fps <= 0:
    raise ValueError("--fps must be positive")
  if cfg.min_rows <= 0:
    raise ValueError("--min-rows must be positive")

  data = np.loadtxt(input_path, delimiter=",")
  if data.ndim == 1:
    data = data[None, :]
  if data.shape[1] != cfg.expected_cols:
    raise ValueError(
      f"Expected {cfg.expected_cols} columns, got {data.shape[1]} in {input_path}"
    )

  source_row = cfg.source_row
  if source_row < 0:
    source_row = data.shape[0] + source_row
  if source_row < 0 or source_row >= data.shape[0]:
    raise IndexError(
      f"--source-row {cfg.source_row} is out of range for {data.shape[0]} rows"
    )

  num_rows = max(int(math.ceil(cfg.seconds * cfg.fps)), cfg.min_rows)
  static_clip = np.repeat(data[source_row : source_row + 1], num_rows, axis=0)

  output_path = Path(cfg.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_path, static_clip, delimiter=",", fmt="%.10g")

  duration = (num_rows - 1) / cfg.fps
  resampled_frames = int(math.ceil(duration * cfg.target_output_fps))
  windows = max(0, resampled_frames - cfg.window_size + 1)
  print(f"Input: {input_path} rows={data.shape[0]} cols={data.shape[1]}")
  print(f"Repeated source row: {source_row}")
  print(f"Output: {output_path} rows={num_rows} fps={cfg.fps} duration={duration:.3f}s")
  print(
    "Expected after csv_to_npz.py: "
    f"~{resampled_frames} frames at {cfg.target_output_fps} fps, "
    f"~{windows} windows with window_size={cfg.window_size}"
  )


if __name__ == "__main__":
  main(tyro.cli(Cfg))
