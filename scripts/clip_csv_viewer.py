"""Visualize and trim a G1 CSV motion file with a viser GUI.

Usage:
  uv run scripts/clip_csv_viewer.py --input datasets/csv/motion.csv

The viewer plays the interpolated motion for inspection, while export writes
rows from the original CSV so the result can be fed back into csv_to_npz.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.viewer.viser.scene import MjlabViserScene

from csv_to_npz import JOINT_NAMES
from smp.utils import detect_device


@dataclass
class Cfg:
  input: str
  """Input CSV motion file."""
  output: str = ""
  """Output CSV file. Empty = <input_stem>_clip_<start>_<end>.csv in output_dir."""
  output_dir: str = "datasets/csv_clips"
  """Directory for exported clips when output is empty."""
  input_fps: int = 30
  """Frame rate of the input CSV."""
  output_fps: int = 50
  """Frame rate used for visualization interpolation."""
  playback_fps: float = 50.0
  """Playback frame rate in the viewer."""
  device: str = ""
  """Compute device. Empty = auto."""


def _setup_sim(device: str) -> tuple[Simulation, Scene]:
  sim_cfg = SimulationCfg()
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _write_motion_frame(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  joint_indexes: torch.Tensor,
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  dof_pos: torch.Tensor,
) -> None:
  root_states = robot.data.default_root_state.clone()
  root_states[:, 0:3] = base_pos
  root_states[:, :2] += scene.env_origins[:, :2]
  root_states[:, 3:7] = base_quat
  robot.write_root_state_to_sim(root_states)

  joint_pos_full = robot.data.default_joint_pos.clone()
  joint_vel_full = robot.data.default_joint_vel.clone()
  joint_pos_full[:, joint_indexes] = dof_pos
  robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

  sim.forward()
  scene.update(sim.mj_model.opt.timestep)


def _selection_to_input_rows(
  start_frame: int,
  end_frame: int,
  input_frames: int,
  input_fps: int,
  output_fps: int,
) -> tuple[int, int]:
  start_frame, end_frame = sorted((start_frame, end_frame))
  start_time = start_frame / output_fps
  end_time = end_frame / output_fps
  start_row = max(0, min(input_frames - 1, math.floor(start_time * input_fps)))
  end_row = max(0, min(input_frames - 1, math.ceil(end_time * input_fps)))
  return start_row, max(start_row, end_row)


def _unique_path(path: Path) -> Path:
  if not path.exists():
    return path
  stem = path.stem
  suffix = path.suffix
  for i in range(1, 10000):
    candidate = path.with_name(f"{stem}_{i:02d}{suffix}")
    if not candidate.exists():
      return candidate
  raise RuntimeError(f"Could not find an unused output path near {path}")


def _as_numpy(value) -> np.ndarray:
  if hasattr(value, "cpu"):
    value = value.cpu()
  if hasattr(value, "numpy"):
    value = value.numpy()
  return np.asarray(value)


def _format_status(
  frame: int,
  total_frames: int,
  start_frame: int,
  end_frame: int,
  input_rows: tuple[int, int],
  output_fps: int,
) -> str:
  start_row, end_row = input_rows
  return (
    f"Frame `{frame}` / `{total_frames - 1}`  "
    f"Time `{frame / output_fps:.3f}s`  \n"
    f"Selection frames `{start_frame}` -> `{end_frame}`  "
    f"Selection time `{start_frame / output_fps:.3f}s` -> "
    f"`{end_frame / output_fps:.3f}s`  \n"
    f"CSV rows to export `{start_row}` -> `{end_row}` "
    f"(`{end_row - start_row + 1}` rows)"
  )


def main(cfg: Cfg) -> None:
  input_path = Path(cfg.input)
  if not input_path.exists():
    raise FileNotFoundError(f"Input CSV not found: {input_path}")
  if not cfg.device:
    cfg.device = detect_device()

  print(f"Device: {cfg.device}")
  motion = CsvMotionLoader(
    motion_file=str(input_path),
    input_fps=cfg.input_fps,
    output_fps=cfg.output_fps,
    device=cfg.device,
  )
  raw_csv = np.loadtxt(input_path, delimiter=",")
  if raw_csv.ndim == 1:
    raw_csv = raw_csv[None, :]

  sim, scene = _setup_sim(cfg.device)
  robot: Entity = scene["robot"]
  joint_indexes = torch.tensor(
    robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, sim.mj_model, num_envs=1)

  state = {
    "playing": True,
    "start_frame": 0,
    "end_frame": motion.output_frames - 1,
    "last_export": "",
  }

  with server.gui.add_folder("Clip"):
    play_btn = server.gui.add_button("Play / Pause")
    set_start_btn = server.gui.add_button("Set Start")
    set_end_btn = server.gui.add_button("Set End")
    start_num = server.gui.add_number("Start Frame", initial_value=0, step=1)
    end_num = server.gui.add_number(
      "End Frame", initial_value=motion.output_frames - 1, step=1
    )
    export_btn = server.gui.add_button("Export CSV")
    status = server.gui.add_markdown("")
    export_status = server.gui.add_markdown("")
    server.gui.add_html(
      """
      <style>
        .mantine-Slider-root {
          position: fixed !important;
          left: 4vw !important;
          right: 4vw !important;
          bottom: 28px !important;
          width: auto !important;
          max-width: none !important;
          z-index: 10000 !important;
          padding: 28px 34px 38px 34px !important;
          border-radius: 12px !important;
          background: rgba(15, 23, 42, 0.86) !important;
          box-shadow: 0 16px 44px rgba(0, 0, 0, 0.36) !important;
          backdrop-filter: blur(10px);
        }
        .mantine-Slider-track {
          height: 14px !important;
        }
        .mantine-Slider-thumb {
          width: 32px !important;
          height: 32px !important;
          border-width: 4px !important;
        }
        .mantine-Slider-markLabel {
          color: rgba(255, 255, 255, 0.9) !important;
          font-size: 13px !important;
        }
        .mantine-Slider-label {
          font-size: 14px !important;
        }
        body {
          padding-bottom: 120px;
        }
      </style>
      """,
      order=90,
    )
    server.gui.add_markdown("Timeline is pinned to the bottom of the page.", order=91)
    frame_slider = server.gui.add_slider(
      "Frame",
      min=0,
      max=motion.output_frames - 1,
      step=1,
      initial_value=0,
      marks=(
        (0, "0"),
        (motion.output_frames // 2, str(motion.output_frames // 2)),
        (motion.output_frames - 1, str(motion.output_frames - 1)),
      ),
      order=92,
    )
    back_big_btn = server.gui.add_button("-50", order=93)
    back_small_btn = server.gui.add_button("-10", order=94)
    forward_small_btn = server.gui.add_button("+10", order=95)
    forward_big_btn = server.gui.add_button("+50", order=96)

  def clamp_frame(value: int | float) -> int:
    return int(max(0, min(motion.output_frames - 1, round(float(value)))))

  def sync_selection() -> None:
    state["start_frame"] = clamp_frame(state["start_frame"])
    state["end_frame"] = clamp_frame(state["end_frame"])
    start_num.value = state["start_frame"]
    end_num.value = state["end_frame"]

  def selected_rows() -> tuple[int, int]:
    return _selection_to_input_rows(
      int(state["start_frame"]),
      int(state["end_frame"]),
      motion.input_frames,
      cfg.input_fps,
      cfg.output_fps,
    )

  def refresh_status() -> None:
    status.value = _format_status(
      clamp_frame(frame_slider.value),
      motion.output_frames,
      int(state["start_frame"]),
      int(state["end_frame"]),
      selected_rows(),
      cfg.output_fps,
    )
    if state["last_export"]:
      export_status.value = state["last_export"]

  @play_btn.on_click
  def _(_evt) -> None:
    state["playing"] = not bool(state["playing"])

  def jump(delta: int) -> None:
    frame_slider.value = clamp_frame(frame_slider.value + delta)
    refresh_status()

  @back_big_btn.on_click
  def _(_evt) -> None:
    jump(-50)

  @back_small_btn.on_click
  def _(_evt) -> None:
    jump(-10)

  @forward_small_btn.on_click
  def _(_evt) -> None:
    jump(10)

  @forward_big_btn.on_click
  def _(_evt) -> None:
    jump(50)

  @set_start_btn.on_click
  def _(_evt) -> None:
    state["start_frame"] = clamp_frame(frame_slider.value)
    sync_selection()
    refresh_status()

  @set_end_btn.on_click
  def _(_evt) -> None:
    state["end_frame"] = clamp_frame(frame_slider.value)
    sync_selection()
    refresh_status()

  @start_num.on_update
  def _(_evt) -> None:
    state["start_frame"] = clamp_frame(start_num.value)
    sync_selection()
    refresh_status()

  @end_num.on_update
  def _(_evt) -> None:
    state["end_frame"] = clamp_frame(end_num.value)
    sync_selection()
    refresh_status()

  @export_btn.on_click
  def _(_evt) -> None:
    start_row, end_row = selected_rows()
    if cfg.output:
      out_path = Path(cfg.output)
    else:
      out_dir = Path(cfg.output_dir)
      out_path = out_dir / f"{input_path.stem}_clip_{start_row}_{end_row}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = _unique_path(out_path)
    np.savetxt(out_path, raw_csv[start_row : end_row + 1], delimiter=",", fmt="%.10g")
    state["last_export"] = (
      f"Exported `{out_path}`  \n"
      f"Rows `{start_row}` -> `{end_row}` from `{input_path.name}`"
    )
    refresh_status()
    print(f"Exported {out_path} rows {start_row}:{end_row + 1}")

  def render(frame: int) -> None:
    base_pos = motion.motion_base_poss[frame : frame + 1]
    base_quat = motion.motion_base_rots[frame : frame + 1]
    dof_pos = motion.motion_dof_poss[frame : frame + 1]
    _write_motion_frame(sim, scene, robot, joint_indexes, base_pos, base_quat, dof_pos)
    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=_as_numpy(wd.xpos),
      body_xmat=_as_numpy(wd.xmat),
      qpos=_as_numpy(wd.qpos),
      env_idx=0,
    )

  sync_selection()
  refresh_status()
  print("Viser server running. Open the printed URL to clip the CSV.")

  dt_play = 1.0 / cfg.playback_fps
  try:
    while True:
      frame = clamp_frame(frame_slider.value)
      render(frame)
      refresh_status()
      if state["playing"]:
        frame_slider.value = (frame + 1) % motion.output_frames
      time.sleep(dt_play)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
