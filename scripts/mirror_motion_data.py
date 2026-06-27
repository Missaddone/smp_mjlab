"""Mirror G1 motion CSV/NPZ files for left-right data augmentation.

The mirror plane is the robot sagittal x-z plane:

* polar vectors: y -> -y
* axial vectors: x,z -> -x,-z
* left/right joints and end-effectors are swapped
* roll/yaw joints change sign, pitch-like joints keep sign

CSV inputs are assumed to use the mjlab G1 motion layout:
``base_pos(3), base_quat_wxyz(4), joint_pos(29)``.

NPZ inputs are assumed to contain ``windows`` with the SMP feature layout:
``root_pos(3), root_rot_6d(6), joint_pos(29), ee_pos(5*3), root_lin_vel(3),
root_ang_vel(3)``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

CSV_ROOT_DIMS = 7

JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

FEATURE_DIMS = (3, 6, len(JOINT_NAMES), len(EE_BODY_NAMES) * 3, 3, 3)
TOTAL_FEATURE_DIM = sum(FEATURE_DIMS)


@dataclass
class Cfg:
  input_dir: str
  """Directory containing .csv and/or .npz motion files."""

  output_dir: str
  """Directory to write mirrored files."""

  include_original: bool = False
  """Copy original files into output_dir in addition to mirrored files."""

  recursive: bool = False
  """Search input_dir recursively."""

  overwrite: bool = False
  """Allow replacing existing output files."""

  csv: bool = True
  """Process CSV files."""

  npz: bool = True
  """Process NPZ files."""


def _swap_left_right(name: str) -> str:
  if "left_" in name:
    return name.replace("left_", "right_")
  if "right_" in name:
    return name.replace("right_", "left_")
  return name


def _joint_sign(name: str) -> float:
  # Mirror over x-z plane: pitch/flexion are unchanged; roll/yaw change sign.
  if name.endswith("_roll_joint") or name.endswith("_yaw_joint"):
    return -1.0
  return 1.0


def _as_str_tuple(values: np.ndarray | tuple[str, ...]) -> tuple[str, ...]:
  return tuple(str(v) for v in values)


def _mirror_joint_array(joint: np.ndarray, joint_names: tuple[str, ...]) -> np.ndarray:
  name_to_idx = {name: i for i, name in enumerate(joint_names)}
  mirrored = np.empty_like(joint)
  for dst_idx, dst_name in enumerate(joint_names):
    src_name = _swap_left_right(dst_name)
    if src_name not in name_to_idx:
      raise KeyError(f"Joint '{src_name}' required to mirror '{dst_name}' was not found.")
    mirrored[..., dst_idx] = joint[..., name_to_idx[src_name]] * _joint_sign(dst_name)
  return mirrored


def _mirror_polar_vector(vec: np.ndarray) -> np.ndarray:
  mirrored = vec.copy()
  mirrored[..., 1] *= -1.0
  return mirrored


def _mirror_axial_vector(vec: np.ndarray) -> np.ndarray:
  mirrored = vec.copy()
  mirrored[..., 0] *= -1.0
  mirrored[..., 2] *= -1.0
  return mirrored


def _mirror_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  # Equivalent to R' = diag(1,-1,1) @ R @ diag(1,-1,1).
  mirrored = quat.copy()
  mirrored[..., 1] *= -1.0
  mirrored[..., 3] *= -1.0
  return mirrored


def _mirror_ee_array(ee_flat: np.ndarray, ee_names: tuple[str, ...]) -> np.ndarray:
  ee = ee_flat.reshape(*ee_flat.shape[:-1], len(ee_names), 3)
  name_to_idx = {name: i for i, name in enumerate(ee_names)}
  mirrored = np.empty_like(ee)
  for dst_idx, dst_name in enumerate(ee_names):
    src_name = _swap_left_right(dst_name)
    if src_name not in name_to_idx:
      raise KeyError(f"End-effector '{src_name}' required to mirror '{dst_name}' was not found.")
    mirrored[..., dst_idx, :] = _mirror_polar_vector(ee[..., name_to_idx[src_name], :])
  return mirrored.reshape(ee_flat.shape)


def _mirror_csv_array(data: np.ndarray) -> np.ndarray:
  if data.ndim != 2 or data.shape[1] < CSV_ROOT_DIMS + len(JOINT_NAMES):
    raise ValueError(
      f"Expected CSV array shape (T, >= {CSV_ROOT_DIMS + len(JOINT_NAMES)}), got {data.shape}."
    )
  mirrored = data.copy()
  mirrored[:, 0:3] = _mirror_polar_vector(data[:, 0:3])
  mirrored[:, 3:7] = _mirror_quat_wxyz(data[:, 3:7])
  joint_end = CSV_ROOT_DIMS + len(JOINT_NAMES)
  mirrored[:, CSV_ROOT_DIMS:joint_end] = _mirror_joint_array(
    data[:, CSV_ROOT_DIMS:joint_end], JOINT_NAMES
  )
  return mirrored


def _mirror_windows(windows: np.ndarray, joint_names: tuple[str, ...], ee_names: tuple[str, ...]) -> np.ndarray:
  if windows.ndim != 3 or windows.shape[-1] != TOTAL_FEATURE_DIM:
    raise ValueError(f"Expected windows shape (N, W, {TOTAL_FEATURE_DIM}), got {windows.shape}.")

  mirrored = windows.copy()
  i0 = 0
  root_pos = slice(i0, i0 + 3)
  i0 += 3
  root_rot = slice(i0, i0 + 6)
  i0 += 6
  joints = slice(i0, i0 + len(joint_names))
  i0 += len(joint_names)
  ee_pos = slice(i0, i0 + len(ee_names) * 3)
  i0 += len(ee_names) * 3
  lin_vel = slice(i0, i0 + 3)
  i0 += 3
  ang_vel = slice(i0, i0 + 3)

  mirrored[..., root_pos] = _mirror_polar_vector(windows[..., root_pos])
  mirrored[..., root_rot] = windows[..., root_rot]
  mirrored[..., root_rot.start + 1] *= -1.0
  mirrored[..., root_rot.start + 4] *= -1.0
  mirrored[..., joints] = _mirror_joint_array(windows[..., joints], joint_names)
  mirrored[..., ee_pos] = _mirror_ee_array(windows[..., ee_pos], ee_names)
  mirrored[..., lin_vel] = _mirror_polar_vector(windows[..., lin_vel])
  mirrored[..., ang_vel] = _mirror_axial_vector(windows[..., ang_vel])
  return mirrored


def _output_path(input_path: Path, input_root: Path, output_root: Path, suffix: str = "_mirror") -> Path:
  rel = input_path.relative_to(input_root)
  return output_root / rel.with_name(f"{rel.stem}{suffix}{rel.suffix}")


def _copy_original(input_path: Path, input_root: Path, output_root: Path, overwrite: bool) -> None:
  out_path = output_root / input_path.relative_to(input_root)
  if out_path.exists() and not overwrite:
    return
  out_path.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(input_path, out_path)


def _save_csv(path: Path, data: np.ndarray, overwrite: bool) -> None:
  if path.exists() and not overwrite:
    print(f"[SKIP] exists: {path}")
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(path, data, delimiter=",", fmt="%.8f")


def _process_csv(path: Path, input_root: Path, output_root: Path, overwrite: bool) -> None:
  data = np.loadtxt(path, delimiter=",", dtype=np.float64)
  if data.ndim == 1:
    data = data[None, :]
  mirrored = _mirror_csv_array(data)
  out_path = _output_path(path, input_root, output_root)
  _save_csv(out_path, mirrored, overwrite)
  print(f"[CSV] {path.name} -> {out_path.name} shape={mirrored.shape}")


def _process_npz(path: Path, input_root: Path, output_root: Path, overwrite: bool) -> None:
  out_path = _output_path(path, input_root, output_root)
  if out_path.exists() and not overwrite:
    print(f"[SKIP] exists: {out_path}")
    return

  with np.load(path, allow_pickle=True) as data:
    payload = {key: data[key] for key in data.files}
  if "windows" not in payload:
    raise KeyError(f"{path} does not contain a 'windows' array.")

  joint_names = _as_str_tuple(payload.get("joint_names", np.array(JOINT_NAMES)))
  ee_names = _as_str_tuple(payload.get("ee_body_names", np.array(EE_BODY_NAMES)))
  payload["windows"] = _mirror_windows(payload["windows"], joint_names, ee_names).astype(
    payload["windows"].dtype, copy=False
  )
  payload.setdefault("joint_names", np.array(joint_names))
  payload.setdefault("ee_body_names", np.array(ee_names))

  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out_path, **payload)
  print(f"[NPZ] {path.name} -> {out_path.name} windows={payload['windows'].shape}")


def main(cfg: Cfg) -> None:
  input_root = Path(cfg.input_dir)
  output_root = Path(cfg.output_dir)
  output_root.mkdir(parents=True, exist_ok=True)

  pattern = "**/*" if cfg.recursive else "*"
  files: list[Path] = []
  if cfg.csv:
    files.extend(sorted(input_root.glob(f"{pattern}.csv")))
  if cfg.npz:
    files.extend(sorted(input_root.glob(f"{pattern}.npz")))
  files = sorted(files)
  if not files:
    raise FileNotFoundError(f"No requested motion files found in {input_root}.")

  print(f"Input: {input_root}")
  print(f"Output: {output_root}")
  print(f"Files: {len(files)} | include_original={cfg.include_original}")

  for path in files:
    if cfg.include_original:
      _copy_original(path, input_root, output_root, cfg.overwrite)
    if path.suffix == ".csv":
      _process_csv(path, input_root, output_root, cfg.overwrite)
    elif path.suffix == ".npz":
      _process_npz(path, input_root, output_root, cfg.overwrite)


if __name__ == "__main__":
  main(tyro.cli(Cfg))
