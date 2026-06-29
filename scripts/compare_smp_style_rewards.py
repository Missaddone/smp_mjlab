"""Compare two diffusion priors on saved SMP rollout windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from smp.rl.utils import load_denoiser


def _parse_timesteps(value: str) -> tuple[int, ...]:
  return tuple(int(x.strip()) for x in value.split(",") if x.strip())


@torch.inference_mode()
def _score_windows(
  windows: torch.Tensor,
  ckpt_path: str,
  fixed_timesteps: tuple[int, ...],
  ws: float,
  num_noise_samples: int,
  batch_size: int,
  device: str,
  normalize_by_model_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
  model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(ckpt_path, device)
  if windows.shape[1] != window_size or windows.shape[2] != feature_dim:
    raise ValueError(
      f"{ckpt_path}: expected windows (*, {window_size}, {feature_dim}), "
      f"got {tuple(windows.shape)}"
    )
  for t in fixed_timesteps:
    if not 0 <= t < scheduler.num_timesteps:
      raise ValueError(f"{ckpt_path}: timestep {t} out of range [0, {scheduler.num_timesteps})")

  x0 = 2.0 * (windows.to(device) - q_low) / (q_high - q_low + 1.0e-8) - 1.0
  raw_err_chunks: list[torch.Tensor] = []

  for start in range(0, x0.shape[0], batch_size):
    x0_b = x0[start : start + batch_size]
    err = torch.zeros(x0_b.shape[0], device=device)
    for _ in range(num_noise_samples):
      for t_scalar in fixed_timesteps:
        t = torch.full((x0_b.shape[0],), t_scalar, dtype=torch.long, device=device)
        noise = torch.randn_like(x0_b)
        x_t = scheduler.add_noise(x0_b, noise, t)
        eps_hat = model(x_t, t)
        err += ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
    err /= float(num_noise_samples * len(fixed_timesteps))
    raw_err_chunks.append(err.cpu())

  raw_err = torch.cat(raw_err_chunks, dim=0)
  if normalize_by_model_mean:
    score_err = raw_err / raw_err.mean().clamp(min=1.0e-8)
  else:
    score_err = raw_err
  reward = torch.exp(-ws * score_err)
  return reward.numpy(), raw_err.numpy()


def _group_curve(step: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  unique_steps = np.unique(step)
  mean = np.zeros_like(unique_steps, dtype=np.float64)
  std = np.zeros_like(unique_steps, dtype=np.float64)
  for i, s in enumerate(unique_steps):
    v = values[step == s]
    mean[i] = float(v.mean())
    std[i] = float(v.std())
  return unique_steps, mean, std


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--npz", required=True, help="Rollout windows .npz")
  parser.add_argument("--ckpt-a", required=True)
  parser.add_argument("--ckpt-b", required=True)
  parser.add_argument("--label-a", default="prior_a")
  parser.add_argument("--label-b", default="prior_b")
  parser.add_argument("--output", required=True, help="Output plot path, e.g. outputs/style.png")
  parser.add_argument("--csv", default=None, help="Optional per-step CSV path")
  parser.add_argument("--scores-npz", default=None, help="Optional raw per-window scores .npz path")
  parser.add_argument("--device", default=None)
  parser.add_argument("--fixed-timesteps", default="8,15,22")
  parser.add_argument("--ws", type=float, default=6.0)
  parser.add_argument("--num-noise-samples", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument(
    "--normalize-by-model-mean",
    action="store_true",
    help="Plot exp(-ws * raw_mse / model_mean_mse) instead of exp(-ws * raw_mse).",
  )
  args = parser.parse_args()

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  fixed_timesteps = _parse_timesteps(args.fixed_timesteps)
  data = np.load(args.npz, allow_pickle=False)
  windows_np = data["windows"].astype(np.float32, copy=False)
  step = data["step"] if "step" in data else np.arange(windows_np.shape[0], dtype=np.int64)
  windows = torch.from_numpy(windows_np)

  reward_a, raw_err_a = _score_windows(
    windows,
    args.ckpt_a,
    fixed_timesteps,
    args.ws,
    args.num_noise_samples,
    args.batch_size,
    device,
    args.normalize_by_model_mean,
  )
  reward_b, raw_err_b = _score_windows(
    windows,
    args.ckpt_b,
    fixed_timesteps,
    args.ws,
    args.num_noise_samples,
    args.batch_size,
    device,
    args.normalize_by_model_mean,
  )

  x, mean_a, std_a = _group_curve(step, reward_a)
  _, mean_b, std_b = _group_curve(step, reward_b)

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  plt.figure(figsize=(10, 5))
  plt.plot(x, mean_a, label=args.label_a, linewidth=2)
  plt.fill_between(x, mean_a - std_a, mean_a + std_a, alpha=0.2)
  plt.plot(x, mean_b, label=args.label_b, linewidth=2)
  plt.fill_between(x, mean_b - std_b, mean_b + std_b, alpha=0.2)
  plt.xlabel("rollout step")
  plt.ylabel("style reward")
  plt.title("SMP diffusion style reward comparison")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(output, dpi=160)

  csv_path = Path(args.csv) if args.csv else output.with_suffix(".csv")
  with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", f"{args.label_a}_mean", f"{args.label_a}_std", f"{args.label_b}_mean", f"{args.label_b}_std"])
    for row in zip(x, mean_a, std_a, mean_b, std_b, strict=True):
      writer.writerow(row)

  scores_path = Path(args.scores_npz) if args.scores_npz else output.with_suffix(".scores.npz")
  np.savez_compressed(
    scores_path,
    step=step,
    reward_a=reward_a,
    reward_b=reward_b,
    raw_err_a=raw_err_a,
    raw_err_b=raw_err_b,
    label_a=args.label_a,
    label_b=args.label_b,
    fixed_timesteps=np.asarray(fixed_timesteps, dtype=np.int64),
    ws=np.asarray(args.ws, dtype=np.float32),
    normalize_by_model_mean=np.asarray(args.normalize_by_model_mean),
  )

  print(f"[INFO] Saved plot: {output}")
  print(f"[INFO] Saved curve CSV: {csv_path}")
  print(f"[INFO] Saved per-window scores: {scores_path}")
  print(f"[INFO] {args.label_a}: mean={reward_a.mean():.6f}, raw_mse={raw_err_a.mean():.6f}")
  print(f"[INFO] {args.label_b}: mean={reward_b.mean():.6f}, raw_mse={raw_err_b.mean():.6f}")


if __name__ == "__main__":
  main()
