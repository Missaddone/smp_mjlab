#!/usr/bin/env python3
"""Launch and replay selected SMP experiment groups through a shared W&B registry."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WANDB_ENTITY = "robinbird-harbin-institute-of-technology"
DEFAULT_WANDB_PROJECT = "smp"
WANDB_EXPERIMENT_NAMES = {
  "exp11": "smp_exp11_body_velocity_foot_tilt",
  "exp12": "smp_exp12_body_velocity_theme_prior",
  "exp13": "smp_exp13_body_velocity_moving_reward_mix",
  "exp14": "smp_exp14_body_velocity_flatfoot_duty",
}
WandbRegistryRow = dict[str, str]


@dataclass(frozen=True)
class ExperimentSpec:
  max_group: int
  train_script: str
  play_script: str


SPECS = {
  "exp11": ExperimentSpec(
    max_group=26,
    train_script="scripts/run_exp11_body_velocity_foot_tilt.sh",
    play_script="scripts/play_exp11_groups1_26_wandb.sh",
  ),
  "exp12": ExperimentSpec(
    max_group=18,
    train_script="scripts/run_exp12_theme_policy_groups1_18.sh",
    play_script="scripts/play_exp12_groups1_18_wandb.sh",
  ),
  "exp13": ExperimentSpec(
    max_group=21,
    train_script="scripts/run_exp13_groups1_21.sh",
    play_script="scripts/play_exp13_groups1_21_wandb.sh",
  ),
  "exp14": ExperimentSpec(
    max_group=16,
    train_script="scripts/run_exp14_groups1_16.sh",
    play_script="scripts/play_exp14_groups1_16_wandb.sh",
  ),
}


def state_dir() -> Path:
  return ROOT / ".experiment_runs"


def registry_path() -> Path:
  return state_dir() / "runs.jsonl"


def wandb_run_registry_path() -> Path:
  """Return the portable experiment-to-W&B-id mapping shared by all servers."""
  return ROOT / "wandb_run_registry.csv"


def lock_path() -> Path:
  return state_dir() / ".lock"


def utc_now() -> str:
  return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def registry_lock() -> Iterator[None]:
  state_dir().mkdir(parents=True, exist_ok=True)
  with lock_path().open("a+", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_records_unlocked() -> list[dict[str, object]]:
  if not registry_path().exists():
    return []
  records: list[dict[str, object]] = []
  for line in registry_path().read_text(encoding="utf-8").splitlines():
    if line.strip():
      records.append(json.loads(line))
  return records


def read_records() -> list[dict[str, object]]:
  with registry_lock():
    return _read_records_unlocked()


def _write_records_unlocked(records: list[dict[str, object]]) -> None:
  temp_path = registry_path().with_suffix(".tmp")
  contents = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
  temp_path.write_text(contents, encoding="utf-8")
  temp_path.replace(registry_path())


def _read_wandb_run_registry_unlocked() -> list[WandbRegistryRow]:
  path = wandb_run_registry_path()
  if not path.exists():
    return []

  with path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    if reader.fieldnames != ["exp", "group", "run_id"]:
      raise ValueError(
        f"{path.name} must have exactly these columns: exp,group,run_id"
      )
    rows: list[WandbRegistryRow] = []
    for row in reader:
      experiment = (row.get("exp") or "").strip()
      group = (row.get("group") or "").strip()
      run_id = (row.get("run_id") or "").strip()
      if not experiment and not group and not run_id:
        continue
      if experiment not in SPECS:
        raise ValueError(f"Unknown experiment {experiment!r} in {path.name}")
      try:
        validate_group(experiment, int(group))
      except ValueError as error:
        raise ValueError(f"Invalid row in {path.name}: {error}") from error
      rows.append({"exp": experiment, "group": str(int(group)), "run_id": run_id})
  return rows


def _complete_wandb_run_registry_rows(
  rows: list[WandbRegistryRow],
) -> list[WandbRegistryRow]:
  """Add one blank row for every registered experiment group, retaining known ids."""
  by_key: dict[tuple[str, int], WandbRegistryRow] = {}
  for row in rows:
    key = (row["exp"], int(row["group"]))
    if key in by_key:
      raise ValueError(
        f"Duplicate row for {row['exp']} group {row['group']} in "
        f"{wandb_run_registry_path().name}"
      )
    by_key[key] = row

  for experiment, spec in SPECS.items():
    for group in range(1, spec.max_group + 1):
      by_key.setdefault(
        (experiment, group), {"exp": experiment, "group": str(group), "run_id": ""}
      )
  return [
    by_key[key]
    for key in sorted(by_key, key=lambda item: (item[0], item[1]))
  ]


def _write_wandb_run_registry_unlocked(rows: list[WandbRegistryRow]) -> None:
  path = wandb_run_registry_path()
  temp_path = path.with_suffix(".tmp")
  with temp_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
      csv_file,
      fieldnames=["exp", "group", "run_id"],
      lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(_complete_wandb_run_registry_rows(rows))
  temp_path.replace(path)


def ensure_wandb_run_registry() -> list[WandbRegistryRow]:
  """Create missing planned rows so new experiments appear before they are trained."""
  with registry_lock():
    rows = _read_wandb_run_registry_unlocked()
    complete_rows = _complete_wandb_run_registry_rows(rows)
    if complete_rows != rows or not wandb_run_registry_path().exists():
      _write_wandb_run_registry_unlocked(complete_rows)
    return complete_rows


def set_wandb_run_id(experiment: str, group: int, run_id: str) -> None:
  validate_group(experiment, group)
  run_id = run_id.strip()
  if not run_id or "/" in run_id:
    raise ValueError("W&B run_id must be a non-empty bare id, not a full run path")

  with registry_lock():
    rows = _complete_wandb_run_registry_rows(_read_wandb_run_registry_unlocked())
    for row in rows:
      if row["exp"] == experiment and int(row["group"]) == group:
        row["run_id"] = run_id
        _write_wandb_run_registry_unlocked(rows)
        return
  raise AssertionError(f"Missing registry row for {experiment} group {group}")


def registered_wandb_run_id(experiment: str, group: int) -> str | None:
  validate_group(experiment, group)
  for row in ensure_wandb_run_registry():
    if row["exp"] == experiment and int(row["group"]) == group:
      return row["run_id"] or None
  raise AssertionError(f"Missing registry row for {experiment} group {group}")


def find_wandb_run_ids(
  experiment: str, groups: list[int], wandb_entity: str
) -> dict[int, str]:
  """Find newest cloud runs for one experiment with a single W&B API query.

  Training writes an explicit experiment tag through ``WANDB_TAGS``. The group
  segment is matched with boundaries so group 1 cannot resolve to group 10.
  """
  try:
    import wandb
  except ImportError as error:
    raise RuntimeError("W&B SDK is unavailable; run `uv sync` first.") from error

  patterns = {
    group: re.compile(rf"(?:^|_)group0?{group}(?:_|$)")
    for group in groups
  }
  matches_by_group: dict[int, list[object]] = {group: [] for group in groups}

  def add_matches(runs: object, *, require_tag: bool) -> None:
    expected_experiment_name = WANDB_EXPERIMENT_NAMES[experiment]
    for run in runs:
      tags = run.tags or []
      if require_tag:
        if experiment not in tags:
          continue
      else:
        config_values = {str(value) for value in (run.config or {}).values()}
        if expected_experiment_name not in config_values:
          continue
      for group, pattern in patterns.items():
        if pattern.search(run.name or "") and run not in matches_by_group[group]:
          matches_by_group[group].append(run)

  try:
    api = wandb.Api(timeout=30)
    tagged_runs = api.runs(
      f"{wandb_entity}/{DEFAULT_WANDB_PROJECT}",
      filters={"tags": {"$in": [experiment]}},
    )
    add_matches(tagged_runs, require_tag=True)

    # Older direct shell launches predate the launcher and therefore do not
    # carry WANDB_TAGS. Their saved agent experiment-name still identifies them.
    if any(not matches for matches in matches_by_group.values()):
      add_matches(
        api.runs(f"{wandb_entity}/{DEFAULT_WANDB_PROJECT}"),
        require_tag=False,
      )
  except Exception as error:
    raise RuntimeError(f"Could not query W&B for {experiment}: {error}") from error

  run_ids: dict[int, str] = {}
  for group, matches in matches_by_group.items():
    if not matches:
      continue
    matches.sort(key=lambda run: str(run.created_at), reverse=True)
    if len(matches) > 1:
      print(
        f"[launcher] Found {len(matches)} W&B attempts for {experiment} group {group}; "
        f"using newest: {matches[0].id}",
        file=sys.stderr,
      )
    run_ids[group] = str(matches[0].id)
  return run_ids


def find_wandb_run_id(experiment: str, group: int, wandb_entity: str) -> str | None:
  return find_wandb_run_ids(experiment, [group], wandb_entity).get(group)


def resolve_wandb_run_id(experiment: str, group: int, wandb_entity: str) -> str | None:
  """Read the shared CSV first, then fill a blank cell from W&B once."""
  run_id = registered_wandb_run_id(experiment, group)
  if run_id is not None:
    return run_id
  run_id = find_wandb_run_id(experiment, group, wandb_entity)
  if run_id is not None:
    set_wandb_run_id(experiment, group, run_id)
    print(f"[launcher] Filled {experiment} group {group} in {wandb_run_registry_path().name}: {run_id}")
  return run_id


def update_record(record_id: str, **changes: object) -> dict[str, object]:
  with registry_lock():
    records = _read_records_unlocked()
    for record in records:
      if record["record_id"] == record_id:
        record.update(changes)
        record["updated_at"] = utc_now()
        _write_records_unlocked(records)
        return record
  raise ValueError(f"No local experiment record found for {record_id}")


def get_record(record_id: str) -> dict[str, object]:
  for record in read_records():
    if record["record_id"] == record_id:
      return record
  raise ValueError(f"No local experiment record found for {record_id}")


def validate_group(experiment: str, group: int) -> None:
  max_group = SPECS[experiment].max_group
  if not 1 <= group <= max_group:
    raise ValueError(f"{experiment} group must be in 1..{max_group}, got {group}")


def exp13_finetune_parent_group(group: int) -> int | None:
  """Return the Exp13 parent checkpoint group for a fine-tuning group."""
  if 7 <= group <= 10:
    return 4
  if 11 <= group <= 14:
    return 5
  if 15 <= group <= 18:
    return 6
  if group == 19:
    return 4
  if group == 20:
    return 5
  if group == 21:
    return 6
  return None


def finetune_source_group(experiment: str, group: int) -> tuple[str, int] | None:
  """Return the experiment/group providing a fine-tune source checkpoint."""
  if experiment == "exp13":
    parent_group = exp13_finetune_parent_group(group)
    return None if parent_group is None else ("exp13", parent_group)
  if experiment == "exp14":
    return ("exp13", 5)
  return None


def training_source_wandb_path(
  experiment: str,
  group: int,
  wandb_entity: str,
  explicit_source: str | None,
) -> str | None:
  """Resolve implicit parent checkpoints for registered fine-tune groups."""
  if explicit_source is not None:
    return explicit_source
  source = finetune_source_group(experiment, group)
  if source is None:
    return None
  parent_experiment, parent_group = source
  parent_run_id = resolve_wandb_run_id(parent_experiment, parent_group, wandb_entity)
  if parent_run_id is None:
    raise ValueError(
      f"{experiment} group {group} requires {parent_experiment} group {parent_group}, but its W&B "
      "run id is absent from wandb_run_registry.csv and could not be found. "
      "Train/sync the parent first or pass --source-wandb-path explicitly."
    )
  return f"{wandb_entity}/{DEFAULT_WANDB_PROJECT}/{parent_run_id}"


def generate_run_id(existing_records: list[dict[str, object]]) -> str:
  existing_ids = {str(record["wandb_run_id"]) for record in existing_records}
  while True:
    run_id = secrets.token_hex(4)
    if run_id not in existing_ids:
      return run_id


def create_record(
  experiment: str,
  group: int,
  gpu: int,
  wandb_entity: str,
  source_wandb_path: str | None,
  checkpoint_name: str,
) -> dict[str, object]:
  validate_group(experiment, group)
  if gpu < 0:
    raise ValueError(f"GPU must be non-negative, got {gpu}")

  with registry_lock():
    records = _read_records_unlocked()
    run_id = generate_run_id(records)
    created_at = utc_now()
    record_id = f"{created_at.replace(':', '').replace('+00:00', 'Z')}_{experiment}_g{group:02d}_{run_id}"
    record: dict[str, object] = {
      "record_id": record_id,
      "experiment": experiment,
      "group": group,
      "gpu": gpu,
      "status": "queued",
      "created_at": created_at,
      "updated_at": created_at,
      "wandb_entity": wandb_entity,
      "wandb_project": DEFAULT_WANDB_PROJECT,
      "wandb_run_id": run_id,
      "wandb_run_path": f"{wandb_entity}/{DEFAULT_WANDB_PROJECT}/{run_id}",
      "source_wandb_path": source_wandb_path,
      "checkpoint_name": checkpoint_name,
    }
    records.append(record)
    _write_records_unlocked(records)
    return record


def train_command(record: dict[str, object]) -> list[str]:
  experiment = str(record["experiment"])
  group = int(record["group"])
  gpu = int(record["gpu"])
  command = ["bash", SPECS[experiment].train_script, str(group), str(gpu)]

  if experiment == "exp11" or finetune_source_group(experiment, group) is not None:
    source_path = record["source_wandb_path"]
    checkpoint_name = str(record["checkpoint_name"])
    if source_path is None:
      raise ValueError(f"{experiment} group {group} needs a source W&B checkpoint")
    command.extend([str(source_path), checkpoint_name])
  return command


def play_command(
  record: dict[str, object], gpu: int, num_envs: int, video_length: int, viewer: str
) -> list[str]:
  experiment = str(record["experiment"])
  group = int(record["group"])
  return [
    "bash",
    SPECS[experiment].play_script,
    "--gpu",
    str(gpu),
    "--num-envs",
    str(num_envs),
    "--video-length",
    str(video_length),
    "--viewer",
    viewer,
    str(group),
    str(record["wandb_run_path"]),
  ]


def run_record(record_id: str) -> int:
  record = get_record(record_id)
  update_record(record_id, status="running", started_at=utc_now())
  command = train_command(record)
  env = os.environ.copy()
  env.update(
    {
      "WANDB_RUN_ID": str(record["wandb_run_id"]),
      "WANDB_ENTITY": str(record["wandb_entity"]),
      "WANDB_TAGS": str(record["experiment"]),
      "WANDB_RESUME": "never",
      "PYTHONUNBUFFERED": "1",
    }
  )
  print(f"[launcher] W&B run: {record['wandb_run_path']}", flush=True)
  print(f"[launcher] Train: {shlex.join(command)}", flush=True)
  completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
  status = "succeeded" if completed.returncode == 0 else "failed"
  update_record(
    record_id,
    status=status,
    finished_at=utc_now(),
    exit_code=completed.returncode,
  )
  print(f"[launcher] {status}: exit code {completed.returncode}", flush=True)
  return completed.returncode


def print_records(show_all: bool) -> None:
  records = read_records()
  if not show_all:
    latest: dict[tuple[str, int], dict[str, object]] = {}
    for record in records:
      latest[(str(record["experiment"]), int(record["group"]))] = record
    records = sorted(latest.values(), key=lambda record: str(record["created_at"]))

  if not records:
    print("No launcher records yet.")
    return

  print("experiment group status    gpu  wandb run path")
  for record in records:
    print(
      f"{str(record['experiment']):10} {int(record['group']):>5} "
      f"{str(record['status']):9} {int(record['gpu']):>3}  "
      f"{str(record['wandb_run_path']):51}"
    )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Launch or replay Exp11/12/13/14 groups through wandb_run_registry.csv."
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  subparsers.add_parser("list", help="Show supported experiment group ranges.")

  train = subparsers.add_parser("train", help="Start one group and register its W&B path.")
  train.add_argument("experiment", choices=sorted(SPECS))
  train.add_argument("group", type=int)
  train.add_argument("--gpu", required=True, type=int)
  train.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY))
  train.add_argument(
    "--source-wandb-path",
    help="Override the source checkpoint run path for Exp11, Exp13, or Exp14 fine-tuning.",
  )
  train.add_argument("--checkpoint-name", default="model_9999.pt")

  play = subparsers.add_parser("play", help="Replay a group resolved from wandb_run_registry.csv.")
  play.add_argument("experiment", choices=sorted(SPECS))
  play.add_argument("group", type=int)
  play.add_argument("--gpu", required=True, type=int)
  play.add_argument("--num-envs", type=int, default=1)
  play.add_argument("--video-length", type=int, default=1500)
  play.add_argument("--viewer", choices=("auto", "native", "viser"), default="viser")
  play.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY))

  sync = subparsers.add_parser("sync", help="Fill empty W&B ids in the shared registry from W&B.")
  sync.add_argument("experiment", choices=sorted(SPECS), nargs="?")
  sync.add_argument("group", type=int, nargs="?")
  sync.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY))

  status = subparsers.add_parser("status", help="Show the latest registered runs.")
  status.add_argument("--all", action="store_true", help="Include older attempts for each group.")

  return parser


def main() -> int:
  args = build_parser().parse_args()

  if args.command == "list":
    for name, spec in SPECS.items():
      print(f"{name}: groups 1..{spec.max_group}")
    return 0

  if args.command == "train":
    try:
      source_wandb_path = training_source_wandb_path(
        args.experiment,
        args.group,
        args.wandb_entity,
        args.source_wandb_path,
      )
      record = create_record(
        args.experiment,
        args.group,
        args.gpu,
        args.wandb_entity,
        source_wandb_path,
        args.checkpoint_name,
      )
    except ValueError as error:
      print(f"[ERROR] {error}", file=sys.stderr)
      return 2
    try:
      set_wandb_run_id(args.experiment, args.group, str(record["wandb_run_id"]))
    except ValueError as error:
      print(f"[ERROR] {error}", file=sys.stderr)
      return 2
    print(f"[launcher] Registered W&B run: {record['wandb_run_path']}")
    return run_record(str(record["record_id"]))

  if args.command == "play":
    try:
      validate_group(args.experiment, args.group)
    except ValueError as error:
      print(f"[ERROR] {error}", file=sys.stderr)
      return 2
    try:
      run_id = resolve_wandb_run_id(args.experiment, args.group, args.wandb_entity)
    except (RuntimeError, ValueError) as error:
      print(f"[ERROR] {error}", file=sys.stderr)
      return 2
    if run_id is None:
      print(
        f"[ERROR] {wandb_run_registry_path().name} has no W&B run id for "
        f"{args.experiment} group {args.group}, and W&B found no matching run.",
        file=sys.stderr,
      )
      return 2
    record = {
      "experiment": args.experiment,
      "group": args.group,
      "wandb_run_path": f"{args.wandb_entity}/{DEFAULT_WANDB_PROJECT}/{run_id}",
    }
    command = play_command(record, args.gpu, args.num_envs, args.video_length, args.viewer)
    print(f"[launcher] Play: {shlex.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode

  if args.command == "sync":
    if args.group is not None and args.experiment is None:
      print("[ERROR] sync group requires an experiment.", file=sys.stderr)
      return 2
    experiments = [args.experiment] if args.experiment is not None else list(SPECS)
    for experiment in experiments:
      groups = list(
        [args.group] if args.group is not None else range(1, SPECS[experiment].max_group + 1)
      )
      for group in groups:
        try:
          validate_group(experiment, group)
        except (RuntimeError, ValueError) as error:
          print(f"[ERROR] {error}", file=sys.stderr)
          return 2
      existing_ids = {
        group: registered_wandb_run_id(experiment, group)
        for group in groups
      }
      missing_groups = [group for group, run_id in existing_ids.items() if run_id is None]
      try:
        discovered_ids = find_wandb_run_ids(experiment, missing_groups, args.wandb_entity)
      except RuntimeError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
      for group, run_id in discovered_ids.items():
        set_wandb_run_id(experiment, group, run_id)
      for group in groups:
        run_id = existing_ids[group] or discovered_ids.get(group)
        status = run_id if run_id is not None else "<not found>"
        print(f"{experiment} group {group}: {status}")
    return 0

  if args.command == "status":
    print_records(args.all)
    return 0

  raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
  raise SystemExit(main())
