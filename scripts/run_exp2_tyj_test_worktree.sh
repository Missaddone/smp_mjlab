#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-tyj-test}"
SOURCE_REPO="${SOURCE_REPO:-}"
WORKTREE_DIR="${WORKTREE_DIR:-}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
NUM_ENVS="${NUM_ENVS:-4096}"
WANDB_PROJECT="${WANDB_PROJECT:-smp}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-smp_exp2_body_velocity_static_switch}"
PRIOR_REL="${PRIOR_REL:-datasets/pretrain_ckpt/amp_loco_clips2_mirrored_lafan_norm_12000.pt}"
PULL="${PULL:-0}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_exp2_tyj_test_worktree.sh [--dry-run]

Environment overrides:
  BRANCH=tyj-test
  SOURCE_REPO=/path/to/current/repo
  WORKTREE_DIR=/path/to/new/worktree
  GPU_IDS="0 1"
  MAX_ITERATIONS=10000
  NUM_ENVS=4096
  WANDB_PROJECT=smp
  EXPERIMENT_NAME=smp_exp2_body_velocity_static_switch
  PRIOR_REL=datasets/pretrain_ckpt/amp_loco_clips2_mirrored_lafan_norm_12000.pt
  PULL=1  # optionally git pull --ff-only inside an existing worktree
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$SOURCE_REPO" ]]; then
  SOURCE_REPO="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
fi
SOURCE_REPO="$(cd "$SOURCE_REPO" && pwd)"

if [[ -z "$WORKTREE_DIR" ]]; then
  repo_parent="$(cd "$SOURCE_REPO/.." && pwd)"
  repo_name="$(basename "$SOURCE_REPO")"
  branch_slug="${BRANCH//\//_}"
  WORKTREE_DIR="$repo_parent/${repo_name}_${branch_slug}"
fi

read -r -a gpu_id_array <<< "$GPU_IDS"
if [[ "${#gpu_id_array[@]}" -lt 2 ]]; then
  echo "[ERROR] GPU_IDS must contain two GPU ids, for example: GPU_IDS=\"0 1\"" >&2
  exit 2
fi

ensure_branch_exists() {
  if git -C "$SOURCE_REPO" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    return
  fi
  echo "[INFO] Local branch '$BRANCH' not found. Fetching origin/$BRANCH..."
  git -C "$SOURCE_REPO" fetch origin "$BRANCH:$BRANCH"
}

ensure_worktree() {
  ensure_branch_exists
  if [[ -d "$WORKTREE_DIR/.git" ]] || git -C "$WORKTREE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    current_branch="$(git -C "$WORKTREE_DIR" branch --show-current)"
    if [[ "$current_branch" != "$BRANCH" ]]; then
      echo "[ERROR] Existing WORKTREE_DIR is on '$current_branch', expected '$BRANCH': $WORKTREE_DIR" >&2
      exit 1
    fi
    echo "[INFO] Reusing existing worktree: $WORKTREE_DIR"
  else
    echo "[INFO] Creating worktree: $WORKTREE_DIR ($BRANCH)"
    git -C "$SOURCE_REPO" worktree add "$WORKTREE_DIR" "$BRANCH"
  fi

  if [[ "$PULL" == "1" ]]; then
    echo "[INFO] Pulling latest '$BRANCH' in worktree..."
    git -C "$WORKTREE_DIR" pull --ff-only
  fi
}

ensure_prior() {
  worktree_prior="$WORKTREE_DIR/$PRIOR_REL"
  source_prior="$SOURCE_REPO/$PRIOR_REL"

  if [[ -e "$worktree_prior" ]]; then
    return
  fi

  mkdir -p "$(dirname "$worktree_prior")"
  if [[ -e "$source_prior" ]]; then
    echo "[INFO] Linking prior into worktree: $PRIOR_REL"
    ln -s "$(realpath "$source_prior")" "$worktree_prior"
    return
  fi

  echo "[ERROR] Prior checkpoint not found in worktree or source repo:" >&2
  echo "  $worktree_prior" >&2
  echo "  $source_prior" >&2
  exit 1
}

print_command() {
  local gpu_id="$1"
  local task_name="$2"
  local run_name="$3"
  cat <<EOF
cd "$WORKTREE_DIR"
CUDA_VISIBLE_DEVICES=$gpu_id uv run scripts/train.py $task_name \\
  --agent.max-iterations=$MAX_ITERATIONS \\
  --env.scene.num-envs=$NUM_ENVS \\
  --env.events.init-smp-state.params.ckpt-path=$PRIOR_REL \\
  --agent.logger=wandb \\
  --agent.wandb-project=$WANDB_PROJECT \\
  --agent.experiment-name=$EXPERIMENT_NAME \\
  --agent.run-name=$run_name
EOF
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] Worktree: $WORKTREE_DIR"
  echo
  print_command "${gpu_id_array[0]}" "Smp-BodyVelocity-Exp2-G1" "base_linear_yaw_static_cmd"
  echo
  print_command "${gpu_id_array[1]}" "Smp-BodyVelocity-StaticExp2-G1" "static_cmd_root_foot_vel"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is not available in PATH." >&2
  exit 127
fi

ensure_worktree
ensure_prior

cd "$WORKTREE_DIR"

CUDA_VISIBLE_DEVICES="${gpu_id_array[0]}" uv run scripts/train.py Smp-BodyVelocity-Exp2-G1 \
  --agent.max-iterations="$MAX_ITERATIONS" \
  --env.scene.num-envs="$NUM_ENVS" \
  --env.events.init-smp-state.params.ckpt-path="$PRIOR_REL" \
  --agent.logger=wandb \
  --agent.wandb-project="$WANDB_PROJECT" \
  --agent.experiment-name="$EXPERIMENT_NAME" \
  --agent.run-name=base_linear_yaw_static_cmd

CUDA_VISIBLE_DEVICES="${gpu_id_array[1]}" uv run scripts/train.py Smp-BodyVelocity-StaticExp2-G1 \
  --agent.max-iterations="$MAX_ITERATIONS" \
  --env.scene.num-envs="$NUM_ENVS" \
  --env.events.init-smp-state.params.ckpt-path="$PRIOR_REL" \
  --agent.logger=wandb \
  --agent.wandb-project="$WANDB_PROJECT" \
  --agent.experiment-name="$EXPERIMENT_NAME" \
  --agent.run-name=static_cmd_root_foot_vel
