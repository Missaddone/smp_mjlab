#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_exp3_prior_worktrees.sh [--launch]

Creates four git worktrees from master for Experiment 3 prior comparison:
  1. Smp-Forward-G1 baseline prior on GPU 0
  2. Smp-Forward-G1 custom prior on GPU 1
  3. Smp-Steering-G1 baseline prior on GPU 2
  4. Smp-Steering-G1 custom prior on GPU 3

Default behavior only creates worktrees and per-worktree run_train.sh files.
Pass --launch to start all four training jobs in the background.

Environment overrides:
  WORKTREE_ROOT      default: ../smp_mjlab_exp3_worktrees
  NUM_ENVS           default: 4096
  MAX_ITERATIONS     default: 10000
  GPU_IDS            default: "0 1 2 3"
  LOGGER             default: wandb
  WANDB_PROJECT      default: smp
  BASE_BRANCH        default: master
  CUSTOM_PRIOR       default: <source repo>/datasets/pretrain_ckpt/amp_loco_clips2_mirrored_lafan_norm_12000.pt
EOF
}

LAUNCH=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --launch)
      LAUNCH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BASE_BRANCH="${BASE_BRANCH:-master}"
WORKTREE_ROOT="${WORKTREE_ROOT:-$(dirname "$REPO_ROOT")/smp_mjlab_exp3_worktrees}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-smp}"

SOURCE_PRETRAIN_DIR="$REPO_ROOT/datasets/pretrain_ckpt"

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$REPO_ROOT/$1" ;;
  esac
}

CUSTOM_PRIOR="$(abs_path "${CUSTOM_PRIOR:-datasets/pretrain_ckpt/amp_loco_clips2_mirrored_lafan_norm_12000.pt}")"
FORWARD_BASELINE_PRIOR="$SOURCE_PRETRAIN_DIR/pretrained_loco.pt"
STEERING_BASELINE_PRIOR="$SOURCE_PRETRAIN_DIR/pretrained_lafan_run.pt"

GPU_IDS="${GPU_IDS//,/ }"
read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ID_ARRAY[@]}" -lt 4 ]; then
  echo "[ERROR] GPU_IDS must contain four GPU ids, e.g. GPU_IDS=\"0 1 2 3\"." >&2
  exit 1
fi

ensure_master_base() {
  if git show-ref --verify --quiet "refs/heads/${BASE_BRANCH}"; then
    return
  fi

  if [ "$BASE_BRANCH" = "master" ] && git show-ref --verify --quiet refs/remotes/origin/master; then
    echo "[INFO] Local master is missing; creating it from origin/master."
    git branch --track master origin/master
    return
  fi

  echo "[ERROR] Required base branch '${BASE_BRANCH}' was not found." >&2
  echo "        Create/fetch it first. Refusing to use my-dev or tyj_test." >&2
  exit 1
}

ensure_clean_master_ref() {
  local current
  current="$(git branch --show-current || true)"
  if [ "$current" != "$BASE_BRANCH" ]; then
    echo "[INFO] Current branch is '${current:-detached}', but worktrees will be based on '${BASE_BRANCH}'."
  fi
}

ensure_worktree() {
  local branch="$1"
  local path="$2"

  if [ -e "$path/.git" ]; then
    echo "[INFO] Worktree already exists: $path"
  elif git show-ref --verify --quiet "refs/heads/${branch}"; then
    echo "[INFO] Adding existing branch '${branch}' at $path"
    git worktree add "$path" "$branch"
  else
    echo "[INFO] Creating worktree branch '${branch}' from '${BASE_BRANCH}' at $path"
    git worktree add -b "$branch" "$path" "$BASE_BRANCH"
  fi

  if [ -d "$REPO_ROOT/.venv" ] && [ ! -e "$path/.venv" ]; then
    ln -s "$REPO_ROOT/.venv" "$path/.venv"
  fi
}

write_runner() {
  local path="$1"
  local task_id="$2"
  local run_name="$3"
  local prior_path="$4"
  local gpu_id="$5"
  local runner="$path/run_train.sh"

  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "\$(dirname "\$0")"

PRIOR_PATH="$prior_path"
if [ ! -f "\$PRIOR_PATH" ]; then
  echo "[ERROR] Prior checkpoint not found: \$PRIOR_PATH" >&2
  echo "        Put the checkpoint in this worktree, or copy/symlink datasets/pretrain_ckpt from the source repo." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "[ERROR] .venv/bin/python is missing. The setup script normally symlinks the source .venv." >&2
  exit 1
fi

export MUJOCO_GL=egl
export PYTHONPATH="\$PWD/src\${PYTHONPATH:+:\$PYTHONPATH}"

exec ./.venv/bin/python scripts/train.py "$task_id" \\
  --env.scene.num-envs "$NUM_ENVS" \\
  --agent.max-iterations "$MAX_ITERATIONS" \\
  --agent.run-name "$run_name" \\
  --agent.logger "$LOGGER" \\
  --agent.wandb-project "$WANDB_PROJECT" \\
  --gpu-ids "$gpu_id" \\
  --env.events.init-smp-state.params.ckpt-path "\$PRIOR_PATH"
EOF

  chmod +x "$runner"

  cat > "$path/EXP3_TASK.txt" <<EOF
task_id=$task_id
run_name=$run_name
prior_path=$prior_path
num_envs=$NUM_ENVS
max_iterations=$MAX_ITERATIONS
gpu_ids=$GPU_IDS
assigned_gpu_id=$gpu_id
logger=$LOGGER
wandb_project=$WANDB_PROJECT
EOF
}

setup_task() {
  local slug="$1"
  local task_id="$2"
  local variant="$3"
  local prior_path="$4"
  local gpu_id="$5"

  local branch="exp3-${slug}-${variant}"
  local path="$WORKTREE_ROOT/$branch"
  local run_name="exp3_${slug}_${variant}"

  ensure_worktree "$branch" "$path"
  write_runner "$path" "$task_id" "$run_name" "$prior_path" "$gpu_id"

  echo "[OK] $branch"
  echo "     worktree: $path"
  echo "     gpu:      $gpu_id"
  echo "     run:      cd $path && ./run_train.sh"

  if [ "$LAUNCH" -eq 1 ]; then
    echo "[INFO] Launching $branch in background."
    (cd "$path" && nohup ./run_train.sh > train.log 2>&1 & echo $! > train.pid)
    echo "     pid:      $(cat "$path/train.pid")"
    echo "     log:      $path/train.log"
  fi
}

ensure_master_base
ensure_clean_master_ref
mkdir -p "$WORKTREE_ROOT"

echo "[INFO] Worktree root: $WORKTREE_ROOT"
echo "[INFO] Experiment 3 compares only prior checkpoints."
echo "[INFO] Custom prior path: $CUSTOM_PRIOR"
echo "[INFO] GPU assignment: forward baseline=${GPU_ID_ARRAY[0]}, forward custom=${GPU_ID_ARRAY[1]}, steering baseline=${GPU_ID_ARRAY[2]}, steering custom=${GPU_ID_ARRAY[3]}"

setup_task "forward" "Smp-Forward-G1" "baseline" "$FORWARD_BASELINE_PRIOR" "${GPU_ID_ARRAY[0]}"
setup_task "forward" "Smp-Forward-G1" "custom" "$CUSTOM_PRIOR" "${GPU_ID_ARRAY[1]}"
setup_task "steering" "Smp-Steering-G1" "baseline" "$STEERING_BASELINE_PRIOR" "${GPU_ID_ARRAY[2]}"
setup_task "steering" "Smp-Steering-G1" "custom" "$CUSTOM_PRIOR" "${GPU_ID_ARRAY[3]}"

cat <<EOF

[DONE] Experiment 3 worktrees are ready.

Before training custom-prior tasks, ensure this file exists:
  $CUSTOM_PRIOR
EOF
