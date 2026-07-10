#!/usr/bin/env bash
set -euo pipefail

FORWARD_CSV_DIR="${FORWARD_CSV_DIR:-datasets/csv/forward}"
WORK_ROOT="${WORK_ROOT:-datasets/exp6_prior_sources}"
NPZ_ROOT="${NPZ_ROOT:-datasets/npz/exp6}"
CKPT_ROOT="${CKPT_ROOT:-datasets/pretrain_ckpt}"
LOG_DIR="${LOG_DIR:-logs/pretrain}"
PRIOR_NAME="${PRIOR_NAME:-forward_stop_all}"
INPUT_FPS="${INPUT_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10000}"
PRETRAIN_SAVE_INTERVAL="${PRETRAIN_SAVE_INTERVAL:-5000}"
PRETRAIN_D_MODEL="${PRETRAIN_D_MODEL:-128}"
PRETRAIN_NUM_LAYERS="${PRETRAIN_NUM_LAYERS:-2}"

RAW_DIR="$WORK_ROOT/$PRIOR_NAME/raw"
STOP_MIRROR_DIR="$WORK_ROOT/$PRIOR_NAME/stop_static_mirror"
NPZ_DIR="$NPZ_ROOT/$PRIOR_NAME"
CKPT_PATH="$CKPT_ROOT/$PRIOR_NAME.pt"

require_inputs() {
  if [ ! -d "$FORWARD_CSV_DIR" ]; then
    echo "[ERROR] FORWARD_CSV_DIR not found: $FORWARD_CSV_DIR" >&2
    exit 1
  fi
  if [ ! -f "$FORWARD_CSV_DIR/stop_static.csv" ]; then
    echo "[ERROR] stop_static.csv not found: $FORWARD_CSV_DIR/stop_static.csv" >&2
    exit 1
  fi
  if [ ! -f scripts/mirror_motion_data.py ]; then
    echo "[ERROR] Mirror tool not found: scripts/mirror_motion_data.py" >&2
    exit 1
  fi
}

reset_dirs() {
  rm -rf "$WORK_ROOT/$PRIOR_NAME" "$NPZ_DIR"
  mkdir -p "$RAW_DIR" "$STOP_MIRROR_DIR" "$NPZ_DIR" "$CKPT_ROOT"
}

copy_existing_forward_actions() {
  cp -a "$FORWARD_CSV_DIR"/g1_walk.csv "$RAW_DIR"/
  cp -a "$FORWARD_CSV_DIR"/g1_walk_mirror.csv "$RAW_DIR"/
  cp -a "$FORWARD_CSV_DIR"/g1_jog.csv "$RAW_DIR"/
  cp -a "$FORWARD_CSV_DIR"/g1_jog_mirror.csv "$RAW_DIR"/
  cp -a "$FORWARD_CSV_DIR"/g1_run.csv "$RAW_DIR"/
  cp -a "$FORWARD_CSV_DIR"/g1_run_mirror.csv "$RAW_DIR"/
}

mirror_stop_static() {
  local stop_only_dir="$WORK_ROOT/$PRIOR_NAME/stop_static_raw"
  mkdir -p "$stop_only_dir"
  cp -a "$FORWARD_CSV_DIR/stop_static.csv" "$stop_only_dir"/

  uv run scripts/mirror_motion_data.py \
    --input-dir "$stop_only_dir" \
    --output-dir "$STOP_MIRROR_DIR" \
    --include-original \
    --overwrite \
    --csv \
    --no-npz

  cp -a "$STOP_MIRROR_DIR"/stop_static.csv "$RAW_DIR"/
  cp -a "$STOP_MIRROR_DIR"/stop_static_mirror.csv "$RAW_DIR"/
}

check_action_count() {
  local count
  count="$(find "$RAW_DIR" -maxdepth 1 -type f -name "*.csv" | wc -l)"
  echo "[INFO] $RAW_DIR csv_count=$count"
  if [ "$count" -ne 8 ]; then
    echo "[ERROR] Expected exactly 8 CSV actions for $PRIOR_NAME, got $count." >&2
    find "$RAW_DIR" -maxdepth 1 -type f -name "*.csv" -printf "  %f\n" | sort >&2
    exit 1
  fi
}

convert_and_pretrain() {
  local run_root="$LOG_DIR/$PRIOR_NAME"
  local latest_run

  uv run scripts/csv_to_npz.py \
    --input-dir "$RAW_DIR" \
    --output-dir "$NPZ_DIR" \
    --input-fps "$INPUT_FPS" \
    --output-fps "$OUTPUT_FPS"

  uv run scripts/pretrain.py \
    --data-dir "$NPZ_DIR" \
    --num-layers "$PRETRAIN_NUM_LAYERS" \
    --no-use-ema \
    --save-interval "$PRETRAIN_SAVE_INTERVAL" \
    --num-epochs "$PRETRAIN_EPOCHS" \
    --train-split 1.0 \
    --d-model "$PRETRAIN_D_MODEL" \
    --name "$PRIOR_NAME" \
    --log-dir "$LOG_DIR" \
    --wandb-project smp

  latest_run="$(find "$run_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [ -z "$latest_run" ] || [ ! -f "$latest_run/pretrained.pt" ]; then
    echo "[ERROR] pretrained.pt not found under $run_root" >&2
    exit 1
  fi

  cp -a "$latest_run/pretrained.pt" "$CKPT_PATH"
  echo "[DONE] $PRIOR_NAME -> $CKPT_PATH"
}

require_inputs
reset_dirs
copy_existing_forward_actions
mirror_stop_static
check_action_count
convert_and_pretrain
