#!/usr/bin/env bash
set -euo pipefail

CSV_CLIPS_DIR="${CSV_CLIPS_DIR:-datasets/csv_clips}"
FORWARD_WALK_PATH="${FORWARD_WALK_PATH:-datasets/csv/forward/g1_walk.csv}"
WORK_ROOT="${WORK_ROOT:-datasets/exp5_prior_sources}"
NPZ_ROOT="${NPZ_ROOT:-datasets/npz/exp5}"
CKPT_ROOT="${CKPT_ROOT:-datasets/pretrain_ckpt}"
LOG_DIR="${LOG_DIR:-logs/pretrain}"
INPUT_FPS="${INPUT_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10000}"
PRETRAIN_SAVE_INTERVAL="${PRETRAIN_SAVE_INTERVAL:-5000}"
PRETRAIN_D_MODEL="${PRETRAIN_D_MODEL:-128}"
PRETRAIN_NUM_LAYERS="${PRETRAIN_NUM_LAYERS:-2}"

require_csv_clips() {
  if [ ! -d "$CSV_CLIPS_DIR" ]; then
    echo "[ERROR] CSV_CLIPS_DIR not found: $CSV_CLIPS_DIR" >&2
    exit 1
  fi
  if [ ! -f scripts/mirror_motion_data.py ]; then
    echo "[ERROR] Mirror tool not found: scripts/mirror_motion_data.py" >&2
    exit 1
  fi
}

reset_group_dir() {
  local group_name="$1"
  local raw_dir="$WORK_ROOT/$group_name/raw"
  local mirrored_dir="$WORK_ROOT/$group_name/mirrored"
  local npz_dir="$NPZ_ROOT/$group_name"

  rm -rf "$WORK_ROOT/$group_name" "$npz_dir"
  mkdir -p "$raw_dir" "$mirrored_dir" "$npz_dir" "$CKPT_ROOT"
}

copy_all_clips() {
  local src_dir="$1"
  find "$CSV_CLIPS_DIR" -maxdepth 1 -type f -name "*.csv" -exec cp -a {} "$src_dir"/ \;
}

copy_clips_without_stop() {
  local src_dir="$1"
  find "$CSV_CLIPS_DIR" -maxdepth 1 -type f -name "*.csv" ! -name "*stop*.csv" \
    -exec cp -a {} "$src_dir"/ \;
}

copy_clips_without_stop2() {
  local src_dir="$1"
  find "$CSV_CLIPS_DIR" -maxdepth 1 -type f -name "*.csv" ! -name "*stop2*.csv" \
    -exec cp -a {} "$src_dir"/ \;
}

copy_dir_back_and_forward_walk() {
  local src_dir="$1"
  find "$CSV_CLIPS_DIR" -maxdepth 1 -type f -name "*.csv" \
    \( -name "low_dir_back*.csv" -o -name "dir_back*.csv" -o -name "*dir-back*.csv" \) \
    -exec cp -a {} "$src_dir"/ \;

  if [ -f "$FORWARD_WALK_PATH" ]; then
    cp -a "$FORWARD_WALK_PATH" "$src_dir"/
  elif [ -d "$FORWARD_WALK_PATH" ]; then
    find "$FORWARD_WALK_PATH" -maxdepth 1 -type f -name "*walk*.csv" \
      -exec cp -a {} "$src_dir"/ \;
  else
    echo "[ERROR] FORWARD_WALK_PATH not found: $FORWARD_WALK_PATH" >&2
    exit 1
  fi
}

check_group_nonempty() {
  local src_dir="$1"
  local count
  count="$(find "$src_dir" -maxdepth 1 -type f -name "*.csv" | wc -l)"
  if [ "$count" -eq 0 ]; then
    echo "[ERROR] No CSV files selected for $src_dir" >&2
    exit 1
  fi
  echo "[INFO] $src_dir csv_count=$count"
}

mirror_group_csvs() {
  local group_name="$1"
  local raw_dir="$WORK_ROOT/$group_name/raw"
  local mirrored_dir="$WORK_ROOT/$group_name/mirrored"

  check_group_nonempty "$raw_dir"
  uv run scripts/mirror_motion_data.py \
    --input-dir "$raw_dir" \
    --output-dir "$mirrored_dir" \
    --include-original \
    --overwrite \
    --csv \
    --no-npz
  check_group_nonempty "$mirrored_dir"
}

convert_and_pretrain() {
  local group_name="$1"
  local mirrored_dir="$WORK_ROOT/$group_name/mirrored"
  local npz_dir="$NPZ_ROOT/$group_name"
  local ckpt_path="$CKPT_ROOT/$group_name.pt"
  local run_root="$LOG_DIR/$group_name"
  local latest_run

  mirror_group_csvs "$group_name"

  uv run scripts/csv_to_npz.py \
    --input-dir "$mirrored_dir" \
    --output-dir "$npz_dir" \
    --input-fps "$INPUT_FPS" \
    --output-fps "$OUTPUT_FPS"

  uv run scripts/pretrain.py \
    --data-dir "$npz_dir" \
    --num-layers "$PRETRAIN_NUM_LAYERS" \
    --no-use-ema \
    --save-interval "$PRETRAIN_SAVE_INTERVAL" \
    --num-epochs "$PRETRAIN_EPOCHS" \
    --train-split 1.0 \
    --d-model "$PRETRAIN_D_MODEL" \
    --name "$group_name" \
    --log-dir "$LOG_DIR" \
    --wandb-project smp

  latest_run="$(find "$run_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [ -z "$latest_run" ] || [ ! -f "$latest_run/pretrained.pt" ]; then
    echo "[ERROR] pretrained.pt not found under $run_root" >&2
    exit 1
  fi

  cp -a "$latest_run/pretrained.pt" "$ckpt_path"
  echo "[DONE] $group_name -> $ckpt_path"
}

prepare_group() {
  local group_name="$1"
  local mode="$2"
  local raw_dir="$WORK_ROOT/$group_name/raw"

  reset_group_dir "$group_name"
  case "$mode" in
    all)
      copy_all_clips "$raw_dir"
      ;;
    no_stop)
      copy_clips_without_stop "$raw_dir"
      ;;
    no_stop2)
      copy_clips_without_stop2 "$raw_dir"
      ;;
    dir_back_forward_walk)
      copy_dir_back_and_forward_walk "$raw_dir"
      ;;
    *)
      echo "[ERROR] Unknown mode: $mode" >&2
      exit 1
      ;;
  esac
  convert_and_pretrain "$group_name"
}

require_csv_clips
prepare_group "exp5_group1_all_csv_clips" "all"
prepare_group "exp5_group2_no_stop" "no_stop"
prepare_group "exp5_group3_no_stop2" "no_stop2"
prepare_group "exp5_group4_dir_back_forward_walk" "dir_back_forward_walk"
