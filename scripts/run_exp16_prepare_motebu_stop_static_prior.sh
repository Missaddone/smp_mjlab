#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp16_prepare_motebu_stop_static_prior.sh [--gpu N]

Build the Exp16 prior from exactly four files:
  motebu.csv, stop_static.csv, and one generated mirror of each.

Output:
  datasets/pretrain_ckpt/exp16_motebu_stop_static.pt
EOF
}

GPU=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

MOTEBU_CSV="${MOTEBU_CSV:-datasets/csv/theme/motebu.csv}"
STOP_STATIC_CSV="${STOP_STATIC_CSV:-datasets/csv/theme/stop_static.csv}"
WORK_ROOT="${WORK_ROOT:-datasets/exp16_prior_sources}"
NPZ_ROOT="${NPZ_ROOT:-datasets/npz/exp16}"
CKPT_ROOT="${CKPT_ROOT:-datasets/pretrain_ckpt}"
LOG_DIR="${LOG_DIR:-logs/pretrain}"
PRIOR_NAME="${PRIOR_NAME:-exp16_motebu_stop_static}"
INPUT_FPS="${INPUT_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10000}"
PRETRAIN_SAVE_INTERVAL="${PRETRAIN_SAVE_INTERVAL:-5000}"
PRETRAIN_D_MODEL="${PRETRAIN_D_MODEL:-128}"
PRETRAIN_NUM_LAYERS="${PRETRAIN_NUM_LAYERS:-2}"

RAW_DIR="$WORK_ROOT/$PRIOR_NAME/raw"
NPZ_DIR="$NPZ_ROOT/$PRIOR_NAME"
RUN_ROOT="$LOG_DIR/$PRIOR_NAME"
CKPT_PATH="$CKPT_ROOT/$PRIOR_NAME.pt"

run_uv() {
  if [ -n "$GPU" ]; then
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" uv run "$@"
  else
    WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" uv run "$@"
  fi
}

require_inputs() {
  for path in "$MOTEBU_CSV" "$STOP_STATIC_CSV" scripts/mirror_motion_data.py; do
    if [ ! -f "$path" ]; then
      echo "[ERROR] Required file not found: $path" >&2
      exit 1
    fi
  done
}

reset_dirs() {
  rm -rf "$WORK_ROOT/$PRIOR_NAME" "$NPZ_DIR" "$RUN_ROOT"
  mkdir -p "$RAW_DIR" "$NPZ_DIR" "$CKPT_ROOT" "$RUN_ROOT"
}

mirror_one_source() {
  local source_csv="$1"
  local source_name="$2"
  local expected_mirror="$3"
  local mirror_root="$WORK_ROOT/$PRIOR_NAME/mirror_${source_name%.csv}"
  local input_dir="$mirror_root/input"
  local output_dir="$mirror_root/output"
  mkdir -p "$input_dir" "$output_dir"
  cp -a "$source_csv" "$input_dir/$source_name"
  run_uv scripts/mirror_motion_data.py \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --csv \
    --no-npz
  if [ ! -f "$output_dir/$expected_mirror" ]; then
    echo "[ERROR] Mirror output missing: $output_dir/$expected_mirror" >&2
    exit 1
  fi
  cp -a "$output_dir/$expected_mirror" "$RAW_DIR/$expected_mirror"
}

stage_exact_sources() {
  cp -a "$MOTEBU_CSV" "$RAW_DIR/motebu.csv"
  cp -a "$STOP_STATIC_CSV" "$RAW_DIR/stop_static.csv"
  mirror_one_source "$MOTEBU_CSV" "motebu.csv" "motebu_mirror.csv"
  mirror_one_source "$STOP_STATIC_CSV" "stop_static.csv" "stop_static_mirror.csv"

  local expected_raw_files=(
    motebu.csv
    motebu_mirror.csv
    stop_static.csv
    stop_static_mirror.csv
  )
  local actual_raw_files=()
  mapfile -t actual_raw_files < <(find "$RAW_DIR" -maxdepth 1 -type f -name '*.csv' -printf '%f\n' | sort)
  if [ "${#actual_raw_files[@]}" -ne "${#expected_raw_files[@]}" ] || \
     [ "$(printf '%s\n' "${actual_raw_files[@]}")" != "$(printf '%s\n' "${expected_raw_files[@]}")" ]; then
    echo "[ERROR] Raw staging must contain exactly: ${expected_raw_files[*]}" >&2
    printf '[ERROR] Found: %s\n' "${actual_raw_files[*]}" >&2
    exit 1
  fi
}

convert_and_pretrain() {
  local latest_run
  run_uv scripts/csv_to_npz.py \
    --input-dir "$RAW_DIR" \
    --output-dir "$NPZ_DIR" \
    --input-fps "$INPUT_FPS" \
    --output-fps "$OUTPUT_FPS"
  run_uv scripts/pretrain.py \
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
  latest_run="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [ -z "$latest_run" ] || [ ! -f "$latest_run/pretrained.pt" ]; then
    echo "[ERROR] pretrained.pt not found under $RUN_ROOT" >&2
    exit 1
  fi
  cp -a "$latest_run/pretrained.pt" "$CKPT_PATH"
  echo "[DONE] $PRIOR_NAME -> $CKPT_PATH"
}

require_inputs
reset_dirs
stage_exact_sources
echo "[INFO] Raw files: $(find "$RAW_DIR" -maxdepth 1 -type f -name '*.csv' -printf '%f ' | sort)"
convert_and_pretrain
