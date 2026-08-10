#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_new_dev_exp13_prepare_prior.sh [--gpu N]

Rebuild the Exp13 G4--G6 SMP prior from the original Exp10 source set:
  datasets/csv/loco/*.csv + datasets/csv/forward/stop_static.csv

Every source, including stop_static.csv, must have exactly one mirrored pair.

Output:
  datasets/pretrain_ckpt/new_dev_exp13_loco_stop_static_mirrored.pt
EOF
}

GPU=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

LOCO_CSV_DIR="${LOCO_CSV_DIR:-datasets/csv/loco}"
STOP_STATIC_CSV="${STOP_STATIC_CSV:-datasets/csv/forward/stop_static.csv}"
WORK_ROOT="${WORK_ROOT:-datasets/new_dev_exp13_prior_sources}"
NPZ_ROOT="${NPZ_ROOT:-datasets/npz/new_dev_exp13}"
CKPT_ROOT="${CKPT_ROOT:-datasets/pretrain_ckpt}"
LOG_DIR="${LOG_DIR:-logs/pretrain}"
PRIOR_NAME="${PRIOR_NAME:-new_dev_exp13_loco_stop_static_mirrored}"
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
  [ -d "$LOCO_CSV_DIR" ] || { echo "[ERROR] Missing $LOCO_CSV_DIR" >&2; exit 1; }
  [ -f "$STOP_STATIC_CSV" ] || { echo "[ERROR] Missing $STOP_STATIC_CSV" >&2; exit 1; }
  [ -f scripts/mirror_motion_data.py ] || { echo "[ERROR] Missing mirror tool" >&2; exit 1; }
}

reset_dirs() {
  rm -rf "$WORK_ROOT/$PRIOR_NAME" "$NPZ_DIR" "$RUN_ROOT"
  mkdir -p "$RAW_DIR" "$NPZ_DIR" "$CKPT_ROOT" "$RUN_ROOT"
}

stage_sources() {
  find "$LOCO_CSV_DIR" -maxdepth 1 -type f -name '*.csv' -exec cp -a {} "$RAW_DIR"/ \;
  cp -a "$STOP_STATIC_CSV" "$RAW_DIR"/stop_static.csv
  [ "$(find "$RAW_DIR" -maxdepth 1 -type f -name '*.csv' | wc -l)" -gt 0 ] || {
    echo "[ERROR] No source CSVs staged" >&2; exit 1;
  }
}

ensure_missing_mirrors() {
  local mirror_root="$WORK_ROOT/$PRIOR_NAME/mirror_tmp"
  rm -rf "$mirror_root"
  mkdir -p "$mirror_root"
  local source base mirror_name input_dir output_dir
  for source in "$RAW_DIR"/*.csv; do
    base="$(basename "$source")"
    case "$base" in *_mirror.csv) continue ;; esac
    mirror_name="${base%.csv}_mirror.csv"
    [ -f "$RAW_DIR/$mirror_name" ] && continue
    input_dir="$mirror_root/${base%.csv}/input"
    output_dir="$mirror_root/${base%.csv}/output"
    mkdir -p "$input_dir" "$output_dir"
    cp -a "$source" "$input_dir/$base"
    run_uv scripts/mirror_motion_data.py --input-dir "$input_dir" --output-dir "$output_dir" --csv --no-npz
    [ -f "$output_dir/$mirror_name" ] || {
      echo "[ERROR] Mirror missing for $base" >&2; exit 1;
    }
    cp -a "$output_dir/$mirror_name" "$RAW_DIR/$mirror_name"
  done
}

verify_complete_mirroring() {
  local source base mirror
  for source in "$RAW_DIR"/*.csv; do
    base="$(basename "$source")"
    case "$base" in
      *_mirror.csv)
        mirror="${base%_mirror.csv}.csv"
        [ -f "$RAW_DIR/$mirror" ] || { echo "[ERROR] Orphan mirror: $base" >&2; exit 1; }
        ;;
      *)
        mirror="${base%.csv}_mirror.csv"
        [ -f "$RAW_DIR/$mirror" ] || { echo "[ERROR] Missing mirror: $mirror" >&2; exit 1; }
        ;;
    esac
  done
  [ -f "$RAW_DIR/stop_static.csv" ] && [ -f "$RAW_DIR/stop_static_mirror.csv" ] || {
    echo "[ERROR] stop_static.csv must be mirrored" >&2; exit 1;
  }
}

convert_and_pretrain() {
  run_uv scripts/csv_to_npz.py --input-dir "$RAW_DIR" --output-dir "$NPZ_DIR" --input-fps "$INPUT_FPS" --output-fps "$OUTPUT_FPS"
  run_uv scripts/pretrain.py --data-dir "$NPZ_DIR" --num-layers "$PRETRAIN_NUM_LAYERS" --no-use-ema --save-interval "$PRETRAIN_SAVE_INTERVAL" --num-epochs "$PRETRAIN_EPOCHS" --train-split 1.0 --d-model "$PRETRAIN_D_MODEL" --name "$PRIOR_NAME" --log-dir "$LOG_DIR" --wandb-project smp
  local latest_run
  latest_run="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  [ -n "$latest_run" ] && [ -f "$latest_run/pretrained.pt" ] || {
    echo "[ERROR] pretrained.pt not found under $RUN_ROOT" >&2; exit 1;
  }
  cp -a "$latest_run/pretrained.pt" "$CKPT_PATH"
  echo "[DONE] Fully mirrored NewDev Exp13 prior -> $CKPT_PATH"
}

require_inputs
reset_dirs
stage_sources
ensure_missing_mirrors
verify_complete_mirroring
find "$RAW_DIR" -maxdepth 1 -type f -name '*.csv' -printf '[INFO] %f\n' | sort
convert_and_pretrain
