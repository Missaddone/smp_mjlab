#!/usr/bin/env bash
set -euo pipefail

FORWARD_CSV_DIR="${FORWARD_CSV_DIR:-datasets/csv/forward}"
WORK_ROOT="${WORK_ROOT:-datasets/exp9_prior_sources}"
NPZ_ROOT="${NPZ_ROOT:-datasets/npz/exp9}"
CKPT_ROOT="${CKPT_ROOT:-datasets/pretrain_ckpt}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10000}"
PRETRAIN_SAVE_INTERVAL="${PRETRAIN_SAVE_INTERVAL:-5000}"
PRETRAIN_D_MODEL="${PRETRAIN_D_MODEL:-128}"
PRETRAIN_NUM_LAYERS="${PRETRAIN_NUM_LAYERS:-2}"

copy_required() {
  local dest_dir="$1"
  local label="$2"
  shift 2

  local src
  for src in "$@"; do
    if [ -f "$FORWARD_CSV_DIR/$src" ]; then
      cp -a "$FORWARD_CSV_DIR/$src" "$dest_dir/$src"
      return 0
    fi
  done

  echo "[ERROR] Missing required CSV for $label. Tried:" >&2
  for src in "$@"; do
    echo "  $FORWARD_CSV_DIR/$src" >&2
  done
  exit 1
}

copy_optional() {
  local dest_dir="$1"
  shift

  local src
  for src in "$@"; do
    if [ -f "$FORWARD_CSV_DIR/$src" ]; then
      cp -a "$FORWARD_CSV_DIR/$src" "$dest_dir/$src"
      return 0
    fi
  done
  return 0
}

copy_forward_base() {
  local dest_dir="$1"
  cp -a "$FORWARD_CSV_DIR/g1_walk.csv" "$dest_dir/"
  cp -a "$FORWARD_CSV_DIR/g1_walk_mirror.csv" "$dest_dir/"
  cp -a "$FORWARD_CSV_DIR/g1_jog.csv" "$dest_dir/"
  cp -a "$FORWARD_CSV_DIR/g1_jog_mirror.csv" "$dest_dir/"
  cp -a "$FORWARD_CSV_DIR/g1_run.csv" "$dest_dir/"
  cp -a "$FORWARD_CSV_DIR/g1_run_mirror.csv" "$dest_dir/"
}

ensure_missing_mirrors() {
  local prior_name="$1"
  local raw_dir="$WORK_ROOT/$prior_name/raw"
  local mirror_tmp_root="$WORK_ROOT/$prior_name/mirror_tmp"

  rm -rf "$mirror_tmp_root"
  mkdir -p "$mirror_tmp_root"

  local csv_path
  for csv_path in "$raw_dir"/*.csv; do
    [ -e "$csv_path" ] || continue

    local base
    base="$(basename "$csv_path")"
    case "$base" in
      *_mirror.csv)
        continue
        ;;
    esac

    local mirror_name="${base%.csv}_mirror.csv"
    if [ -f "$raw_dir/$mirror_name" ]; then
      continue
    fi

    local one_in="$mirror_tmp_root/${base%.csv}_in"
    local one_out="$mirror_tmp_root/${base%.csv}_out"
    mkdir -p "$one_in" "$one_out"
    cp -a "$csv_path" "$one_in/$base"

    uv run scripts/mirror_motion_data.py \
      --input-dir "$one_in" \
      --output-dir "$one_out" \
      --csv \
      --no-npz

    cp -a "$one_out/$mirror_name" "$raw_dir/$mirror_name"
    echo "[INFO] Generated missing mirror: $mirror_name"
  done

  rm -rf "$mirror_tmp_root"
}

prepare_prior_raw() {
  local prior_name="$1"
  local mode="$2"
  local raw_dir="$WORK_ROOT/$prior_name/raw"
  local npz_dir="$NPZ_ROOT/$prior_name"

  rm -rf "$WORK_ROOT/$prior_name" "$npz_dir"
  mkdir -p "$raw_dir" "$npz_dir" "$CKPT_ROOT"

  copy_forward_base "$raw_dir"

  case "$mode" in
    w2s)
      copy_required "$raw_dir" "walk_to_stop" walk_to_stop.csv w_2_s.csv
      copy_optional "$raw_dir" walk_to_stop_mirror.csv w_2_s_mirror.csv
      ;;
    w2s_s2w)
      copy_required "$raw_dir" "walk_to_stop" walk_to_stop.csv w_2_s.csv
      copy_required "$raw_dir" "stop_to_walk" stop_to_walk.csv s_2_w.csv
      copy_optional "$raw_dir" walk_to_stop_mirror.csv w_2_s_mirror.csv
      copy_optional "$raw_dir" stop_to_walk_mirror.csv s_2_w_mirror.csv
      ;;
    w2s_s2w_stop_static)
      copy_required "$raw_dir" "walk_to_stop" walk_to_stop.csv w_2_s.csv
      copy_required "$raw_dir" "stop_to_walk" stop_to_walk.csv s_2_w.csv
      copy_required "$raw_dir" "stop_static" stop_static.csv
      copy_optional "$raw_dir" walk_to_stop_mirror.csv w_2_s_mirror.csv
      copy_optional "$raw_dir" stop_to_walk_mirror.csv s_2_w_mirror.csv
      copy_optional "$raw_dir" stop_static_mirror.csv
      ;;
    stop_static)
      copy_required "$raw_dir" "stop_static" stop_static.csv
      copy_optional "$raw_dir" stop_static_mirror.csv
      ;;
    *)
      echo "[ERROR] Unknown prior mode: $mode" >&2
      exit 1
      ;;
  esac

  ensure_missing_mirrors "$prior_name"

  local count
  count="$(find "$raw_dir" -maxdepth 1 -type f -name "*.csv" | wc -l)"
  echo "[INFO] $prior_name raw csv_count=$count"
  find "$raw_dir" -maxdepth 1 -type f -name "*.csv" -printf "  %f\n" | sort
}

convert_and_pretrain() {
  local prior_name="$1"
  local raw_dir="$WORK_ROOT/$prior_name/raw"
  local npz_dir="$NPZ_ROOT/$prior_name"
  local ckpt_path="$CKPT_ROOT/$prior_name.pt"

  uv run scripts/csv_to_npz.py \
    --input-dir "$raw_dir" \
    --output-dir "$npz_dir" \
    --output-fps "$OUTPUT_FPS"

  uv run scripts/pretrain.py \
    --data-dir "$npz_dir" \
    --name "$prior_name" \
    --num-layers "$PRETRAIN_NUM_LAYERS" \
    --no-use-ema \
    --save-interval "$PRETRAIN_SAVE_INTERVAL" \
    --num-epochs "$PRETRAIN_EPOCHS" \
    --train-split 1.0 \
    --d-model "$PRETRAIN_D_MODEL"

  local latest_run
  latest_run="$(find "logs/pretrain/$prior_name" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
  if [ -z "$latest_run" ] || [ ! -f "$latest_run/pretrained.pt" ]; then
    echo "[ERROR] pretrained.pt not found under logs/pretrain/$prior_name" >&2
    exit 1
  fi

  cp -a "$latest_run/pretrained.pt" "$ckpt_path"
  echo "[DONE] $prior_name -> $ckpt_path"
}

if [ ! -d "$FORWARD_CSV_DIR" ]; then
  echo "[ERROR] FORWARD_CSV_DIR not found: $FORWARD_CSV_DIR" >&2
  exit 1
fi

prepare_prior_raw "exp9_prior1_forward_walk_to_stop" "w2s"
prepare_prior_raw "exp9_prior2_forward_w2s_s2w" "w2s_s2w"
prepare_prior_raw "exp9_prior3_forward_w2s_s2w_stop_static" "w2s_s2w_stop_static"
prepare_prior_raw "exp9_prior4_forward_stop_static" "stop_static"

convert_and_pretrain "exp9_prior1_forward_walk_to_stop"
convert_and_pretrain "exp9_prior2_forward_w2s_s2w"
convert_and_pretrain "exp9_prior3_forward_w2s_s2w_stop_static"
convert_and_pretrain "exp9_prior4_forward_stop_static"
