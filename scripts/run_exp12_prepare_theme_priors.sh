#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_exp12_prepare_theme_priors.sh [--gpu N] [--styles "male female children"]

Builds Experiment 12 theme priors from datasets/csv/theme.
For each style, stages:
  - CSV files whose filename contains the style token
  - common stop/static/stand CSV files

Outputs:
  datasets/pretrain_ckpt/exp12_theme_male.pt
  datasets/pretrain_ckpt/exp12_theme_female.pt
  datasets/pretrain_ckpt/exp12_theme_children.pt
EOF
}

GPU=""
STYLES="male female children"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --styles)
      STYLES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

THEME_CSV_DIR="${THEME_CSV_DIR:-datasets/csv/theme}"
WORK_ROOT="${WORK_ROOT:-datasets/exp12_prior_sources}"
NPZ_ROOT="${NPZ_ROOT:-datasets/npz/exp12}"
CKPT_ROOT="${CKPT_ROOT:-datasets/pretrain_ckpt}"
LOG_DIR="${LOG_DIR:-logs/pretrain}"
INPUT_FPS="${INPUT_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10000}"
PRETRAIN_SAVE_INTERVAL="${PRETRAIN_SAVE_INTERVAL:-5000}"
PRETRAIN_D_MODEL="${PRETRAIN_D_MODEL:-128}"
PRETRAIN_NUM_LAYERS="${PRETRAIN_NUM_LAYERS:-2}"

run_uv() {
  if [ -n "$GPU" ]; then
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" uv run "$@"
  else
    WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}" uv run "$@"
  fi
}

if [ ! -d "$THEME_CSV_DIR" ]; then
  echo "[ERROR] THEME_CSV_DIR not found: $THEME_CSV_DIR" >&2
  exit 1
fi

mkdir -p "$CKPT_ROOT"

for STYLE in $STYLES; do
  PRIOR_NAME="exp12_theme_${STYLE}"
  RAW_DIR="$WORK_ROOT/$PRIOR_NAME/raw"
  NPZ_DIR="$NPZ_ROOT/$PRIOR_NAME"
  CKPT_PATH="$CKPT_ROOT/$PRIOR_NAME.pt"

  rm -rf "$WORK_ROOT/$PRIOR_NAME" "$NPZ_DIR"
  mkdir -p "$RAW_DIR" "$NPZ_DIR"

  STYLE_COUNT=0
  for CSV_PATH in "$THEME_CSV_DIR"/*.csv; do
    [ -e "$CSV_PATH" ] || continue
    BASE="$(basename "$CSV_PATH")"
    STEM="${BASE%.csv}"
    if [[ "$STEM" =~ (^|[_\.-])${STYLE}([_\.-]|$) ]]; then
      cp -a "$CSV_PATH" "$RAW_DIR"/
      STYLE_COUNT=$((STYLE_COUNT + 1))
    fi
  done
  find "$THEME_CSV_DIR" -maxdepth 1 -type f \
    \( -name "*stop*.csv" -o -name "*static*.csv" -o -name "*stand*.csv" \) \
    -exec cp -a {} "$RAW_DIR"/ \;

  CSV_COUNT="$(find "$RAW_DIR" -maxdepth 1 -type f -name "*.csv" | wc -l)"
  if [ "$STYLE_COUNT" -eq 0 ]; then
    echo "[ERROR] No style CSV files staged for style '$STYLE' in $RAW_DIR" >&2
    echo "        Use --styles with tokens matching filenames under $THEME_CSV_DIR." >&2
    exit 1
  fi

  echo "[INFO] style=$STYLE raw_dir=$RAW_DIR csv_count=$CSV_COUNT"
  find "$RAW_DIR" -maxdepth 1 -type f -name "*.csv" -printf "  %f\n" | sort

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

  LATEST_RUN="$(find "$LOG_DIR/$PRIOR_NAME" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [ -z "$LATEST_RUN" ] || [ ! -f "$LATEST_RUN/pretrained.pt" ]; then
    echo "[ERROR] pretrained.pt not found under $LOG_DIR/$PRIOR_NAME" >&2
    exit 1
  fi

  cp -a "$LATEST_RUN/pretrained.pt" "$CKPT_PATH"
  echo "[DONE] style=$STYLE -> $CKPT_PATH"
done
