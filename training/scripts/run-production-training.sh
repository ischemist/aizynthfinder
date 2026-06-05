#!/usr/bin/env bash
set -euo pipefail

release="${RELEASE:-v2026-06-05}"
workers="${WORKERS:-8}"
epochs="${EPOCHS:-100}"
batch_size="${BATCH_SIZE:-256}"
template_occurrence="${TEMPLATE_OCCURRENCE:-3}"

usage() {
  cat <<'EOF'
usage: scripts/run-production-training.sh [reaction|route|both]

environment:
  RELEASE=v2026-06-05
  WORKERS=8
  EPOCHS=100
  BATCH_SIZE=256
  TEMPLATE_OCCURRENCE=3
EOF
}

target="${1:-both}"
case "$target" in
  reaction|route|both) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

run_one() {
  local kind="$1"
  local artifact="single-step-${kind}-holdout-n1-n5"
  local run="retrocast_${release}_ss_${kind}-holdout-n1-n5"

  uv run aizynth-train-one-step download-retrocast \
    --release "$release" \
    --artifact "$artifact" \
    --split all \
    --format rsmi \
    --output-dir data/raw

  uv run aizynth-train-one-step normalize-reactions \
    "data/raw/${release}/${artifact}/all.rsmi.txt.gz" \
    --output "data/processed/${run}_all_reactions.csv"

  uv run aizynth-train-one-step extract \
    "data/processed/${run}_all_reactions.csv" \
    --output "runs/${run}/${run}_all_raw_template_library.csv" \
    --radius 1 \
    --min-count 1 \
    --workers "$workers"

  uv run aizynth-train-one-step preprocess-all \
    "runs/${run}/${run}_all_raw_template_library.csv" \
    --work-dir "runs/${run}" \
    --file-prefix "$run" \
    --template-occurrence "$template_occurrence"

  uv run aizynth-train-one-step train \
    --work-dir "runs/${run}" \
    --file-prefix "$run" \
    --epochs "$epochs" \
    --batch-size "$batch_size"

  uv run aizynth-train-one-step write-config \
    --work-dir "runs/${run}" \
    --file-prefix "$run" \
    --output "runs/${run}/aizynth_config.yml"
}

if [[ "$target" == "both" ]]; then
  run_one reaction
  run_one route
else
  run_one "$target"
fi
