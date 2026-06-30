#!/usr/bin/env bash
set -euo pipefail

mkdir -p submissions
LOG="${LOG:-submissions/01_prepare_data.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[01_prepare_data] bootstrap: $(date -Is)"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

OUT_DIR="${OUT_DIR:-data}"
MAX_ROWS="${MAX_ROWS:-80000}"

echo "[01_prepare_data] started: $(date -Is)"
echo "[01_prepare_data] pwd=$(pwd)"
echo "[01_prepare_data] python=${PYTHON}"
echo "[01_prepare_data] out_dir=${OUT_DIR} max_rows=${MAX_ROWS}"

"$PYTHON" -u -m data_prep.build_pairs \
  --out-dir "$OUT_DIR" \
  --max-rows "$MAX_ROWS"

echo "[01_prepare_data] generated files:"
ls -lah "$OUT_DIR"/sft.jsonl "$OUT_DIR"/dpo.jsonl
echo "[01_prepare_data] line counts:"
wc -l "$OUT_DIR"/sft.jsonl "$OUT_DIR"/dpo.jsonl
echo "[01_prepare_data] finished: $(date -Is)"
