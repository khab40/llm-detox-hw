#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/03_eval_sft.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[03_eval_sft] started: $(date -Is)"
"$PYTHON" -m tasks.task1_sft_eval \
  --sft-dir "${SFT_DIR:-checkpoints/sft}" \
  --out "${OUT_JSON:-submissions/task1_sft_eval.json}" \
  2>&1 | tee "${OUT_TXT:-submissions/task1_sft_eval.txt}"
echo "[03_eval_sft] finished: $(date -Is)"
