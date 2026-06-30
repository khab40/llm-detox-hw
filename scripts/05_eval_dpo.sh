#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/05_eval_dpo.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[05_eval_dpo] started: $(date -Is)"
"$PYTHON" -m tasks.task3_dpo_eval \
  --sft-dir "${SFT_DIR:-checkpoints/sft}" \
  --dpo-dir "${DPO_DIR:-checkpoints/dpo}" \
  --out "${OUT_JSON:-submissions/task3_dpo_eval.json}" \
  2>&1 | tee "${OUT_TXT:-submissions/task3_dpo_eval.txt}"
echo "[05_eval_dpo] finished: $(date -Is)"
