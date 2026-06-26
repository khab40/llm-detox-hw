#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" -m tasks.task3_dpo_eval \
  --sft-dir "${SFT_DIR:-checkpoints/sft}" \
  --dpo-dir "${DPO_DIR:-checkpoints/dpo}" \
  --out "${OUT_JSON:-submissions/task3_dpo_eval.json}" \
  2>&1 | tee "${OUT_TXT:-submissions/task3_dpo_eval.txt}"
