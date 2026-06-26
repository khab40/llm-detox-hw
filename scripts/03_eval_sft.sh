#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" -m tasks.task1_sft_eval \
  --sft-dir "${SFT_DIR:-checkpoints/sft}" \
  --out "${OUT_JSON:-submissions/task1_sft_eval.json}" \
  2>&1 | tee "${OUT_TXT:-submissions/task1_sft_eval.txt}"
