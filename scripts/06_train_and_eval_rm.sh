#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" -m tasks.task4_bt_loss
"$PYTHON" -m tests.test_task5_reward_head
"$PYTHON" -m src.detox_hw.train_rm \
  --train "${DPO_DATA:-data/dpo.jsonl}" \
  --out "${RM_OUT:-checkpoints/rm}" \
  --val-fraction "${VAL_FRACTION:-0.1}"
"$PYTHON" -m tasks.rm_eval \
  --rm-dir "${RM_DIR:-checkpoints/rm}" \
  --pairs "${DPO_DATA:-data/dpo.jsonl}" \
  2>&1 | tee "${OUT_TXT:-submissions/rm_eval.txt}"
