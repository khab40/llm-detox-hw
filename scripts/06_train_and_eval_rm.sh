#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/06_train_and_eval_rm.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[06_train_and_eval_rm] started: $(date -Is)"
echo "[06_train_and_eval_rm] pwd=$(pwd)"
echo "[06_train_and_eval_rm] python=${PYTHON}"
"$PYTHON" -u -m tasks.task4_bt_loss
"$PYTHON" -u -m tests.test_task5_reward_head
"$PYTHON" -u -m src.detox_hw.train_rm \
  --train "${DPO_DATA:-data/dpo.jsonl}" \
  --out "${RM_OUT:-checkpoints/rm}" \
  --val-fraction "${VAL_FRACTION:-0.1}"
"$PYTHON" -m tasks.rm_eval \
  --rm-dir "${RM_DIR:-checkpoints/rm}" \
  --pairs "${DPO_DATA:-data/dpo.jsonl}" \
  2>&1 | tee "${OUT_TXT:-submissions/rm_eval.txt}"
echo "[06_train_and_eval_rm] finished: $(date -Is)"
