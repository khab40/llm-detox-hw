#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/04_train_dpo.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[04_train_dpo] started: $(date -Is)"
echo "[04_train_dpo] pwd=$(pwd)"
echo "[04_train_dpo] python=${PYTHON}"
"$PYTHON" -u -m tasks.task2_dpo_loss
"$PYTHON" -u -m src.detox_hw.train_dpo \
  --train "${DPO_DATA:-data/dpo.jsonl}" \
  --sft-dir "${SFT_DIR:-checkpoints/sft}" \
  --out "${DPO_OUT:-checkpoints/dpo}" \
  --epochs "${EPOCHS:-1}"
echo "[04_train_dpo] finished: $(date -Is)"
