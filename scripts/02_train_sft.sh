#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/02_train_sft.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[02_train_sft] started: $(date -Is)"
echo "[02_train_sft] pwd=$(pwd)"
echo "[02_train_sft] python=${PYTHON}"
"$PYTHON" -u -m src.detox_hw.train_sft \
  --train "${SFT_DATA:-data/sft.jsonl}" \
  --out "${SFT_OUT:-checkpoints/sft}" \
  --epochs "${EPOCHS:-1}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --log-every "${LOG_EVERY:-10}"
echo "[02_train_sft] finished: $(date -Is)"
