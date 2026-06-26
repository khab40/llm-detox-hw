#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" -m tasks.task2_dpo_loss
"$PYTHON" -m src.detox_hw.train_dpo \
  --train "${DPO_DATA:-data/dpo.jsonl}" \
  --sft-dir "${SFT_DIR:-checkpoints/sft}" \
  --out "${DPO_OUT:-checkpoints/dpo}" \
  --epochs "${EPOCHS:-1}"
