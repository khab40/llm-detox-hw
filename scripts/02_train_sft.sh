#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" -m src.detox_hw.train_sft \
  --train "${SFT_DATA:-data/sft.jsonl}" \
  --out "${SFT_OUT:-checkpoints/sft}" \
  --epochs "${EPOCHS:-1}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum "${GRAD_ACCUM:-4}"
