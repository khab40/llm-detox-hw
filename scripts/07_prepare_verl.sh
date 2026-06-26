#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SYS="You are a helpful assistant. Respond to the user thoughtfully and kindly."

"$PYTHON" -m src.toxic_rl.prompts \
  --src "${DPO_DATA:-data/dpo.jsonl}" \
  --out "${TRAIN_PARQUET:-data/train.parquet}" \
  --system-prompt "$SYS"
"$PYTHON" -m src.toxic_rl.prompts \
  --src "${DPO_DATA:-data/dpo.jsonl}" \
  --out "${VAL_PARQUET:-data/val.parquet}" \
  --system-prompt "$SYS" \
  --max "${VAL_MAX:-200}"

sudo docker pull verlai/verl:vllm023.dev1
sudo docker run --rm --gpus all verlai/verl:vllm023.dev1 nvidia-smi \
  > submissions/verl_setup.txt
printf '%s\n' '---' >> submissions/verl_setup.txt
ls -la data/*.parquet checkpoints/rm/ >> submissions/verl_setup.txt
