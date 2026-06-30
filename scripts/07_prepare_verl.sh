#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/07_prepare_verl.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[07_prepare_verl] started: $(date -Is)"
echo "[07_prepare_verl] pwd=$(pwd)"
echo "[07_prepare_verl] python=${PYTHON}"
SYS="You are a helpful assistant. Respond to the user thoughtfully and kindly."

"$PYTHON" -u -m src.toxic_rl.prompts \
  --src "${DPO_DATA:-data/dpo.jsonl}" \
  --out "${TRAIN_PARQUET:-data/train.parquet}" \
  --system-prompt "$SYS"
"$PYTHON" -u -m src.toxic_rl.prompts \
  --src "${DPO_DATA:-data/dpo.jsonl}" \
  --out "${VAL_PARQUET:-data/val.parquet}" \
  --system-prompt "$SYS" \
  --max "${VAL_MAX:-200}"

"$PYTHON" -u -m src.detox_hw.merge_lora \
  --adapter-dir "${SFT_DIR:-checkpoints/sft}" \
  --out "${SFT_MERGED_DIR:-checkpoints/sft_merged}" \
  ${FORCE_MERGE_SFT:+--force}
chmod a+r "${SFT_MERGED_DIR:-checkpoints/sft_merged}"/model.safetensors

sudo docker pull verlai/verl:vllm023.dev1
sudo docker run --rm --gpus all verlai/verl:vllm023.dev1 nvidia-smi \
  > submissions/verl_setup.txt
printf '%s\n' '---' >> submissions/verl_setup.txt
ls -la data/*.parquet checkpoints/rm/ "${SFT_MERGED_DIR:-checkpoints/sft_merged}"/ >> submissions/verl_setup.txt
echo "[07_prepare_verl] finished: $(date -Is)"
