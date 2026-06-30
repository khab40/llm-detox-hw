#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/10b_merge_eval_ppo_custom.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[10b_merge_eval_ppo_custom] started: $(date -Is)"
CUSTOM_SAVE_STEP="${CUSTOM_SAVE_STEP:-100}"
CUSTOM_MERGED_DIR="${CUSTOM_MERGED_DIR:-checkpoints/ppo_custom_merged}"
CUSTOM_EVAL_OUT="${CUSTOM_EVAL_OUT:-submissions/task8_ppo_custom_eval.json}"
CUSTOM_EVAL_TXT="${CUSTOM_EVAL_TXT:-submissions/task8_ppo_custom_eval.txt}"
CUSTOM_MERGED_LS="${CUSTOM_MERGED_LS:-submissions/task8_merged_ls.txt}"
echo "[10b_merge_eval_ppo_custom] save_step=${CUSTOM_SAVE_STEP} merged_dir=${CUSTOM_MERGED_DIR}"
sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -u -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_custom/global_step_${CUSTOM_SAVE_STEP}/actor \
             --target_dir /workspace/${CUSTOM_MERGED_DIR}"

sudo chmod 644 "${CUSTOM_MERGED_DIR}/model.safetensors"
ls -la "${CUSTOM_MERGED_DIR}/" > "${CUSTOM_MERGED_LS}"

"$PYTHON" -m tasks.task7_ppo_rm_eval \
  --ppo-dir "${CUSTOM_MERGED_DIR}" \
  --out "${CUSTOM_EVAL_OUT}" \
  2>&1 | tee "${CUSTOM_EVAL_TXT}"
echo "[10b_merge_eval_ppo_custom] finished: $(date -Is)"
