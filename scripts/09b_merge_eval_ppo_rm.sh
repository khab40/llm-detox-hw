#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/09b_merge_eval_ppo_rm.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[09b_merge_eval_ppo_rm] started: $(date -Is)"
sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -u -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_rm/global_step_100/actor \
             --target_dir /workspace/checkpoints/ppo_rm_merged"

sudo chmod 644 checkpoints/ppo_rm_merged/model.safetensors
ls -la checkpoints/ppo_rm_merged/ > submissions/task7_merged_ls.txt

"$PYTHON" -m tasks.task7_ppo_rm_eval \
  --ppo-dir checkpoints/ppo_rm_merged \
  --out submissions/task7_ppo_rm_eval.json \
  2>&1 | tee submissions/task7_ppo_rm_eval.txt
echo "[09b_merge_eval_ppo_rm] finished: $(date -Is)"
