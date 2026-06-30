#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/08_run_ppo_inv_detoxify.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[08_run_ppo_inv_detoxify] started: $(date -Is)"
echo "[08_run_ppo_inv_detoxify] pwd=$(pwd)"
PPO_ACTOR_PATH="${PPO_ACTOR_PATH:-/workspace/checkpoints/sft_merged}"
echo "[08_run_ppo_inv_detoxify] actor_path=${PPO_ACTOR_PATH}"
sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=inv:detoxify \
  -e HYDRA_FULL_ERROR=1 \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -u -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path ${PPO_ACTOR_PATH} \
             --out outputs/ppo_inv_detoxify \
             --reward inv:detoxify \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task6_log.txt

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -u -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_inv_detoxify/global_step_100/actor \
             --target_dir /workspace/checkpoints/ppo_inv_detoxify_merged"

sudo chmod 644 checkpoints/ppo_inv_detoxify_merged/model.safetensors
ls -la checkpoints/ppo_inv_detoxify_merged/ > submissions/task6_merged_ls.txt

"$PYTHON" -m tasks.task6_ppo_detoxify_eval \
  --ppo-dir checkpoints/ppo_inv_detoxify_merged \
  --out submissions/task6_ppo_detoxify_eval.json \
  2>&1 | tee submissions/task6_ppo_detoxify_eval.txt
echo "[08_run_ppo_inv_detoxify] finished: $(date -Is)"
