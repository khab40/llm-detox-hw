#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LOG="${LOG:-submissions/10_run_ppo_custom.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[10_run_ppo_custom] started: $(date -Is)"
echo "[10_run_ppo_custom] pwd=$(pwd)"
CUSTOM_ACTOR_LR="${CUSTOM_ACTOR_LR:-8e-7}"
CUSTOM_KL_COEF="${CUSTOM_KL_COEF:-0.02}"
CUSTOM_TOTAL_STEPS="${CUSTOM_TOTAL_STEPS:-60}"
CUSTOM_SAVE_STEP="${CUSTOM_SAVE_STEP:-60}"
PPO_ACTOR_PATH="${PPO_ACTOR_PATH:-/workspace/checkpoints/sft_merged}"
echo "[10_run_ppo_custom] actor_path=${PPO_ACTOR_PATH}"
echo "[10_run_ppo_custom] actor_lr=${CUSTOM_ACTOR_LR} kl_coef=${CUSTOM_KL_COEF} total_steps=${CUSTOM_TOTAL_STEPS}"

if [[ "${CLEAN_PPO_CUSTOM:-1}" == "1" ]]; then
  echo "[10_run_ppo_custom] removing stale Task 8 PPO artifacts"
  sudo rm -rf -- outputs/ppo_custom checkpoints/ppo_custom_merged
fi

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -e TOXIC_REWARD=custom:tasks.task8_custom_reward \
  -e HYDRA_FULL_ERROR=1 \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify 2>&1 | tail -1 && \
           python -u -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path ${PPO_ACTOR_PATH} \
             --out outputs/ppo_custom \
             --reward custom:tasks.task8_custom_reward \
             --total-steps ${CUSTOM_TOTAL_STEPS} --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr ${CUSTOM_ACTOR_LR} --critic-lr 1e-5 --kl-coef ${CUSTOM_KL_COEF} \
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task8_log.txt

sudo docker run --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
           python -u -m verl.model_merger merge --backend fsdp \
             --local_dir /workspace/outputs/ppo_custom/global_step_${CUSTOM_SAVE_STEP}/actor \
             --target_dir /workspace/checkpoints/ppo_custom_merged"

sudo chmod 644 checkpoints/ppo_custom_merged/model.safetensors
ls -la checkpoints/ppo_custom_merged/ > submissions/task8_merged_ls.txt

"$PYTHON" -m tasks.task7_ppo_rm_eval \
  --ppo-dir checkpoints/ppo_custom_merged \
  --out submissions/task8_ppo_custom_eval.json \
  2>&1 | tee submissions/task8_ppo_custom_eval.txt
echo "[10_run_ppo_custom] finished: $(date -Is)"
