#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p submissions

if [[ -z "${TMUX:-}" && "${RUN_WITHOUT_TMUX:-0}" != "1" ]]; then
  exec scripts/tmux_run.sh detox-full scripts/11_run_steps_1_to_8.sh
fi

LOG="${LOG:-submissions/11_run_steps_1_to_8.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[11_run_steps_1_to_8] started: $(date -Is)"
echo "[11_run_steps_1_to_8] pwd=$(pwd)"
scripts/01_prepare_data.sh
scripts/02_train_sft.sh
scripts/03_eval_sft.sh
scripts/04_train_dpo.sh
scripts/05_eval_dpo.sh
scripts/06_train_and_eval_rm.sh
scripts/07_prepare_verl.sh
scripts/08_run_ppo_inv_detoxify.sh
scripts/09_run_ppo_rm.sh
scripts/10_run_ppo_custom.sh
echo "[11_run_steps_1_to_8] finished: $(date -Is)"
