#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
