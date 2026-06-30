#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p submissions evidence

echo "Removing previous evidence files and logs from submissions/ and evidence/ ..."

rm -f \
  submissions/*.txt \
  submissions/*.json \
  submissions/*.out \
  submissions/*.log \
  submissions/*_log.txt \
  submissions/*_merged_ls.txt \
  submissions/verl_setup.txt \
  submissions/task8_writeup.md \
  evidence/llm-detox-evidence-*.zip \
  llm-detox-evidence-*.zip

if [[ "${CLEAN_RUN_ARTIFACTS:-0}" == "1" ]]; then
  echo "CLEAN_RUN_ARTIFACTS=1 set; removing PPO outputs and merged PPO checkpoints ..."
  rm -rf \
    outputs/ppo_inv_detoxify \
    outputs/ppo_rm \
    outputs/ppo_custom \
    checkpoints/ppo_inv_detoxify_merged \
    checkpoints/ppo_rm_merged \
    checkpoints/ppo_custom_merged
fi

echo "cleanup complete"
