#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

required=(
  "submissions/task1_sft_eval.txt"
  "submissions/task3_dpo_eval.txt"
  "submissions/rm_eval.txt"
  "submissions/task6_ppo_detoxify_eval.txt"
  "submissions/task7_ppo_rm_eval.txt"
  "submissions/task8_ppo_custom_eval.txt"
  "submissions/task8_writeup.md"
  "submissions/verl_setup.txt"
  "submissions/task6_log.txt"
  "submissions/task6_merged_ls.txt"
  "submissions/task7_log.txt"
  "submissions/task7_merged_ls.txt"
  "submissions/task8_log.txt"
  "submissions/task8_merged_ls.txt"
)

missing=()
empty=()

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    missing+=("$path")
  elif [[ ! -s "$path" ]]; then
    empty+=("$path")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "missing required evidence files:" >&2
  printf '  %s\n' "${missing[@]}" >&2
fi

if (( ${#empty[@]} > 0 )); then
  echo "empty required evidence files:" >&2
  printf '  %s\n' "${empty[@]}" >&2
fi

if (( ${#missing[@]} > 0 || ${#empty[@]} > 0 )); then
  exit 1
fi

echo "all required evidence files are present and non-empty"
