#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <user@host> <remote_repo_dir> [local_out_dir]" >&2
  echo "example: $0 ubuntu@1.2.3.4 ~/llm-detox-hw ./evidence" >&2
  exit 2
fi

HOST="$1"
REMOTE_DIR="$2"
LOCAL_OUT="${3:-./evidence}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_ZIP="llm-detox-evidence-${STAMP}.zip"
REMOTE_ZIP_PATH="/tmp/${REMOTE_ZIP}"

normalize_remote_dir() {
  local dir="$1"
  local remote_user="${HOST%@*}"
  if [[ "$remote_user" == "$HOST" ]]; then
    remote_user="${USER:-}"
  fi

  if [[ "$dir" == "~" ]]; then
    if [[ "$remote_user" == "root" ]]; then
      dir="/root"
    else
      dir="/home/$remote_user"
    fi
  elif [[ "$dir" == "~/"* ]]; then
    if [[ "$remote_user" == "root" ]]; then
      dir="/root/${dir#~/}"
    else
      dir="/home/$remote_user/${dir#~/}"
    fi
  fi
  dir="${dir//\/~\//\/}"

  printf '%s\n' "$dir"
}

REMOTE_DIR="$(normalize_remote_dir "$REMOTE_DIR")"

if [[ "$REMOTE_DIR" == /Users/* ]]; then
  echo "remote_repo_dir looks like a local macOS path: $REMOTE_DIR" >&2
  echo "Quote the remote home path so your local shell does not expand it:" >&2
  echo "  $0 $HOST '~/llm-detox-hw' ${LOCAL_OUT}" >&2
  exit 2
fi

mkdir -p "$LOCAL_OUT"

CHECK_CMD="bash scripts/12_check_evidence.sh"
if [[ "${ALLOW_PARTIAL_EVIDENCE:-0}" == "1" ]]; then
  CHECK_CMD="bash scripts/12_check_evidence.sh || echo '[collect] WARNING: evidence check failed; collecting partial evidence because ALLOW_PARTIAL_EVIDENCE=1'"
fi

echo "remote: $HOST:$REMOTE_DIR"
echo "local output dir: $LOCAL_OUT"

ssh "$HOST" "cd '$REMOTE_DIR' && $CHECK_CMD && rm -f '$REMOTE_ZIP_PATH' && if command -v zip >/dev/null 2>&1; then zip -r '$REMOTE_ZIP_PATH' \
  tasks/task2_dpo_loss.py \
  tasks/task4_bt_loss.py \
  tasks/task5_reward_head.py \
  tasks/task8_custom_reward.py \
  src/detox_hw/eval_lib.py \
  submissions/task1_sft_eval.txt \
  submissions/task3_dpo_eval.txt \
  submissions/rm_eval.txt \
  submissions/task6_ppo_detoxify_eval.txt \
  submissions/task7_ppo_rm_eval.txt \
  submissions/task8_ppo_custom_eval.txt \
  submissions/ \
  README.md \
  assets/docs/ARCHITECTURE.md; else python3 -c \"from pathlib import Path; import sys, zipfile; out=sys.argv[1]; roots=sys.argv[2:]; seen=set(); z=zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED); paths=(p for root in roots for p in ([Path(root)] if Path(root).is_file() else Path(root).rglob('*')) if p.is_file()); [z.write(p, p.as_posix()) for p in paths if not (p.as_posix() in seen or seen.add(p.as_posix()))]; z.close()\" '$REMOTE_ZIP_PATH' \
  tasks/task2_dpo_loss.py \
  tasks/task4_bt_loss.py \
  tasks/task5_reward_head.py \
  tasks/task8_custom_reward.py \
  src/detox_hw/eval_lib.py \
  submissions/task1_sft_eval.txt \
  submissions/task3_dpo_eval.txt \
  submissions/rm_eval.txt \
  submissions/task6_ppo_detoxify_eval.txt \
  submissions/task7_ppo_rm_eval.txt \
  submissions/task8_ppo_custom_eval.txt \
  submissions/ \
  README.md \
  assets/docs/ARCHITECTURE.md; fi && test -s '$REMOTE_ZIP_PATH' && ls -lh '$REMOTE_ZIP_PATH'"

scp "$HOST:$REMOTE_ZIP_PATH" "$LOCAL_OUT/$REMOTE_ZIP"

test -s "$LOCAL_OUT/$REMOTE_ZIP"

echo "downloaded: $LOCAL_OUT/$REMOTE_ZIP"
echo "inspect with: unzip -l '$LOCAL_OUT/$REMOTE_ZIP'"
