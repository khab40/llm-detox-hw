#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <session_name> <script_path> [script args...]" >&2
  echo "example: $0 detox-task7 scripts/09_run_ppo_rm.sh" >&2
  exit 2
fi

SESSION="$1"
SCRIPT="$2"
shift 2

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p submissions
LOG="submissions/${SESSION}.tmux.log"
touch "$LOG"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install it with: sudo apt-get install -y tmux" >&2
  echo "[tmux_run] tmux is not installed" >>"$LOG"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "attach with: tmux attach -t $SESSION" >&2
  {
    echo "[tmux_run] session already exists: $SESSION"
    tmux list-sessions
  } >>"$LOG"
  exit 1
fi

ROOT="$(pwd)"
printf -v ROOT_Q "%q" "$ROOT"
printf -v SCRIPT_Q "%q" "$SCRIPT"
printf -v LOG_Q "%q" "$LOG"

ARGS_Q=()
for arg in "$@"; do
  printf -v ARG_Q "%q" "$arg"
  ARGS_Q+=("$ARG_Q")
done

CMD="set -o pipefail; cd $ROOT_Q && { echo '[tmux_run] started:' \$(date -Is); bash $SCRIPT_Q ${ARGS_Q[*]}; status=\$?; echo '[tmux_run] exited with status:' \$status 'at' \$(date -Is); exit \$status; } 2>&1 | tee -a $LOG_Q"

tmux new-session -d -s "$SESSION" "$CMD"

echo "started tmux session: $SESSION"
echo "log: $LOG"
echo "attach: tmux attach -t $SESSION"
echo "tail: tail -f $LOG"
