#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <user@host> <remote_repo_dir> <script_path> [script args...]" >&2
  echo "example: $0 ubuntu@1.2.3.4 '~/llm-detox-hw' scripts/01_prepare_data.sh" >&2
  echo "set FORCE_KILL=1 to kill an existing tmux session with the same name" >&2
  exit 2
fi

HOST="$1"
REMOTE_DIR="$2"
SCRIPT="$3"
shift 3

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
  echo "  $0 $HOST '~/llm-detox-hw' $SCRIPT" >&2
  exit 2
fi

SESSION="${TMUX_SESSION:-$(basename "$SCRIPT" .sh)}"
SESSION="${SESSION//[^A-Za-z0-9_.-]/-}"

if [[ "${FORCE_KILL:-0}" == "1" ]]; then
  ssh "$HOST" "cd '$REMOTE_DIR' && tmux kill-session -t '$SESSION' 2>/dev/null || true"
else
  ssh "$HOST" "cd '$REMOTE_DIR' && if tmux has-session -t '$SESSION' 2>/dev/null; then echo 'tmux session is still running: $SESSION' >&2; echo 'attach with: tmux attach -t $SESSION' >&2; exit 1; fi"
fi

ssh "$HOST" "cd '$REMOTE_DIR' && mkdir -p submissions && rm -f 'submissions/${SESSION}.tmux.log' 'submissions/${SESSION}.log'"

exec "$(dirname "$0")/ssh_run_nebius.sh" "$HOST" "$REMOTE_DIR" "$SCRIPT" "$@"
