#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <user@host> <remote_repo_dir> <script_path> [script args...]" >&2
  echo "example: $0 ubuntu@1.2.3.4 ~/llm-detox-hw scripts/00_check_env.sh" >&2
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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  --exclude .git \
  --exclude .venv \
  --exclude data \
  --exclude checkpoints \
  --exclude outputs \
  --exclude submissions \
  "$ROOT"/ "$HOST":"$REMOTE_DIR"/

SESSION="${TMUX_SESSION:-$(basename "$SCRIPT" .sh)}"
SESSION="${SESSION//[^A-Za-z0-9_.-]/-}"

if [[ "${SSH_RUN_DIRECT:-0}" == "1" || "$SCRIPT" == "scripts/00_setup_nebius_vm.sh" ]]; then
  ssh -t "$HOST" "cd '$REMOTE_DIR' && bash '$SCRIPT' $*"
else
  ssh -t "$HOST" "cd '$REMOTE_DIR' && mkdir -p submissions && touch 'submissions/${SESSION}.tmux.log' && echo '[ssh_sync_and_run] launching ${SCRIPT} in tmux session ${SESSION}' >> 'submissions/${SESSION}.tmux.log' && bash scripts/tmux_run.sh '$SESSION' '$SCRIPT' $* && sleep 2 && if test -f 'submissions/${SESSION}.tmux.log'; then echo 'remote log exists:' 'submissions/${SESSION}.tmux.log'; tail -n 20 'submissions/${SESSION}.tmux.log'; else echo 'remote log missing:' 'submissions/${SESSION}.tmux.log'; pwd; ls -lah; ls -lah submissions 2>/dev/null || true; fi"
fi
