#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <user@host> <repo_url> <remote_repo_dir> [branch]" >&2
  echo "example: $0 ubuntu@1.2.3.4 git@github.com:me/llm-detox-hw.git ~/llm-detox-hw main" >&2
  exit 2
fi

HOST="$1"
REPO_URL="$2"
REMOTE_DIR="$3"
BRANCH="${4:-}"

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
  echo "  $0 $HOST $REPO_URL '~/llm-detox-hw' ${BRANCH:-main}" >&2
  exit 2
fi

if [[ -n "$BRANCH" ]]; then
  ssh "$HOST" "if [[ -d '$REMOTE_DIR/.git' ]]; then cd '$REMOTE_DIR' && git fetch origin '$BRANCH' && git checkout '$BRANCH' && git pull --ff-only; else git clone --branch '$BRANCH' '$REPO_URL' '$REMOTE_DIR'; fi && cd '$REMOTE_DIR' && chmod +x scripts/*.sh"
else
  ssh "$HOST" "if [[ -d '$REMOTE_DIR/.git' ]]; then cd '$REMOTE_DIR' && git pull --ff-only; else git clone '$REPO_URL' '$REMOTE_DIR'; fi && cd '$REMOTE_DIR' && chmod +x scripts/*.sh"
fi

echo "project ready at $HOST:$REMOTE_DIR"
