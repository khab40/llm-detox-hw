#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <user@host> <remote_repo_dir>" >&2
  echo "example: $0 ubuntu@1.2.3.4 ~/llm-detox-hw" >&2
  exit 2
fi

HOST="$1"
REMOTE_DIR="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$(mktemp -t llm-detox-project.XXXXXX.tar.gz)"
REMOTE_ARCHIVE="/tmp/$(basename "$ARCHIVE")"

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
  echo "  $0 $HOST '~/llm-detox-hw'" >&2
  exit 2
fi

cleanup() {
  rm -f "$ARCHIVE"
}
trap cleanup EXIT

cd "$ROOT"
COPYFILE_DISABLE=1 tar --no-xattrs \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude data \
  --exclude checkpoints \
  --exclude outputs \
  --exclude submissions \
  --exclude evidence \
  -czf "$ARCHIVE" .

ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
scp "$ARCHIVE" "$HOST:$REMOTE_ARCHIVE"
ssh "$HOST" "test -s '$REMOTE_ARCHIVE' && cd '$REMOTE_DIR' && tar -xzf '$REMOTE_ARCHIVE' && rm -f '$REMOTE_ARCHIVE' && chmod +x scripts/*.sh"

echo "uploaded project to $HOST:$REMOTE_DIR"
