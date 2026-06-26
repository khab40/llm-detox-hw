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
ssh -t "$HOST" "cd '$REMOTE_DIR' && bash '$SCRIPT' $*"
