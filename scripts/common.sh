#!/usr/bin/env bash
set -euo pipefail

repo_root() {
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  printf '%s\n' "$dir"
}

cd "$(repo_root)"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

mkdir -p data checkpoints outputs submissions
