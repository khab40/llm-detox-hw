#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p submissions
LOG="${LOG:-submissions/00_setup_nebius_vm.log}"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "[00_setup_nebius_vm] started: $(date -Is)"
echo "[00_setup_nebius_vm] pwd=$(pwd)"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip tmux
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-nebius.txt

mkdir -p data checkpoints outputs submissions

if command -v docker >/dev/null 2>&1; then
  sudo docker pull verlai/verl:vllm023.dev1
else
  echo "docker is not installed. Install docker + nvidia-container-toolkit before PPO steps." >&2
fi

echo "[00_setup_nebius_vm] finished: $(date -Is)"
