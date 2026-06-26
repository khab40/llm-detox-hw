#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" - <<'PY'
import importlib

mods = [
    "torch", "transformers", "peft", "datasets", "detoxify",
    "sklearn", "tqdm", "pyarrow", "pytest", "safetensors",
]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append((mod, repr(exc)))

if missing:
    print("Missing/broken Python dependencies:")
    for mod, exc in missing:
        print(f"  {mod}: {exc}")
    raise SystemExit(1)

import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if command -v docker >/dev/null 2>&1; then
  docker --version
  sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
else
  echo "docker not found"
fi
