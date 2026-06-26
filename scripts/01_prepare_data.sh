#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" -m data_prep.build_pairs \
  --out-dir "${OUT_DIR:-data}" \
  --max-rows "${MAX_ROWS:-80000}"
