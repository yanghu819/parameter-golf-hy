#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found at $PYTHON_BIN"
    echo "Run: bash setup.sh"
    exit 1
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards "$TRAIN_SHARDS"

"$PYTHON_BIN" - <<PY
from pathlib import Path

variant = "${VARIANT}"
data_dir = Path("data/datasets") / f"fineweb10B_{variant}"
train_count = len(sorted(data_dir.glob("fineweb_train_*.bin")))
val_count = len(sorted(data_dir.glob("fineweb_val_*.bin")))
print(f"downloaded_variant={variant}")
print(f"train_shards={train_count}")
print(f"val_shards={val_count}")
print(f"tokenizers={Path('data/tokenizers').resolve()}")
PY
