#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="${REPO_NAME:-parameter-golf-hy}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
VARIANT="${VARIANT:-sp1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT_DIR/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/.cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"

if [[ -d /workspace && "$ROOT_DIR" != "/workspace/$REPO_NAME" ]]; then
    echo "Remote root must be /workspace/$REPO_NAME, got $ROOT_DIR" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found at $PYTHON_BIN"
    echo "Run: bash setup.sh"
    exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp"
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
