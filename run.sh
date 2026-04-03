#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT_DIR/.venv/bin/torchrun}"
NUM_GPUS="${NUM_GPUS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$NUM_GPUS}"
VARIANT="${VARIANT:-sp1024}"
VOCAB_SIZE="${VOCAB_SIZE:-$(printf '%s\n' "$VARIANT" | tr -cd '0-9')}"

if [[ -z "$VOCAB_SIZE" ]]; then
    echo "Unable to infer VOCAB_SIZE from VARIANT=$VARIANT"
    echo "Set VOCAB_SIZE explicitly."
    exit 1
fi

DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/datasets/fineweb10B_${VARIANT}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/data/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model}"
RUN_ID="${RUN_ID:-baseline_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "torchrun not found at $TORCHRUN_BIN"
    echo "Run: bash setup.sh"
    exit 1
fi

if [[ ! -d "$DATA_PATH" ]]; then
    echo "Dataset path missing: $DATA_PATH"
    echo "Run: bash down.sh"
    exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
    echo "Tokenizer path missing: $TOKENIZER_PATH"
    echo "Run: bash down.sh"
    exit 1
fi

cd "$ROOT_DIR"
echo "run_id=$RUN_ID"
echo "variant=$VARIANT vocab_size=$VOCAB_SIZE nproc_per_node=$NPROC_PER_NODE"
echo "data_path=$DATA_PATH"
echo "tokenizer_path=$TOKENIZER_PATH"

export RUN_ID
export DATA_PATH
export TOKENIZER_PATH
export VOCAB_SIZE
export MASTER_PORT

exec "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py "$@"
