#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
: "${OUT_ROOT:=results/train_free_proxy}"
: "${PYTHON_BIN:=/workspace/parameter-golf-hy/.venv-h100/bin/python}"
: "${TTT_PROXY_DOCS:=32}"
: "${DATA_PATH:=$ROOT/data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=$ROOT/data/tokenizers/fineweb_1024_bpe.model}"

mkdir -p "$OUT_ROOT"

run_eval() {
  local name="$1"
  shift
  local log="$OUT_ROOT/${name}.log"
  echo "=== $name ===" | tee "$log"
  env \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    CHECKPOINT_PATH="$CHECKPOINT_PATH" \
    BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}" \
    BIGRAM_DIM="${BIGRAM_DIM:-112}" \
    TTT_MODE=slot_delta \
    TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-32}" \
    TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}" \
    TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-16}" \
    TTT_MAX_DOCS="$TTT_PROXY_DOCS" \
    ALLOW_FA3_FALLBACK="${ALLOW_FA3_FALLBACK:-1}" \
    "$@" \
    "$PYTHON_BIN" "$RECORD_DIR/eval_train_free_checkpoint.py" 2>&1 | tee -a "$log"
}

run_eval baseline_steps0 \
  TTT_SLOT_STEPS=0 \
  TTT_SLOT_ADAPT_BLOCK=1 \
  TTT_SLOT_LR=0.0003 \
  TTT_SLOT_UPDATE_EVERY=2

run_eval adamw_best_like \
  TTT_SLOT_STEPS=1 \
  TTT_SLOT_ADAPT_BLOCK=1 \
  TTT_SLOT_LR=0.0003 \
  TTT_SLOT_UPDATE=adamw \
  TTT_SLOT_UPDATE_EVERY=2

run_eval adamw_lr5e4 \
  TTT_SLOT_STEPS=1 \
  TTT_SLOT_ADAPT_BLOCK=1 \
  TTT_SLOT_LR=0.0005 \
  TTT_SLOT_UPDATE=adamw \
  TTT_SLOT_UPDATE_EVERY=2

run_eval adamw_every1 \
  TTT_SLOT_STEPS=1 \
  TTT_SLOT_ADAPT_BLOCK=1 \
  TTT_SLOT_LR=0.0003 \
  TTT_SLOT_UPDATE=adamw \
  TTT_SLOT_UPDATE_EVERY=1

run_eval adamw_block2 \
  TTT_SLOT_STEPS=1 \
  TTT_SLOT_ADAPT_BLOCK=2 \
  TTT_SLOT_LR=0.0003 \
  TTT_SLOT_UPDATE=adamw \
  TTT_SLOT_UPDATE_EVERY=2
