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
: "${SEARCH_PRESET:=slot-coarse}"
: "${SEARCH_TAG:=$(date -u +%Y-%m-%dT%H%M%SZ)}"

RUN_DIR="$OUT_ROOT/${SEARCH_TAG}_${SEARCH_PRESET}_docs${TTT_PROXY_DOCS}"
mkdir -p "$RUN_DIR"

COMMON_ENV=(
  DATA_PATH="$DATA_PATH"
  TOKENIZER_PATH="$TOKENIZER_PATH"
  CHECKPOINT_PATH="$CHECKPOINT_PATH"
  BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
  BIGRAM_DIM="${BIGRAM_DIM:-112}"
  TTT_MODE=slot_delta
  TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-32}"
  TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
  TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-16}"
  TTT_MAX_DOCS="$TTT_PROXY_DOCS"
  ALLOW_FA3_FALLBACK="${ALLOW_FA3_FALLBACK:-1}"
)

config_args() {
  local name="$1"
  case "$name" in
    baseline_steps0)
      printf '%s\n' TTT_SLOT_STEPS=0 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=2
      ;;
    adamw_best_lr2e4)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0002 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=1
      ;;
    adamw_best_lr3e4)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=1
      ;;
    adamw_best_lr5e4)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0005 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=1
      ;;
    adamw_gate2_lr3e4)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=2
      ;;
    adamw_gate2_lr5e4)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0005 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=2
      ;;
    adamw_best_thr320)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=1 TTT_SLOT_UPDATE_LOSS_THRESHOLD=3.20
      ;;
    adamw_best_thr330)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=1 TTT_SLOT_UPDATE_LOSS_THRESHOLD=3.30
      ;;
    adamw_best_thr340)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=1 TTT_SLOT_UPDATE_LOSS_THRESHOLD=3.40
      ;;
    adamw_gate2_thr320)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=2 TTT_SLOT_UPDATE_LOSS_THRESHOLD=3.20
      ;;
    adamw_gate2_thr330)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=2 TTT_SLOT_UPDATE_LOSS_THRESHOLD=3.30
      ;;
    adamw_firstk2)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=1 TTT_SLOT_UPDATE_FIRST_K_CHUNKS=2
      ;;
    adamw_firstk4)
      printf '%s\n' TTT_SLOT_STEPS=1 TTT_SLOT_ADAPT_BLOCK=1 TTT_SLOT_LR=0.0003 TTT_SLOT_UPDATE=adamw TTT_SLOT_UPDATE_EVERY=1 TTT_SLOT_UPDATE_FIRST_K_CHUNKS=4
      ;;
    *)
      echo "Unknown config name: $name" >&2
      exit 1
      ;;
  esac
}

preset_configs() {
  local preset="$1"
  case "$preset" in
    slot-coarse)
      printf '%s\n' \
        baseline_steps0 \
        adamw_best_lr2e4 \
        adamw_best_lr3e4 \
        adamw_best_lr5e4 \
        adamw_gate2_lr3e4 \
        adamw_gate2_lr5e4 \
        adamw_best_thr320 \
        adamw_best_thr330 \
        adamw_gate2_thr320 \
        adamw_gate2_thr330
      ;;
    slot-sparse)
      printf '%s\n' \
        baseline_steps0 \
        adamw_best_lr3e4 \
        adamw_gate2_lr3e4 \
        adamw_best_thr320 \
        adamw_best_thr330 \
        adamw_best_thr340 \
        adamw_firstk2 \
        adamw_firstk4
      ;;
    *)
      echo "Unknown SEARCH_PRESET=$preset" >&2
      exit 1
      ;;
  esac
}

run_eval() {
  local name="$1"
  shift
  local log="$RUN_DIR/${name}.log"
  if [[ -f "$log" ]]; then
    echo "skip_existing:$name"
    return
  fi
  echo "=== $name ===" | tee "$log"
  env \
    "${COMMON_ENV[@]}" \
    "$@" \
    "$PYTHON_BIN" "$RECORD_DIR/eval_train_free_checkpoint.py" 2>&1 | tee -a "$log"
  "$PYTHON_BIN" "$RECORD_DIR/summarize_train_free_logs.py" "$RUN_DIR" > "$RUN_DIR/summary.tsv"
}

CONFIGS=()
if [[ -n "${RUN_CONFIGS:-}" ]]; then
  IFS=',' read -r -a CONFIGS <<< "$RUN_CONFIGS"
else
  while IFS= read -r line; do
    [[ -n "$line" ]] && CONFIGS+=("$line")
  done < <(preset_configs "$SEARCH_PRESET")
fi

for name in "${CONFIGS[@]}"; do
  mapfile -t args < <(config_args "$name")
  run_eval "$name" "${args[@]}"
done

"$PYTHON_BIN" "$RECORD_DIR/summarize_train_free_logs.py" "$RUN_DIR" > "$RUN_DIR/summary.tsv"
cat "$RUN_DIR/summary.tsv"
echo "summary:$RUN_DIR/summary.tsv"
