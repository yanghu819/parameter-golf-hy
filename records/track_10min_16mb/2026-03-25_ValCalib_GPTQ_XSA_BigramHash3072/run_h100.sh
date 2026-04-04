#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_NAME="${REPO_NAME:-parameter-golf-hy}"
MODE="${MODE:-smoke}"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv-h100/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
VARIANT="${VARIANT:-sp1024}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_${VARIANT}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model}"
MASTER_PORT="${MASTER_PORT:-29500}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/.cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"

if [[ -d /workspace && "$REPO_ROOT" != "/workspace/$REPO_NAME" ]]; then
    echo "Remote root must be /workspace/$REPO_NAME, got $REPO_ROOT" >&2
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Missing python environment at $VENV_PYTHON" >&2
    echo "Run: bash $SCRIPT_DIR/setup_h100.sh" >&2
    exit 1
fi

if [[ ! -d "$DATA_PATH" ]]; then
    echo "Missing dataset at $DATA_PATH" >&2
    echo "Run: TRAIN_SHARDS=80 bash $REPO_ROOT/down.sh" >&2
    exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
    echo "Missing tokenizer at $TOKENIZER_PATH" >&2
    echo "Run: TRAIN_SHARDS=80 bash $REPO_ROOT/down.sh" >&2
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/.cache" "$REPO_ROOT/.tmp" "$REPO_ROOT/logs"

export DATA_PATH
export TOKENIZER_PATH
export VOCAB_SIZE
export MASTER_PORT

case "$MODE" in
    smoke)
        export SMOKE_TEST="${SMOKE_TEST:-1}"
        export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"
        export ALLOW_FA3_FALLBACK="${ALLOW_FA3_FALLBACK:-1}"
        export ITERATIONS="${ITERATIONS:-6}"
        export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"
        export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
        export WARMUP_STEPS="${WARMUP_STEPS:-0}"
        export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-45}"
        export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
        export RUN_ID="${RUN_ID:-record_best_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
        ;;
    preflight)
        export SMOKE_TEST="${SMOKE_TEST:-1}"
        export DISABLE_COMPILE="${DISABLE_COMPILE:-0}"
        export ALLOW_FA3_FALLBACK="${ALLOW_FA3_FALLBACK:-0}"
        export ITERATIONS="${ITERATIONS:-1}"
        export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"
        export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
        export WARMUP_STEPS="${WARMUP_STEPS:-0}"
        export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-90}"
        export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-16384}"
        export RUN_ID="${RUN_ID:-record_best_preflight_$(date -u +%Y%m%dT%H%M%SZ)}"
        ;;
    formal)
        export SMOKE_TEST="${SMOKE_TEST:-0}"
        export DISABLE_COMPILE="${DISABLE_COMPILE:-0}"
        export ALLOW_FA3_FALLBACK="${ALLOW_FA3_FALLBACK:-0}"
        export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
        export BIGRAM_DIM="${BIGRAM_DIM:-112}"
        export WARMDOWN_ITERS="${WARMDOWN_ITERS:-4000}"
        export TARGET_MB="${TARGET_MB:-15.9}"
        export SEED="${SEED:-314}"
        export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
        export RUN_ID="${RUN_ID:-record_best_formal_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
        ;;
    *)
        echo "MODE must be one of: smoke, preflight, formal" >&2
        exit 1
        ;;
esac

echo "mode=$MODE"
echo "run_id=$RUN_ID"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "data_path=$DATA_PATH"
echo "tokenizer_path=$TOKENIZER_PATH"
echo "allow_fa3_fallback=$ALLOW_FA3_FALLBACK"
echo "disable_compile=$DISABLE_COMPILE"

exec "$VENV_PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$SCRIPT_DIR/train_gpt.py" \
    "$@"
