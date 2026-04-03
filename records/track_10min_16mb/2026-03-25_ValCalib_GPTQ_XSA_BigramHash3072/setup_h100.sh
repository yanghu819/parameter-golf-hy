#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_NAME="${REPO_NAME:-parameter-golf-hy}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-h100}"
LOCAL_BIN_DIR="${LOCAL_BIN_DIR:-$REPO_ROOT/.local/bin}"
UV_BIN="${UV_BIN:-$LOCAL_BIN_DIR/uv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.9.1}"
FLASH_ATTN_LINKS="${FLASH_ATTN_LINKS:-https://windreamer.github.io/flash-attention3-wheels/cu128_torch291}"
INSTALL_FA3="${INSTALL_FA3:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_ROOT/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/.cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-$REPO_ROOT/.tmp}"

ensure_uv() {
    if [[ -x "$UV_BIN" ]]; then
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return
    fi
    mkdir -p "$LOCAL_BIN_DIR"
    UV_INSTALL_DIR="$REPO_ROOT/.local" curl -LsSf https://astral.sh/uv/install.sh | sh
    if [[ -x "$LOCAL_BIN_DIR/uv" ]]; then
        UV_BIN="$LOCAL_BIN_DIR/uv"
        return
    fi
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        cp "$HOME/.local/bin/uv" "$LOCAL_BIN_DIR/uv"
        UV_BIN="$LOCAL_BIN_DIR/uv"
        return
    fi
    echo "uv install failed" >&2
    exit 1
}

main() {
    if [[ -d /workspace && "$REPO_ROOT" != "/workspace/$REPO_NAME" ]]; then
        echo "Remote root must be /workspace/$REPO_NAME, got $REPO_ROOT" >&2
        exit 1
    fi

    mkdir -p "$REPO_ROOT/.cache" "$REPO_ROOT/.tmp" "$LOCAL_BIN_DIR"
    ensure_uv

    cd "$REPO_ROOT"
    "$UV_BIN" venv --clear --python "$PYTHON_BIN" "$VENV_DIR"
    "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
        numpy \
        -r "$SCRIPT_DIR/requirements.txt" \
        huggingface_hub
    "$UV_BIN" pip install --python "$VENV_DIR/bin/python" --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
    if [[ "$INSTALL_FA3" == "1" ]]; then
        "$UV_BIN" pip install --python "$VENV_DIR/bin/python" flash_attn_3 --find-links "$FLASH_ATTN_LINKS"
    fi

    "$VENV_DIR/bin/python" - <<'PY'
import torch
import sentencepiece
import zstandard
try:
    from flash_attn_interface import flash_attn_func
    fa3_available = bool(flash_attn_func)
except ImportError:
    fa3_available = False

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_count={torch.cuda.device_count()}")
print(f"fa3_available={fa3_available}")
PY
}

main "$@"
