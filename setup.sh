#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="${REPO_NAME:-parameter-golf-hy}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
SYSTEM_PYTHON_BIN="${SYSTEM_PYTHON_BIN:-python3}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-auto}"
LOCAL_BIN_DIR="${LOCAL_BIN_DIR:-$ROOT_DIR/.local/bin}"
UV_BIN="${UV_BIN:-$LOCAL_BIN_DIR/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT_DIR/.cache/uv/python}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT_DIR/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/.cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"

if [[ "$(uname -s)" == "Darwin" ]]; then
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
else
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
fi

ensure_uv() {
    if [[ -x "$UV_BIN" ]]; then
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return
    fi

    echo "uv not found; installing to $LOCAL_BIN_DIR"
    mkdir -p "$LOCAL_BIN_DIR"
    UV_INSTALL_DIR="$ROOT_DIR/.local" curl -LsSf https://astral.sh/uv/install.sh | sh
    if [[ -x "$LOCAL_BIN_DIR/uv" ]]; then
        UV_BIN="$LOCAL_BIN_DIR/uv"
        return
    fi
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return
    fi
    echo "uv install succeeded but uv binary was not found" >&2
    exit 1
}

ensure_torch() {
    local python_bin="$1"
    if "$python_bin" -c "import torch" >/dev/null 2>&1; then
        return
    fi

    echo "Installing torch from ${TORCH_INDEX_URL}"
    "$UV_BIN" pip install --python "$python_bin" --index-url "$TORCH_INDEX_URL" torch
}

main() {
    if [[ -d /workspace && "$ROOT_DIR" != "/workspace/$REPO_NAME" ]]; then
        echo "Remote root must be /workspace/$REPO_NAME, got $ROOT_DIR" >&2
        exit 1
    fi

    mkdir -p "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp" "$LOCAL_BIN_DIR"
    ensure_uv

    cd "$ROOT_DIR"
    local venv_args=()
    local venv_python="$PYTHON_VERSION"
    if [[ "$USE_SYSTEM_SITE_PACKAGES" == "1" ]]; then
        venv_args+=(--system-site-packages)
    elif [[ "$USE_SYSTEM_SITE_PACKAGES" == "auto" ]] && "$SYSTEM_PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
        venv_args+=(--system-site-packages)
        venv_python="$SYSTEM_PYTHON_BIN"
        echo "Reusing system torch via --system-site-packages and $SYSTEM_PYTHON_BIN"
    else
        "$UV_BIN" python install "$PYTHON_VERSION"
    fi

    echo "Using Python ${venv_python}"
    "$UV_BIN" venv --clear --python "$venv_python" "${venv_args[@]}" "$VENV_DIR"
    "$UV_BIN" sync --frozen --python "$VENV_DIR/bin/python"
    ensure_torch "$VENV_DIR/bin/python"

    "$VENV_DIR/bin/python" - <<'PY'
import platform
import sys

import numpy
import torch

print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
print(f"numpy={numpy.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"gpu[{index}]={props.name} vram_gb={props.total_memory // 1024**3}")
PY

    cat <<EOF

Environment ready.
Activate with:
  source "$VENV_DIR/bin/activate"
EOF
}

main "$@"
