#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
UV_BIN="${UV_BIN:-uv}"

ensure_uv() {
    if command -v "$UV_BIN" >/dev/null 2>&1; then
        return
    fi

    echo "uv not found; installing to \$HOME/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    UV_BIN="uv"
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
    ensure_uv

    cd "$ROOT_DIR"
    echo "Using Python ${PYTHON_VERSION}"
    "$UV_BIN" python install "$PYTHON_VERSION"
    "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
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
