FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_DIR=/opt/pg-h100-venv
ENV PATH=/opt/pg-h100-venv/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV FLASH_ATTN_LINKS=https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VENV_DIR}" \
    && "${VENV_DIR}/bin/pip" install --upgrade pip \
    && "${VENV_DIR}/bin/pip" install \
        numpy \
        sentencepiece \
        zstandard \
        huggingface_hub \
    && "${VENV_DIR}/bin/pip" install --index-url "${TORCH_INDEX_URL}" "torch==2.9.1" \
    && "${VENV_DIR}/bin/pip" install --find-links "${FLASH_ATTN_LINKS}" flash_attn_3

RUN "${VENV_DIR}/bin/python" - <<'PY'
import torch
import sentencepiece
import zstandard
from flash_attn_interface import flash_attn_func

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"flash_attn_3={bool(flash_attn_func)}")
PY
