FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ARG CPU_ONLY=false

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation and set proper link mode for cache mounting
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=4

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN pip install easyocr

# Pre-download EasyOCR models with better GPU detection
RUN python -c "import easyocr; reader = easyocr.Reader(['en'], gpu=True); print('EasyOCR GPU models downloaded successfully')"

