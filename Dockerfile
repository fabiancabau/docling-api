FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl wget git procps \
    && rm -rf /var/lib/apt/lists/*

# Copy UV from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/tmp/ \
    TORCH_HOME=/tmp/ \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN echo "# Docling API" > README.md

# Install dependencies first (for better layer caching)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

ARG CPU_ONLY=false
RUN if [ "$CPU_ONLY" = "true" ]; then \
    uv pip install --system --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
    uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    fi

# Install required packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir docling easyocr "chonkie[semantic]"

# Download models in a single step
RUN python -c 'from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline; \
    from chonkie import SDPMChunker; \
    from easyocr import Reader; \
    artifacts_path = StandardPdfPipeline.download_models_hf(force=True); \
    sdpm_chunker = SDPMChunker(embedding_model="minishlab/potion-base-8M"); \
    reader = Reader(["fr", "de", "es", "en", "it", "pt"], gpu=True); \
    print("Models downloaded successfully")'

# Copy the application code
COPY . .

# Final dependency sync
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Remove cache to save space
RUN rm -rf /root/.cache/uv

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "--port", "8080", "--host", "0.0.0.0", "main:app"]
