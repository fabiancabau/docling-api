# Use a base image with Python
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy UV from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Enable bytecode compilation and set link mode for better performance
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/tmp/ \
    TORCH_HOME=/tmp/ \
    OMP_NUM_THREADS=4

WORKDIR /app

# Create a minimal README.md to satisfy the build requirements
RUN echo "# Docling API" > README.md

# Install dependencies first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies with caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Install PyTorch separately based on CPU_ONLY flag
ARG CPU_ONLY=false
RUN if [ "$CPU_ONLY" = "true" ]; then \
    uv pip install --system torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
    uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    fi

# Install project dependencies for model downloads
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system docling
RUN python -c 'from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline; artifacts_path = StandardPdfPipeline.download_models_hf(force=True);'

RUN uv pip install --system easyocr

# Pre-download EasyOCR models in compatible groups
RUN python -c 'import easyocr; \
    reader = easyocr.Reader(["fr", "de", "es", "en", "it", "pt"], gpu=True); \
    print("EasyOCR models downloaded successfully")'

RUN uv pip install sentence-transformers

RUN uv pip install "chonkie[semantic, model2vec]"

# Download Chonkie models (using Model2Vec for better performance)
RUN python -c 'from chonkie import SDPMChunker; \
    sdpm_chunker = SDPMChunker(embedding_model="minishlab/potion-base-8M"); \
    print("Chonkie models downloaded successfully")'

# Copy the application code
COPY . .

# Final sync to ensure everything is properly installed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

EXPOSE 8080

# Use UV to run the application
CMD ["uv", "run", "uvicorn", "--port", "8080", "--host", "0.0.0.0", "main:app"]