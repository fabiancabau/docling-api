FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ARG CPU_ONLY=false

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install wget -y && \
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

# Copy dependency files and README
COPY pyproject.toml uv.lock README.md ./

# Install dependencies but not the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy the rest of the project
COPY . .

# Better GPU detection: Check both architecture and if NVIDIA is available
RUN ARCH=$(uname -m) && \
    if [ "$CPU_ONLY" = "true" ] || [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ] || ! command -v nvidia-smi >/dev/null 2>&1; then \
    USE_GPU=false; \
    else \
    USE_GPU=true; \
    fi && \
    echo "Detected GPU availability: $USE_GPU" && \
    # For PyTorch installation with architecture detection
    uv pip uninstall -y torch torchvision torchaudio || true && \
    if [ "$USE_GPU" = "false" ]; then \
    # For CPU or ARM architectures or no NVIDIA
    echo "Installing PyTorch for CPU" && \
    uv pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
    # For x86_64 with GPU support
    echo "Installing PyTorch with CUDA support" && \
    uv pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu181; \
    fi

# Install the project in non-editable mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

# Download models for the pipeline
RUN uv run python -c "from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline; artifacts_path = StandardPdfPipeline.download_models_hf(force=True)"

# Pre-download EasyOCR models with better GPU detection
# RUN ARCH=$(uname -m) && \
#     if [ "$CPU_ONLY" = "true" ] || [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ] || ! command -v nvidia-smi >/dev/null 2>&1; then \
#     echo "Downloading EasyOCR models for CPU" && \
#     uv run python -c "import easyocr; reader = easyocr.Reader(['en'], gpu=False); print('EasyOCR CPU models downloaded successfully')"; \
#     else \
#     echo "Downloading EasyOCR models with GPU support" && \
#     uv run python -c "import easyocr; reader = easyocr.Reader(['en'], gpu=True); print('EasyOCR GPU models downloaded successfully')"; \
#     fi




# Production stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends redis-server libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=4 \
    UV_COMPILE_BYTECODE=1

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Create a non-root user
RUN useradd --create-home app && \
    mkdir -p /app && \
    chown -R app:app /app /tmp

# Copy the virtual environment from the builder stage
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy necessary files for the application
COPY --chown=app:app . .

RUN wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip
RUN wget https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip
RUN mkdir ~/.EasyOCR
RUN mkdir ~/.EasyOCR/model
RUN unzip english_g2.zip -d ~/.EasyOCR/model
RUN unzip craft_mlt_25k.zip -d ~/.EasyOCR/model

# Switch to non-root user
USER app

EXPOSE 8080
CMD ["uvicorn", "main:app", "--port", "8080", "--host", "0.0.0.0"]
