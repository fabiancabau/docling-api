# ---- Build Stage ----
    FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

    WORKDIR /app
    
    # Install runtime deps needed during build
    RUN apt-get update && \
        apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
        rm -rf /var/lib/apt/lists/*
    
    # Copy project files
    COPY pyproject.toml uv.lock ./
    RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
    
    # Copy rest of the app
    COPY . .
    
    # Optional: pre-download models
    RUN uv run python -c "from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline; StandardPdfPipeline.download_models_hf(force=True)"
    
    # ---- Final Stage ----
    FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
    
    WORKDIR /app
    
    # Install runtime deps
    RUN apt-get update && \
        apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl unzip && \
        rm -rf /var/lib/apt/lists/*
    
    # Create user
    RUN useradd --create-home app && \
        mkdir -p /app && chown -R app:app /app
    
    # Copy environment + code from builder
    COPY --from=builder /app /app
    COPY --from=builder /app/.venv /app/.venv
    ENV PATH="/app/.venv/bin:$PATH"
    
    # Optional: EasyOCR models
    RUN wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip && \
        wget https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip && \
        mkdir -p /home/app/.EasyOCR/model && \
        unzip english_g2.zip -d /home/app/.EasyOCR/model && \
        unzip craft_mlt_25k.zip -d /home/app/.EasyOCR/model
    
    USER app
    
    EXPOSE 8001
    CMD ["uvicorn", "main:app", "--port", "8001", "--host", "0.0.0.0", "--proxy-headers"]