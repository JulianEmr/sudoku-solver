# ── Sudoku Solver – Docker image ────────────────────────────────────────────
# Scope: /solve (image → solution) and /train (YOLO fine-tuning) only.
# Browser automation, compare, and debug tools run locally on the host.
FROM python:3.13-slim

# ── System dependencies ──────────────────────────────────────────────────────
# - tesseract-ocr        : OCR digit recognition method
# - libgl1 / libglib2.0  : OpenCV runtime libs
# - libgomp1             : OpenMP used by ultralytics/YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv (project package manager) ────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ── Copy dependency manifests first (layer-cache friendly) ──────────────────
COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev --no-group browser --no-group local --frozen --no-install-project

# ── Copy the rest of the project ─────────────────────────────────────────────
COPY src/    ./src/
COPY data/   ./data/
COPY runs/   ./runs/
COPY api.py  ./

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ── Default command ──────────────────────────────────────────────────────────
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
