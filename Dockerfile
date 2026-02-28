FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for pytesseract OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files first for better layer caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev deps, no virtualenv in container)
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --without dev

# Copy application source
COPY config/ config/
COPY src/ src/

# Create data directory for persistent storage
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
