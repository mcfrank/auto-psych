# Cloud Run Job: auto-psych pipeline with Firestore sync.
# Build: docker build -t auto-psych-job .
# Project state is synced from Firestore at runtime; do not copy projects/ into image.

FROM python:3.11-slim

WORKDIR /app

# Install system deps for Playwright (optional but needed for collect with simulated_participants)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright browsers for collect step (simulated participants)
RUN playwright install chromium && playwright install-deps chromium || true

COPY src/ src/
COPY prompts/ prompts/
COPY templates/ templates/
COPY run_pipeline.py .
COPY scripts/ scripts/

# No projects/ in image; state comes from Firestore (sync in entrypoint)
ENV PIPELINE_PROJECTS_DIR=/app/projects

ENTRYPOINT ["python3", "scripts/cloud_run_entrypoint.py"]
