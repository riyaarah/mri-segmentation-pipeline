# Dockerfile for MRI Brain Tumor Segmentation API
# Deployed on Hugging Face Spaces (Docker SDK)
# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# System dependencies needed by nibabel for medical image I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as non-root user (avoids permission issues)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies first (better layer caching on rebuilds)
COPY --chown=user requirements-api.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code (model checkpoint is downloaded from the HF
# Model repo at container startup — see api/model.py — so it's not
# baked into this image, keeping the image lean and decoupling
# model updates from app redeploys)
COPY --chown=user api/ ./api/

# Hugging Face Spaces requires the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
