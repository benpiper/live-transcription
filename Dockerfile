FROM nvidia/cuda:12.1.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 transcriber && \
    mkdir -p /app /data/sessions /data/voice_profiles /data/pretrained_models && \
    chown -R transcriber:transcriber /app /data

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir fastapi uvicorn[standard]

# Copy application
COPY --chown=transcriber:transcriber . .

USER transcriber

VOLUME ["/data/sessions", "/data/voice_profiles", "/data/pretrained_models", "/root/.cache/huggingface"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/ping').raise_for_status()" || exit 1

CMD ["python3", "real_time_transcription.py", "--diarize", "--web", "--config", "config.json"]
