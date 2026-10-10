FROM python:3.12-slim

# webrtcvad (pulled in by resemblyzer) builds from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, so the [ml] extra doesn't pull the multi-GB CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY recall ./recall
RUN pip install --no-cache-dir ".[ml]"

ENV RECALL_DB_PATH=/data/recall.db
VOLUME /data
EXPOSE 8000

# One worker: conversation/face state lives in-process.
CMD ["uvicorn", "recall.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
