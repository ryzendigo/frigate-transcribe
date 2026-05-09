FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates tini procps libsndfile1 sqlite3 build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch wheel first (much smaller than CUDA default) — needed by resemblyzer
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt .
RUN pip install -r requirements.txt
# webrtcvad has no prebuilt wheels — provide a built one via webrtcvad-wheels and
# resemblyzer on top. resemblyzer's stale setup.py doesn't play well with strict
# numpy pins so we install with --no-deps and rely on requirements.txt above.
RUN pip install webrtcvad-wheels==2.0.14 \
    && pip install --no-deps resemblyzer==0.1.4 \
    && pip install librosa==0.10.2

COPY app/ /app/

RUN mkdir -p /data/logs /data/summaries /data/chunks /data/audio /models

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "/app/main.py"]
