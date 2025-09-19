# Minimal, reproducible trainer image (Python 3.11 + wheels-friendly deps)
FROM python:3.11-slim

# Optional: faster installs & predictable locale
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# System deps that occasionally help with wheels; slim but safe
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better build caching
COPY trainer/requirements.txt /app/trainer/requirements.txt
RUN python -m pip install -U pip setuptools wheel && \
    pip install -r /app/trainer/requirements.txt

# Copy the rest of the repo (trainer scripts + site for artifact output)
COPY . /app

# Default command drops you into a shell; we’ll pass run cmds via `docker run …`
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
