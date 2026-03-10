FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py ./
COPY world_models ./world_models
COPY torchwm_ui ./torchwm_ui

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

RUN cd torchwm_ui && npm install

EXPOSE 8000

CMD ["uvicorn", "world_models.ui.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
