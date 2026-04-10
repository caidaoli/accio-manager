FROM python:3.12-slim

ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION} \
    ACCIO_CALLBACK_HOST=127.0.0.1 \
    ACCIO_SERVER_HOST=0.0.0.0 \
    ACCIO_CALLBACK_PORT=4097 \
    ACCIO_AUTO_OPEN_BROWSER=false \
    ACCIO_DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md main.py ./
COPY .env.example ./
COPY accio_panel ./accio_panel
COPY data/config.example.json ./data/config.example.json

RUN pip install --upgrade pip \
    && pip install .

RUN mkdir -p /app/data/accounts \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 4097

VOLUME ["/app/data"]

CMD ["python", "main.py"]
