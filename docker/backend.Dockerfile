FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SUMO_HOME=/usr/share/sumo

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgdal-dev \
        sumo \
        sumo-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install \
        fastapi \
        "uvicorn[standard]" \
        websockets \
        redis \
        celery \
        stable-baselines3 \
        torch \
        gymnasium \
        sumo-rl \
        traci \
        osmnx \
        networkx \
        psycopg2-binary \
        pandas \
        numpy \
        matplotlib

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
