FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY digest ./digest

RUN useradd --create-home --uid 1000 digest && mkdir -p /data && chown digest /data
USER digest
ENV DATA_DIR=/data

CMD ["python", "-m", "digest", "schedule"]
