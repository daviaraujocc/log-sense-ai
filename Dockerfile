FROM python:3.11-slim AS compile-image


RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    g++ \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS build-image
COPY --from=compile-image /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY log_sense.py .
COPY utils.py .
COPY cli.py .

ENTRYPOINT ["python", "cli.py"]