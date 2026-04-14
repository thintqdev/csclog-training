FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch-geometric

COPY conf/ conf/
COPY src/ src/

EXPOSE 8000
