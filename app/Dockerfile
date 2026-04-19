# Shiny for Python on Cloud Run (listens on $PORT)
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run sets PORT; default 8080 for local docker run
ENV PORT=8080
EXPOSE 8080

# Prefer metadata credentials on Cloud Run (ignore empty key path if present)
ENTRYPOINT ["/bin/sh", "-c", "unset GOOGLE_APPLICATION_CREDENTIALS; exec shiny run app.py --host 0.0.0.0 --port ${PORT:-8080}"]
