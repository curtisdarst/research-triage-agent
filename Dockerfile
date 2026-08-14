# Cloud Run image for the web front end (Tier 2). Minimal: Python slim base,
# pinned requirements, uvicorn serving the FastAPI app in web/main.py.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ agent/
COPY ingest/ ingest/
COPY run_demo.py .
COPY web/ web/

# Cloud Run sets $PORT; default 8080 for local `docker run`.
ENV PORT=8080
CMD exec uvicorn web.main:app --host 0.0.0.0 --port ${PORT}
