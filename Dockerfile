# syntax=docker/dockerfile:1

# ---------- Stage 1: build the Vite frontend ----------
FROM node:20-slim AS frontend-build
WORKDIR /build

# Install deps first for better layer caching
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build the static bundle -> /build/dist
COPY frontend/ ./
RUN npm run build


# ---------- Stage 2: Python API + static frontend ----------
FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# System libs occasionally needed by numpy/pandas wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Backend source
COPY backend/ ./backend/

# Built frontend from stage 1 -> /app/frontend/dist (matches server.py mount logic)
COPY --from=frontend-build /build/dist ./frontend/dist

# JSON data files the backend reads directly at runtime
# (server.py reads frontend/src/plates.json and frontend/assets/bronze_age_field.json)
COPY frontend/src/plates.json ./frontend/src/plates.json
COPY frontend/assets/bronze_age_field.json ./frontend/assets/bronze_age_field.json

# data/ is provided at runtime via a volume (see docker-compose.yml)

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/plates" || exit 1

# server.py lives in backend/ and reads data/ + frontend/dist relative to its parent
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
