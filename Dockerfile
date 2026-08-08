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


# ---------- Stage 2: build the Rust solar monitor ----------
FROM rust:1-bookworm AS solar-monitor-build
WORKDIR /build

COPY vendor/ ./vendor/
COPY critical-learning/ ./critical-learning/
COPY solar-monitor/ ./solar-monitor/

RUN cargo build --release --manifest-path solar-monitor/Cargo.toml --bin solar-monitor


# ---------- Stage 3: Python API + static frontend + solar monitor ----------
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

# Rust solar monitor used by /api/solar/* proxy endpoints.
COPY --from=solar-monitor-build /build/solar-monitor/target/release/solar-monitor /usr/local/bin/solar-monitor

# JSON data files the backend reads directly at runtime
# (server.py reads frontend/src/plates.json and frontend/assets/bronze_age_field.json)
COPY frontend/src/plates.json ./frontend/src/plates.json
COPY frontend/assets/bronze_age_field.json ./frontend/assets/bronze_age_field.json

# data/ is provided at runtime via a volume (see docker-compose.yml)

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/plates" >/dev/null \
      && curl -fsS "http://127.0.0.1:${PORT}/api/solar/live" >/dev/null || exit 1

# server.py lives in backend/ and reads data/ + frontend/dist relative to its parent
WORKDIR /app/backend
RUN chmod +x entrypoint.sh
# Refreshes cosmic-ray cache from NMDB, then supervises Rust + uvicorn
CMD ["./entrypoint.sh"]
