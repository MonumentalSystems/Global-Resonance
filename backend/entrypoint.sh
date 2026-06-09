#!/bin/sh
# Container entrypoint: best-effort refresh of file-backed data layers, then serve.
set -e

PORT="${PORT:-8000}"

# Populate cosmic-ray cache from NMDB (live, public). Non-fatal: the live
# endpoints stream regardless, and /api/cosmic_rays degrades gracefully.
echo "[entrypoint] refreshing cached data layers..."
python refresh_data.py || echo "[entrypoint] data refresh skipped (non-fatal)"

echo "[entrypoint] starting uvicorn on 0.0.0.0:${PORT}"
exec uvicorn server:app --host 0.0.0.0 --port "${PORT}"
