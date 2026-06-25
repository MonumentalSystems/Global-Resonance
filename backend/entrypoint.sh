#!/bin/sh
# Container entrypoint: best-effort refresh of file-backed data layers, then serve.
set -e

PORT="${PORT:-8000}"
SOLAR_MONITOR_PORT="${SOLAR_MONITOR_PORT:-8089}"
SOLAR_MONITOR_POLL_INTERVAL="${SOLAR_MONITOR_POLL_INTERVAL:-60}"

if [ -z "${SOLAR_MONITOR_URL:-}" ]; then
  export SOLAR_MONITOR_URL="http://127.0.0.1:${SOLAR_MONITOR_PORT}"
  START_LOCAL_SOLAR_MONITOR="${START_LOCAL_SOLAR_MONITOR:-1}"
else
  START_LOCAL_SOLAR_MONITOR="${START_LOCAL_SOLAR_MONITOR:-0}"
fi

if [ "${START_LOCAL_SOLAR_MONITOR}" != "0" ] && command -v solar-monitor >/dev/null 2>&1; then
  echo "[entrypoint] starting solar-monitor on 127.0.0.1:${SOLAR_MONITOR_PORT}"
  solar-monitor --port "${SOLAR_MONITOR_PORT}" --poll-interval "${SOLAR_MONITOR_POLL_INTERVAL}" &
  SOLAR_MONITOR_PID="$!"
  trap 'kill "${SOLAR_MONITOR_PID}" 2>/dev/null || true' INT TERM EXIT
else
  echo "[entrypoint] /api/solar/* will use ${SOLAR_MONITOR_URL}"
fi

# Populate cosmic-ray cache from NMDB (live, public). Non-fatal: the live
# endpoints stream regardless, and /api/cosmic_rays degrades gracefully.
echo "[entrypoint] refreshing cached data layers..."
python refresh_data.py || echo "[entrypoint] data refresh skipped (non-fatal)"

echo "[entrypoint] starting uvicorn on 0.0.0.0:${PORT}"
exec uvicorn server:app --host 0.0.0.0 --port "${PORT}"
