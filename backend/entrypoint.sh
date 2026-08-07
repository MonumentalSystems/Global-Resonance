#!/bin/sh
# Container entrypoint: best-effort refresh of file-backed data layers, then serve.
set -e

PORT="${PORT:-8000}"
SOLAR_MONITOR_PORT="${SOLAR_MONITOR_PORT:-8089}"
SOLAR_MONITOR_POLL_INTERVAL="${SOLAR_MONITOR_POLL_INTERVAL:-60}"
SOLAR_MONITOR_PID=""
API_PID=""

shutdown() {
  trap - INT TERM
  [ -z "${API_PID}" ] || kill "${API_PID}" 2>/dev/null || true
  [ -z "${SOLAR_MONITOR_PID}" ] || kill "${SOLAR_MONITOR_PID}" 2>/dev/null || true
  [ -z "${API_PID}" ] || wait "${API_PID}" 2>/dev/null || true
  [ -z "${SOLAR_MONITOR_PID}" ] || wait "${SOLAR_MONITOR_PID}" 2>/dev/null || true
  exit 0
}

trap shutdown INT TERM

if [ -z "${SOLAR_MONITOR_URL:-}" ]; then
  export SOLAR_MONITOR_URL="http://127.0.0.1:${SOLAR_MONITOR_PORT}"
  START_LOCAL_SOLAR_MONITOR="${START_LOCAL_SOLAR_MONITOR:-1}"
else
  START_LOCAL_SOLAR_MONITOR="${START_LOCAL_SOLAR_MONITOR:-0}"
fi

if [ "${START_LOCAL_SOLAR_MONITOR}" != "0" ]; then
  if ! command -v solar-monitor >/dev/null 2>&1; then
    echo "[entrypoint] local solar-monitor requested but binary is missing" >&2
    exit 1
  fi
  echo "[entrypoint] starting solar-monitor on 127.0.0.1:${SOLAR_MONITOR_PORT}"
  solar-monitor --host 127.0.0.1 --port "${SOLAR_MONITOR_PORT}" --poll-interval "${SOLAR_MONITOR_POLL_INTERVAL}" &
  SOLAR_MONITOR_PID="$!"
else
  echo "[entrypoint] /api/solar/* will use ${SOLAR_MONITOR_URL}"
fi

# Populate cosmic-ray cache from NMDB (live, public). Non-fatal: the live
# endpoints stream regardless, and /api/cosmic_rays degrades gracefully.
echo "[entrypoint] refreshing cached data layers..."
python refresh_data.py || echo "[entrypoint] data refresh skipped (non-fatal)"

echo "[entrypoint] starting uvicorn on 0.0.0.0:${PORT}"
uvicorn server:app --host 0.0.0.0 --port "${PORT}" &
API_PID="$!"

# Supervise both bundled processes. Data freshness is intentionally not part of
# this loop: stale NOAA observations inhibit alerts but must not cause restarts.
set +e
while kill -0 "${API_PID}" 2>/dev/null; do
  if [ -n "${SOLAR_MONITOR_PID}" ] && ! kill -0 "${SOLAR_MONITOR_PID}" 2>/dev/null; then
    wait "${SOLAR_MONITOR_PID}"
    SOLAR_STATUS="$?"
    [ "${SOLAR_STATUS}" -ne 0 ] || SOLAR_STATUS=1
    echo "[entrypoint] solar-monitor exited (${SOLAR_STATUS}); stopping API" >&2
    kill "${API_PID}" 2>/dev/null || true
    wait "${API_PID}" 2>/dev/null || true
    exit "${SOLAR_STATUS}"
  fi
  sleep 2
done

wait "${API_PID}"
API_STATUS="$?"
if [ -n "${SOLAR_MONITOR_PID}" ]; then
  kill "${SOLAR_MONITOR_PID}" 2>/dev/null || true
  wait "${SOLAR_MONITOR_PID}" 2>/dev/null || true
fi
exit "${API_STATUS}"
