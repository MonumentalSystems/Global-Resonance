#!/usr/bin/env python3
"""
Refresh the cached data layers that /api endpoints read from disk.

Most Global Resonance endpoints fetch live at request time and need nothing here.
Only two layers read pre-cached files:

  * /api/cosmic_rays  -> data/solar_wind/cosmic_rays_<STATION>_<YYYYMM>_clean.csv
  * /api/lightning    -> data/lightning/wglc_climatology_30m_monthly.nc   (optional)

This script pulls cosmic-ray neutron-monitor data LIVE from NMDB (public, no key)
and writes it in the exact semicolon format server.py parses, so that layer
streams too. The lightning NetCDF is a static climatology — see the note at the
bottom for where to fetch it; /api/lightning falls back to Open-Meteo without it.

Run inside the container / repo:
    python backend/refresh_data.py

Writes to DATA_DIR (repo-root /data), matching server.py. Safe to run on a cron.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Match server.py: DATA_DIR is the repo-root data/ directory.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Stations server.py looks for in /api/cosmic_rays.
STATIONS = ["OULU", "ROME", "NEWK", "THUL"]

NMDB_URL = "https://www.nmdb.eu/nest/draw_graph.php"


def fetch_nmdb_ascii(station: str, start: datetime, end: datetime) -> str | None:
    """Fetch hourly corrected count rate for one station as semicolon ASCII.

    NMDB's draw_graph.php returns lines like:
        2026-03-01 00:00:00;123.45
    with a comment/header block we strip out.
    """
    params = {
        "formchk": "1",
        "stations[]": station,
        "tabchoice": "revori",      # revised/original corrected counts
        "dtype": "corr_for_efficiency",
        "tresolution": "60",        # 60-minute resolution
        "yunits": "0",
        "date_choice": "bydate",
        "start_year": start.year, "start_month": start.month, "start_day": start.day,
        "start_hour": 0, "start_min": 0,
        "end_year": end.year, "end_month": end.month, "end_day": end.day,
        "end_hour": 0, "end_min": 0,
        "output": "ascii",
    }
    try:
        with httpx.Client(timeout=60, headers={"User-Agent": "GlobalResonance/1.0"}) as c:
            r = c.get(NMDB_URL, params=params)
            r.raise_for_status()
            return r.text
    except Exception as e:  # noqa: BLE001 — best-effort refresh
        print(f"  [WARN] {station}: {e}")
        return None


def parse_to_clean(raw: str) -> list[str]:
    """Keep only 'YYYY-MM-DD HH:MM:SS;value' data rows, drop comments/headers."""
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ";" not in line:
            continue
        left, _, right = line.partition(";")
        left, right = left.strip(), right.strip()
        # data row starts with a date; skip the column-name header
        if not (len(left) >= 10 and left[:4].isdigit()):
            continue
        try:
            float(right)
        except ValueError:
            continue
        rows.append(f"{left};{right}")
    return rows


def refresh_cosmic_rays() -> int:
    now = datetime.now(timezone.utc)
    # server.py reads the last 72 hourly rows; pull a generous window.
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    yyyymm = now.strftime("%Y%m")
    out_dir = DATA_DIR / "solar_wind"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for station in STATIONS:
        raw = fetch_nmdb_ascii(station, start, now)
        if not raw:
            continue
        rows = parse_to_clean(raw)
        if not rows:
            print(f"  [WARN] {station}: no data rows parsed")
            continue
        out = out_dir / f"cosmic_rays_{station}_{yyyymm}_clean.csv"
        out.write_text("time;value\n" + "\n".join(rows) + "\n", encoding="utf-8")
        print(f"  Wrote {out.relative_to(DATA_DIR.parent)} ({len(rows)} rows)")
        written += 1
    return written


def main() -> int:
    print(f"Refreshing cached data layers -> {DATA_DIR}")
    n = refresh_cosmic_rays()
    print(f"\nCosmic-ray stations refreshed: {n}/{len(STATIONS)}")

    lightning = DATA_DIR / "lightning" / "wglc_climatology_30m_monthly.nc"
    if not lightning.exists():
        print(
            "\n[note] Lightning climatology not present (optional):\n"
            f"  expected: {lightning}\n"
            "  source:   WGLC / WWLLN gridded climatology (NetCDF).\n"
            "  /api/lightning falls back to live Open-Meteo thunderstorm codes without it."
        )
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
