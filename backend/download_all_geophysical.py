#!/usr/bin/env python3
"""
Download ALL geophysical data: magnetometer, ocean pressure, tide gauges
=========================================================================
For the antipodal precursor test:
  - SuperMAG magnetometer (ULF magnetic field at antipode)
  - NDBC DART buoys (ocean bottom pressure = tidal strain + tsunami)
  - Tide gauge data (sea level = tidal + any anomalous signals)
"""
import numpy as np
import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from datetime import timedelta
import sys, os, time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. SUPERMAG MAGNETOMETER DATA
# ═══════════════════════════════════════════════════════════════════════

def download_supermag():
    """
    SuperMAG Web Service API for magnetometer data.
    Requires logon_name parameter (free registration).
    API docs: https://supermag.jhuapl.edu/mag/lib/content/api/supermag_doc_python.pdf
    """
    print("\n[1] SUPERMAG MAGNETOMETER")

    # SuperMAG requires a logon name. Check if we have the Python client.
    print("  SuperMAG API requires registration (free):")
    print("  1. Register at https://supermag.jhuapl.edu/mag/")
    print("  2. Download Python client from their API docs page")
    print("  3. Use logon_name='YOUR_USERNAME' in API calls")
    print()
    print("  Alternative: use the web interface to download CSV:")
    print("  https://supermag.jhuapl.edu/mag/?fidelity=low&interval=23:59")
    print()

    # For now, save the stations we need for antipodal events
    # Chile 2010 antipode is ~36N, 107E (in US east coast area)
    # Turkey 2023 antipode is ~-37, -143 (southern Indian Ocean)
    events_stations = {
        "Chile2010": {
            "eq_time": "2010-02-27T06:34",
            "antipode": (35.9, 107.3),
            "nearby_supermag": ["FRD", "CLF", "FUR"],  # Fredericksburg, Chambon-la-Foret, Furstenfeldbruck
        },
        "Turkey2023": {
            "eq_time": "2023-02-06T01:17",
            "antipode": (-37.2, -143.1),
            "nearby_supermag": ["MCQ", "CSY"],  # Macquarie Island, Casey (Antarctica)
        },
        "Tohoku2011": {
            "eq_time": "2011-03-11T05:46",
            "antipode": (-38.3, -37.6),
            "nearby_supermag": ["HER", "TSU"],  # Hermanus, Tsumeb (South Africa)
        },
    }

    pd.DataFrame([
        {"event": k, "eq_time": v["eq_time"],
         "anti_lat": v["antipode"][0], "anti_lon": v["antipode"][1],
         "stations": ",".join(v["nearby_supermag"])}
        for k, v in events_stations.items()
    ]).to_csv(DATA_DIR / "supermag_stations_needed.csv", index=False)
    print("  Saved: supermag_stations_needed.csv (stations to request)")


# ═══════════════════════════════════════════════════════════════════════
# 2. NDBC / DART BUOY DATA
# ═══════════════════════════════════════════════════════════════════════

def download_dart():
    """Download DART ocean bottom pressure data from NDBC."""
    print("\n[2] DART OCEAN BOTTOM PRESSURE")

    # DART buoys are identified by 5-digit station IDs
    # Major DART buoys near earthquake antipodes:
    dart_buoys = {
        # Chile 2010 M8.8 — the actual tsunami was recorded by DARTs in Pacific
        # Antipode is US east coast — DART 44402 (Atlantic)
        "44402": "Atlantic (near Chile antipode)",
        # Tohoku 2011 — DARTs across Pacific
        "21418": "NW Pacific (near Tohoku)",
        "46402": "NE Pacific",
        # General Pacific DARTs
        "32412": "Central Pacific",
        "51407": "Hawaii",
        "55023": "Southern Pacific",
    }

    for station_id, description in dart_buoys.items():
        cache = DATA_DIR / f"dart_{station_id}.txt"
        if cache.exists():
            print(f"  Already cached: dart_{station_id}.txt")
            continue

        # Try NDBC historical data
        # DART data is at: https://www.ndbc.noaa.gov/station_history.php?station=XXXXX
        # Historical files: https://www.ndbc.noaa.gov/data/historical/dart/
        print(f"  Downloading DART {station_id} ({description})...")

        for year in [2010, 2011, 2023]:
            url = f"https://www.ndbc.noaa.gov/data/historical/dart/{station_id}d{year}.txt.gz"
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    fname = f"dart_{station_id}_{year}.txt.gz"
                    with open(DATA_DIR / fname, 'wb') as f:
                        f.write(resp.content)
                    print(f"    {year}: saved ({len(resp.content)//1024} KB)")
                else:
                    # Try without .gz
                    url2 = f"https://www.ndbc.noaa.gov/data/historical/dart/{station_id}d{year}.txt"
                    resp2 = requests.get(url2, timeout=30)
                    if resp2.status_code == 200:
                        fname = f"dart_{station_id}_{year}.txt"
                        with open(DATA_DIR / fname, 'w') as f:
                            f.write(resp2.text)
                        print(f"    {year}: saved ({len(resp2.text)//1024} KB)")
                    else:
                        print(f"    {year}: not available")
            except Exception as e:
                print(f"    {year}: FAILED ({e})")
            time.sleep(0.3)

    # Also get NCEI DART archive listing
    print("\n  Full DART archive: https://www.ncei.noaa.gov/products/natural-hazards/tsunamis-earthquakes-volcanoes/tsunamis/dart-ocean-bottom-pressure")


# ═══════════════════════════════════════════════════════════════════════
# 3. TIDE GAUGE DATA (UHSLC)
# ═══════════════════════════════════════════════════════════════════════

def download_tide_gauges():
    """Download hourly tide gauge data from University of Hawaii Sea Level Center."""
    print("\n[3] TIDE GAUGE DATA (UHSLC)")

    # UHSLC provides hourly research-quality tide gauge data
    # Format: station_id as 3-digit number
    # URL: https://uhslc.soest.hawaii.edu/data/fd{station_id}h.dat

    # Stations near earthquake antipodes
    stations = {
        "057": ("Bermuda", 32.4, -64.7, "near Chile2010 antipode"),
        "155": ("Kerguelen", -49.4, 70.2, "near Turkey2023 antipode"),
        "111": ("Simons Town", -34.2, 18.4, "near Tohoku2011 antipode"),
        "001": ("Honolulu", 21.3, -157.9, "Pacific reference"),
        "245": ("Christmas Island", -10.4, 105.7, "Indian Ocean reference"),
    }

    for sid, (name, lat, lon, note) in stations.items():
        cache = DATA_DIR / f"tidegauge_{sid}_{name.replace(' ','_')}.dat"
        if cache.exists():
            print(f"  Already cached: {cache.name}")
            continue

        url = f"https://uhslc.soest.hawaii.edu/data/fd{sid}h.dat"
        print(f"  Downloading tide gauge {sid} ({name}, {note})...")
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.text) > 100:
                with open(cache, 'w') as f:
                    f.write(resp.text)
                print(f"    Saved: {cache.name} ({len(resp.text)//1024} KB)")
            else:
                print(f"    HTTP {resp.status_code}")
        except Exception as e:
            print(f"    FAILED: {e}")
        time.sleep(0.3)


# ═══════════════════════════════════════════════════════════════════════
# 4. INTERMAGNET / GEOMAGNETIC OBSERVATORY DATA
# ═══════════════════════════════════════════════════════════════════════

def download_intermagnet():
    """
    INTERMAGNET provides 1-minute geomagnetic observatory data.
    Data available via: https://www.intermagnet.org/data-donnee/download-eng.php
    Also via NOAA NCEI: https://www.ncei.noaa.gov/products/geomagnetic-data
    """
    print("\n[4] INTERMAGNET MAGNETOMETER DATA")

    # INTERMAGNET data requires web form download or FTP
    # Key observatories near earthquake antipodes:
    observatories = {
        "FRD": ("Fredericksburg, VA", 38.2, -77.4, "near Chile2010 antipode"),
        "HER": ("Hermanus, S Africa", -34.4, 19.2, "near Tohoku2011 antipode"),
        "MCQ": ("Macquarie Island", -54.5, 159.0, "near Turkey2023 antipode"),
        "KAK": ("Kakioka, Japan", 36.2, 140.2, "reference (Tohoku local)"),
        "SUA": ("Surlari, Romania", 44.7, 26.3, "reference (Turkey local)"),
    }

    # Try NOAA NCEI for observatory data
    for code, (name, lat, lon, note) in observatories.items():
        cache = DATA_DIR / f"intermagnet_{code}.txt"
        if cache.exists():
            print(f"  Already cached: {cache.name}")
            continue

        # NCEI provides some observatory data
        # Try the definitive minute data
        print(f"  Observatory {code} ({name}, {note}):")
        print(f"    Download from: https://www.intermagnet.org/data-donnee/download-eng.php")
        print(f"    Or NCEI: https://www.ncei.noaa.gov/products/geomagnetic-data")

        # Try to get at least the hourly data from WDC
        # Edinburgh WDC provides some data via HTTP
        for year in [2010, 2011, 2023]:
            url = f"https://wdc.bgs.ac.uk/dataportal/download/{code.lower()}/{year}/{code.lower()}{year}.min.zip"
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    fname = f"intermagnet_{code}_{year}.min.zip"
                    with open(DATA_DIR / fname, 'wb') as f:
                        f.write(resp.content)
                    print(f"    {year}: saved ({len(resp.content)//1024} KB)")
                else:
                    print(f"    {year}: not available via BGS")
            except:
                pass

    # Save station list
    pd.DataFrame([
        {"code": k, "name": v[0], "lat": v[1], "lon": v[2], "note": v[3]}
        for k, v in observatories.items()
    ]).to_csv(DATA_DIR / "intermagnet_stations.csv", index=False)
    print("  Saved: intermagnet_stations.csv")


# ═══════════════════════════════════════════════════════════════════════
# 5. NDBC STANDARD BUOY DATA (ocean waves, wind, pressure)
# ═══════════════════════════════════════════════════════════════════════

def download_ndbc_buoys():
    """Download standard meteorological buoy data from NDBC."""
    print("\n[5] NDBC STANDARD BUOY DATA")

    # Major ocean buoys with long records
    buoys = {
        "41002": "South Atlantic Bight",
        "44025": "Long Island",
        "46005": "Washington coast",
        "51004": "Hawaii",
        "32012": "Central Pacific",
    }

    for station, name in buoys.items():
        for year in [2010, 2011, 2023, 2024]:
            cache = DATA_DIR / f"ndbc_{station}_{year}.txt"
            if cache.exists():
                continue
            url = f"https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    with open(DATA_DIR / f"ndbc_{station}_{year}.txt.gz", 'wb') as f:
                        f.write(resp.content)
                    print(f"  {station} {year}: {len(resp.content)//1024} KB")
                else:
                    url2 = url.replace('.gz', '')
                    resp2 = requests.get(url2, timeout=30)
                    if resp2.status_code == 200:
                        with open(cache, 'w') as f:
                            f.write(resp2.text)
                        print(f"  {station} {year}: {len(resp2.text)//1024} KB")
            except:
                pass
            time.sleep(0.2)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("DOWNLOADING ALL GEOPHYSICAL DATA")
    print("Magnetometer + Ocean Pressure + Tide Gauge + Buoy")
    print("=" * 60)

    download_supermag()
    download_dart()
    download_tide_gauges()
    download_intermagnet()
    download_ndbc_buoys()

    # Inventory
    print("\n" + "=" * 60)
    print("DATA INVENTORY — GEOPHYSICAL")
    print("=" * 60)
    total = 0
    for f in sorted(DATA_DIR.glob("*")):
        if f.is_file() and any(x in f.name for x in ["dart", "tide", "inter", "ndbc", "supermag"]):
            sz = f.stat().st_size
            total += sz
            print(f"  {f.name:50s} {sz//1024:>6d} KB")
    print(f"  {'TOTAL':50s} {total//1024:>6d} KB")

    print("\nFor SuperMAG and INTERMAGNET, manual download from web interface")
    print("may be needed. See station lists in data/ directory.")


if __name__ == "__main__":
    main()
