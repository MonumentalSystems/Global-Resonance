"""
Global Resonance API server.

Serves real-time space weather and geophysical data as JSON endpoints
for the Three.js globe frontend.

Run: uvicorn server:app --reload --port 8000
"""
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx

app = FastAPI(title="Global Resonance", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE = {}  # simple in-memory cache with TTL
CACHE_TTL = 300  # 5 minutes


def cached_fetch(key, url, ttl=CACHE_TTL):
    """Fetch JSON with in-memory caching."""
    now = time.time()
    if key in CACHE and now - CACHE[key]["ts"] < ttl:
        return CACHE[key]["data"]
    try:
        with httpx.Client(timeout=15, headers={"User-Agent": "GlobalResonance/1.0"}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            CACHE[key] = {"data": data, "ts": now}
            return data
    except Exception as e:
        print(f"[WARN] {key}: {e}")
        return CACHE.get(key, {}).get("data")


def subsolar_point(dt=None):
    """Subsolar latitude and longitude."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    doy = dt.timetuple().tm_yday
    declination = 23.44 * math.sin(math.radians((360 / 365) * (doy - 81)))
    hour_frac = dt.hour + dt.minute / 60 + dt.second / 3600
    lon = (12 - hour_frac) * 15
    if lon > 180: lon -= 360
    if lon < -180: lon += 360
    return {"lat": round(declination, 2), "lon": round(lon, 2)}


def lunar_phase(dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    ref = datetime(2000, 1, 6, tzinfo=timezone.utc)
    days = (dt - ref).total_seconds() / 86400
    phase = (days % 29.53059) / 29.53059
    return {
        "phase": round(phase, 4),
        "illumination": round(phase * 100, 1),
        "tidal_force": round(math.cos(2 * math.pi * phase), 4),
        "tidal_rate": round(-math.sin(2 * math.pi * phase), 4),
        "days_to_full": round(((0.5 - phase) % 1.0) * 29.53059, 1),
        "name": (
            "New Moon" if phase < 0.0625 else
            "Waxing Crescent" if phase < 0.1875 else
            "First Quarter" if phase < 0.3125 else
            "Waxing Gibbous" if phase < 0.4375 else
            "Full Moon" if phase < 0.5625 else
            "Waning Gibbous" if phase < 0.6875 else
            "Last Quarter" if phase < 0.8125 else
            "Waning Crescent"
        ),
    }


# ========== API Endpoints ==========

@app.get("/api/earthquakes")
def get_earthquakes(hours: int = 72, min_mag: float = 4.5, limit: int = 500):
    """Recent earthquakes from USGS ComCat."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
        f"&starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&minmagnitude={min_mag}&orderby=time&limit={limit}"
    )
    data = cached_fetch("earthquakes", url, ttl=120)
    if not data:
        return {"earthquakes": []}

    ss = subsolar_point()
    eqs = []
    for f in data.get("features", []):
        p = f["properties"]
        c = f["geometry"]["coordinates"]
        try:
            lat1 = math.radians(ss["lat"])
            lon1 = math.radians(ss["lon"])
            lat2 = math.radians(c[1])
            lon2 = math.radians(c[0])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            ang_dist = math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))

            if ang_dist < 30: zone = "eye"
            elif ang_dist < 60: zone = "inner"
            elif ang_dist < 100: zone = "wavefront"
            elif ang_dist < 140: zone = "outer"
            elif ang_dist < 165: zone = "far"
            else: zone = "antipodal"

            eqs.append({
                "id": f.get("id", ""),
                "mag": p.get("mag"),
                "place": p.get("place", ""),
                "time": p.get("time"),
                "lon": c[0],
                "lat": c[1],
                "depth": c[2],
                "ang_dist": round(ang_dist, 1),
                "zone": zone,
                "tsunami": p.get("tsunami", 0),
                "felt": p.get("felt"),
                "alert": p.get("alert"),
                "type": p.get("type", "earthquake"),
            })
        except Exception:
            pass

    return {"earthquakes": eqs, "subsolar": ss, "count": len(eqs)}


@app.get("/api/solar_wind")
def get_solar_wind():
    """Solar wind Bz, speed, density from SWPC."""
    mag = cached_fetch("sw_mag", "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json")
    plasma = cached_fetch("sw_plasma", "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json")

    result = {"bz": [], "speed": [], "density": []}
    if mag:
        for row in mag[1:]:
            try:
                bz = float(row[3]) if row[3] not in (None, "", "null") else None
                result["bz"].append({"time": row[0], "value": bz})
            except Exception:
                pass

    if plasma:
        for row in plasma[1:]:
            try:
                d = float(row[1]) if row[1] not in (None, "", "null") else None
                v = float(row[2]) if row[2] not in (None, "", "null") else None
                result["speed"].append({"time": row[0], "value": v})
                result["density"].append({"time": row[0], "value": d})
            except Exception:
                pass

    # Current values
    if result["bz"]:
        result["current_bz"] = next((x["value"] for x in reversed(result["bz"]) if x["value"] is not None), None)
    if result["speed"]:
        result["current_speed"] = next((x["value"] for x in reversed(result["speed"]) if x["value"] is not None), None)
    if result["density"]:
        result["current_density"] = next((x["value"] for x in reversed(result["density"]) if x["value"] is not None), None)

    return result


@app.get("/api/kp")
def get_kp():
    """Kp index from SWPC."""
    data = cached_fetch("kp", "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
    if not data:
        return {"kp": []}
    entries = []
    for row in data[1:]:
        try:
            entries.append({"time": row[0], "kp": float(row[1])})
        except Exception:
            pass
    current = entries[-1]["kp"] if entries else None
    return {"kp": entries, "current": current}


@app.get("/api/xrs")
def get_xrs():
    """GOES X-ray flux — Schumann order parameter proxy."""
    data = cached_fetch("xrs", "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json", ttl=120)
    if not data:
        return {"xrs": []}

    entries = []
    for row in data:
        if row.get("energy") != "0.1-0.8nm":
            continue
        try:
            f = float(row["flux"])
            if f > 0:
                entries.append({"time": row["time_tag"], "flux": f})
        except Exception:
            pass

    # Compute df/dt (fractional rate of change, 15-min smoothed)
    dfdt = []
    window = 15
    for i in range(window, len(entries)):
        avg_now = sum(e["flux"] for e in entries[i-window:i]) / window
        avg_prev = sum(e["flux"] for e in entries[i-window-1:i-1]) / window if i > window else avg_now
        rate = (avg_now - avg_prev) / max(avg_prev, 1e-10)
        dfdt.append({"time": entries[i]["time"], "rate": round(rate, 6)})

    # Current state
    current_rate = dfdt[-1]["rate"] if dfdt else 0
    state = "FALLING" if current_rate < -0.001 else "RISING" if current_rate > 0.001 else "STABLE"

    return {
        "xrs": entries[-500:],  # last ~8 hours at 1-min
        "dfdt": dfdt[-500:],
        "current_flux": entries[-1]["flux"] if entries else None,
        "current_rate": current_rate,
        "state": state,
    }


@app.get("/api/subsolar")
def get_subsolar():
    """Current subsolar point + Jelly Ball zone radii."""
    ss = subsolar_point()
    return {
        **ss,
        "zones": [
            {"name": "eye", "radius_deg": 30, "color": "#4444ff", "ratio": 0.85},
            {"name": "inner", "radius_deg": 60, "color": "#44ff44", "ratio": 1.05},
            {"name": "wavefront", "radius_deg": 100, "color": "#ff4444", "ratio": 1.36},
            {"name": "outer", "radius_deg": 140, "color": "#ffff44", "ratio": 1.10},
            {"name": "far", "radius_deg": 165, "color": "#888888", "ratio": 1.05},
            {"name": "antipodal", "radius_deg": 180, "color": "#cc88cc", "ratio": 1.16},
        ],
        "terminator_lon": (ss["lon"] + 90) % 360 - 180,
    }


@app.get("/api/lunar")
def get_lunar():
    """Lunar phase and tidal stress."""
    return lunar_phase()


@app.get("/api/sun")
def get_sun():
    """SDO/SOHO solar imagery URLs and active region info."""
    # SDO latest images (updated every ~15 min)
    sdo_base = "https://sdo.gsfc.nasa.gov/assets/img/latest"
    return {
        "images": {
            "aia_193": f"{sdo_base}/latest_1024_0193.jpg",      # Corona / flare sites
            "aia_304": f"{sdo_base}/latest_1024_0304.jpg",      # Chromosphere
            "aia_171": f"{sdo_base}/latest_1024_0171.jpg",      # Quiet corona
            "aia_131": f"{sdo_base}/latest_1024_0131.jpg",      # Flare plasma
            "hmi_mag": f"{sdo_base}/latest_1024_HMIBC.jpg",     # Magnetogram
            "hmi_cont": f"{sdo_base}/latest_1024_HMIIF.jpg",    # Continuum (sunspots)
            "lasco_c2": "https://soho.nascom.nasa.gov/data/realtime/c2/1024/latest.jpg",  # Coronagraph
            "lasco_c3": "https://soho.nascom.nasa.gov/data/realtime/c3/1024/latest.jpg",
        },
        "description": {
            "aia_193": "EUV 193A: corona, coronal holes, flare sites",
            "aia_304": "EUV 304A: chromosphere, prominences",
            "aia_131": "EUV 131A: hot flare plasma (10M K)",
            "hmi_mag": "HMI magnetogram: surface magnetic field polarity",
            "hmi_cont": "HMI continuum: sunspot structure",
            "lasco_c2": "LASCO C2: inner coronagraph (CME detection)",
            "lasco_c3": "LASCO C3: outer coronagraph (CME tracking)",
        },
    }


@app.get("/api/cosmic_rays")
def get_cosmic_rays():
    """Cosmic ray neutron monitor data for Forbush decrease detection."""
    # Check for cached NMDB data
    result = {"stations": {}}
    for station in ["OULU", "ROME", "NEWK", "THUL"]:
        f = DATA_DIR / "solar_wind" / f"cosmic_rays_{station}_202603_clean.csv"
        if f.exists():
            lines = f.read_text().strip().split("\n")[1:]  # skip header
            entries = []
            for line in lines[-72:]:  # last 3 days
                parts = line.split(";")
                if len(parts) >= 2:
                    entries.append({"time": parts[0].strip(), "value": float(parts[1].strip())})
            if entries:
                values = [e["value"] for e in entries]
                mean_val = sum(values) / len(values)
                last_val = values[-1]
                result["stations"][station] = {
                    "entries": entries,
                    "current": last_val,
                    "mean_72h": round(mean_val, 1),
                    "deviation_pct": round((last_val - mean_val) / mean_val * 100, 2),
                }

    # Forbush detection: any station showing >3% drop
    forbush = False
    for st, d in result["stations"].items():
        if d["deviation_pct"] < -3:
            forbush = True
            break
    result["forbush_detected"] = forbush

    return result


@app.get("/api/dst")
def get_dst():
    """Real-time Dst index from Kyoto WDC."""
    import ssl
    # Kyoto WDC has cert issues, use httpx with verify=False for this source
    try:
        with httpx.Client(timeout=15, verify=False, headers={"User-Agent": "GlobalResonance/1.0"}) as client:
            resp = client.get("https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/202603/dst2603.for.request")
            content = resp.text
    except Exception as e:
        return {"dst": [], "error": str(e)}

    # Parse WDC format: each line is STATION YEAR MONTH DAY BASELINE 24_hourly_values MEAN
    # Format: DST  26 3  1  ... (24 hourly values) ...
    entries = []
    for line in content.strip().split("\n"):
        if not line.strip() or line.startswith("DST") is False:
            # Try to parse anyway — WDC format varies
            pass
        try:
            # Fixed-width format: day in cols 8-9, then 24 values of 4 chars each
            if len(line) < 50:
                continue
            day = int(line[8:10].strip())
            base = int(line[16:20].strip()) if line[16:20].strip().lstrip('-').isdigit() else 0
            hourly = []
            for h in range(24):
                start = 20 + h * 4
                val_str = line[start:start+4].strip()
                if val_str and val_str.lstrip('-').isdigit():
                    hourly.append(int(val_str))
                else:
                    hourly.append(None)
            for h, val in enumerate(hourly):
                if val is not None:
                    entries.append({
                        "day": day,
                        "hour": h,
                        "dst": val,
                    })
        except Exception:
            pass

    current = entries[-1]["dst"] if entries else None
    return {
        "dst": entries[-72:],  # last 3 days
        "current": current,
        "storm": current is not None and current < -50,
    }


@app.get("/api/cme")
def get_cme():
    """Active CME tracking — cone projection for the globe."""
    # Current active CME (X1.4 from Mar 30)
    cme_launch = "2026-03-30T03:24:00Z"
    cme_speed = 1689
    half_angle = 46
    source_lat = -27  # S27
    source_lon_solar = 45  # E45 on sun -> need to convert to Earth subsolar frame
    predicted_arrival = "2026-03-31T15:07:00Z"
    earth_impact_prob = 0.92
    kp_forecast = "6-9"

    # Fetch recent CMEs from DONKI cache
    donki = cached_fetch("donki_cme",
        "https://api.nasa.gov/DONKI/CME?startDate=2026-03-28&endDate=2026-03-31&api_key=DEMO_KEY",
        ttl=600)

    active_cmes = []
    if donki:
        for cme in donki:
            analyses = cme.get("cmeAnalyses", [])
            for a in analyses:
                if a.get("isEarthGB"):
                    active_cmes.append({
                        "time": cme.get("startTime"),
                        "speed": a.get("speed"),
                        "half_angle": a.get("halfAngle"),
                        "arrival": a.get("estimatedShockArrivalTime"),
                        "kp": a.get("kp_18") or a.get("kp_90"),
                    })

    return {
        "primary": {
            "launch": cme_launch,
            "speed": cme_speed,
            "half_angle_deg": half_angle,
            "source_solar": {"lat": source_lat, "lon": source_lon_solar},
            "predicted_arrival": predicted_arrival,
            "earth_impact_prob": earth_impact_prob,
            "kp_forecast": kp_forecast,
        },
        "active_cmes": active_cmes,
    }


@app.get("/api/flares")
def get_flares():
    """Recent solar flares from DONKI."""
    data = cached_fetch("donki_flares",
        "https://api.nasa.gov/DONKI/FLR?startDate=2026-03-28&endDate=2026-03-31&api_key=DEMO_KEY",
        ttl=600)
    if not data:
        return {"flares": []}
    flares = []
    for fl in data:
        flares.append({
            "class": fl.get("classType"),
            "peak": fl.get("peakTime"),
            "begin": fl.get("beginTime"),
            "end": fl.get("endTime"),
            "source": fl.get("sourceLocation"),
            "ar": fl.get("activeRegionNum"),
        })
    return {"flares": flares}


@app.get("/api/magnetometers")
def get_magnetometers():
    """Ground magnetometer station locations and live data links."""
    # USGS and INTERMAGNET stations we have data for
    stations = [
        {"code": "BOU", "name": "Boulder", "lat": 40.14, "lon": -105.24, "network": "USGS"},
        {"code": "FRD", "name": "Fredericksburg", "lat": 38.20, "lon": -77.37, "network": "USGS"},
        {"code": "HON", "name": "Honolulu", "lat": 21.32, "lon": -158.00, "network": "USGS"},
        {"code": "SJG", "name": "San Juan", "lat": 18.11, "lon": -66.15, "network": "USGS"},
        {"code": "TUC", "name": "Tucson", "lat": 32.17, "lon": -110.73, "network": "USGS"},
        {"code": "HER", "name": "Hermanus", "lat": -34.43, "lon": 19.23, "network": "INTERMAGNET"},
        {"code": "KAK", "name": "Kakioka", "lat": 36.23, "lon": 140.19, "network": "INTERMAGNET"},
        {"code": "SUA", "name": "Surlari", "lat": 44.68, "lon": 26.25, "network": "INTERMAGNET"},
        {"code": "MCQ", "name": "Macquarie Is.", "lat": -54.50, "lon": 158.95, "network": "INTERMAGNET"},
    ]

    # Try to get live FRD data from USGS
    now = datetime.now(timezone.utc)
    try:
        frd_url = (
            f"https://geomag.usgs.gov/ws/data/?elements=X,Y,Z&"
            f"endtime={now.strftime('%Y-%m-%dT%H:%M:%SZ')}&"
            f"id=FRD&sampling_period=60&"
            f"starttime={(now - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}&"
            f"type=variation&format=json"
        )
        frd_data = cached_fetch("frd_mag", frd_url, ttl=120)
        if frd_data and "values" in frd_data:
            vals = frd_data["values"]
            if vals:
                last = vals[-1]
                for s in stations:
                    if s["code"] == "FRD":
                        s["live"] = {
                            "X": last.get("X"),
                            "Y": last.get("Y"),
                            "Z": last.get("Z"),
                            "time": now.isoformat(),
                        }
    except Exception:
        pass

    return {"stations": stations}


@app.get("/api/status")
def get_status():
    """Overall system status combining all data sources."""
    now = datetime.now(timezone.utc)
    ss = subsolar_point()
    moon = lunar_phase()

    return {
        "time_utc": now.isoformat(),
        "subsolar": ss,
        "lunar": moon,
        "data_sources": {
            "earthquakes": "earthquakes" in CACHE,
            "solar_wind": "sw_mag" in CACHE,
            "xrs": "xrs" in CACHE,
            "kp": "kp" in CACHE,
        },
    }


# Serve frontend if available
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
