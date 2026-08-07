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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import httpx
import asyncio
import os

app = FastAPI(
    title="Global Resonance API",
    version="0.1.0",
    description=(
        "Real-time space weather and geophysical monitoring. Most endpoints fetch "
        "live from public sources (NOAA SWPC, USGS, NASA, NMDB, IRIS, Open-Meteo) "
        "and cache in-memory for 5 minutes.\n\n"
        "- **Swagger UI**: [/docs](/docs)\n"
        "- **ReDoc**: [/redoc](/redoc)\n"
        "- **OpenAPI schema**: [/openapi.json](/openapi.json)"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    from research_model_context import cascadia_nsaf_advisories, research_model_context
except ImportError:  # supports `uvicorn backend.server:app` from the repository root
    from backend.research_model_context import cascadia_nsaf_advisories, research_model_context

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE = {}  # simple in-memory cache with TTL
CACHE_TTL = 300  # 5 minutes

# NASA DONKI / APOD key. DEMO_KEY works but is rate-limited (30/hr, 50/day).
# Get a free key at https://api.nasa.gov and set NASA_API_KEY to lift the limit.
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")


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


def fill_vector_grid(u_list, v_list, n_lat, n_lon, passes=6):
    """Diffuse sparse vector samples into nearby empty cells for visualization."""
    u_grid = [u_list[i * n_lon:(i + 1) * n_lon] for i in range(n_lat)]
    v_grid = [v_list[i * n_lon:(i + 1) * n_lon] for i in range(n_lat)]

    for _ in range(passes):
        changed = False
        next_u = [row[:] for row in u_grid]
        next_v = [row[:] for row in v_grid]
        for i in range(n_lat):
            for j in range(n_lon):
                if u_grid[i][j] is not None and v_grid[i][j] is not None:
                    continue

                neigh_u = []
                neigh_v = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ii = i + di
                        if ii < 0 or ii >= n_lat:
                            continue
                        jj = (j + dj) % n_lon
                        if u_grid[ii][jj] is None or v_grid[ii][jj] is None:
                            continue
                        neigh_u.append(u_grid[ii][jj])
                        neigh_v.append(v_grid[ii][jj])

                if len(neigh_u) >= 2:
                    next_u[i][j] = round(sum(neigh_u) / len(neigh_u), 3)
                    next_v[i][j] = round(sum(neigh_v) / len(neigh_v), 3)
                    changed = True

        u_grid = next_u
        v_grid = next_v
        if not changed:
            break

    filled_u = []
    filled_v = []
    filled_s = []
    for i in range(n_lat):
        for j in range(n_lon):
            u = u_grid[i][j]
            v = v_grid[i][j]
            filled_u.append(u)
            filled_v.append(v)
            if u is None or v is None:
                filled_s.append(None)
            else:
                filled_s.append(round(math.sqrt(u * u + v * v), 3))

    return filled_u, filled_v, filled_s


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

            # Backtested 10-zone spatial response (183K events, 2000-2026)
            if ang_dist < 15: zone = "eye"              # 1.26x subsolar focusing
            elif ang_dist < 30: zone = "inner"           # 1.22x enhanced
            elif ang_dist < 60: zone = "transition"      # 1.07x slightly enhanced
            elif ang_dist < 75: zone = "wavefront"       # 0.89x suppressed
            elif ang_dist < 100: zone = "wavefront-tail"  # 1.09x enhancement
            elif ang_dist < 120: zone = "neutral"        # 0.95x
            elif ang_dist < 135: zone = "far-suppress"   # 0.82x suppression
            elif ang_dist < 155: zone = "far-neutral"    # 0.90x
            elif ang_dist < 165: zone = "pre-antipodal"  # 1.00x
            else: zone = "antipodal"                     # 1.16x enhancement

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

    return {
        "earthquakes": eqs,
        "subsolar": ss,
        "count": len(eqs),
        "compound_fault_advisories": cascadia_nsaf_advisories(eqs),
    }


@app.get("/api/research/model-context")
def get_research_model_context():
    """Source-audited boundaries for recent fault, solar, and core results."""
    return research_model_context()


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
        return {"kp": [], "current": None}
    entries = []
    for row in data:
        try:
            # Format can be list-of-dicts or list-of-lists
            if isinstance(row, dict):
                entries.append({"time": row.get("time_tag", ""), "kp": float(row.get("Kp", row.get("kp", 0)))})
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                kp_val = row[1]
                if isinstance(kp_val, str) and not kp_val.replace('.','').replace('-','').isdigit():
                    continue  # skip header row
                entries.append({"time": row[0], "kp": float(kp_val)})
        except (ValueError, TypeError, KeyError):
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
    """Current subsolar point + Paper XXV Jelly Ball zone geometry."""
    ss = subsolar_point()
    return {
        **ss,
        "zones": [
            # Backtested spatial response (183K M4.5+ events, 2000-2026)
            # Subsolar focusing: earthquakes cluster NEAR subsolar point and at antipode
            {"name": "eye",            "radius_deg": 15,  "color": "#ff4444", "ratio": 1.26, "effect": "ENHANCED (subsolar focusing)"},
            {"name": "inner",          "radius_deg": 30,  "color": "#ff6644", "ratio": 1.22, "effect": "enhanced"},
            {"name": "transition",     "radius_deg": 60,  "color": "#ff8844", "ratio": 1.07, "effect": "slightly enhanced"},
            {"name": "wavefront",      "radius_deg": 75,  "color": "#446688", "ratio": 0.89, "effect": "suppressed"},
            {"name": "wavefront-tail", "radius_deg": 100, "color": "#4444aa", "ratio": 0.83, "effect": "SUPPRESSED (minimum)"},
            {"name": "neutral",        "radius_deg": 120, "color": "#445566", "ratio": 0.90, "effect": "suppressed"},
            {"name": "far-suppress",   "radius_deg": 135, "color": "#666666", "ratio": 1.04, "effect": "near-neutral"},
            {"name": "far-neutral",    "radius_deg": 155, "color": "#886688", "ratio": 1.19, "effect": "enhanced"},
            {"name": "pre-antipodal",  "radius_deg": 165, "color": "#cc88cc", "ratio": 1.29, "effect": "enhanced"},
            {"name": "antipodal",      "radius_deg": 180, "color": "#ff44ff", "ratio": 1.35, "effect": "ENHANCED (antipodal focusing)"},
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
    # SOHO real-time images (reliable, always available)
    soho = "https://soho.nascom.nasa.gov/data/realtime"
    # SDO via LMSAL mirror (more reliable than sdo.gsfc.nasa.gov)
    sdo = "https://sdo.gsfc.nasa.gov/assets/img/latest"
    return {
        "images": {
            "eit_195": f"{soho}/eit_195/1024/latest.jpg",       # EIT 195A: corona (like AIA 193)
            "eit_304": f"{soho}/eit_304/1024/latest.jpg",       # EIT 304A: chromosphere
            "eit_171": f"{soho}/eit_171/1024/latest.jpg",       # EIT 171A: quiet corona
            "eit_284": f"{soho}/eit_284/1024/latest.jpg",       # EIT 284A: active regions
            "hmi_mag": f"{soho}/hmi_mag/1024/latest.jpg",       # HMI magnetogram
            "hmi_con": f"{soho}/hmi_igr/1024/latest.jpg",       # HMI intensitygram (sunspots)
            "lasco_c2": f"{soho}/c2/1024/latest.jpg",           # Coronagraph inner
            "lasco_c3": f"{soho}/c3/1024/latest.jpg",           # Coronagraph outer
        },
        "description": {
            "eit_195": "EIT 195A: hot corona, flare sites (~1.5 MK)",
            "eit_304": "EIT 304A: chromosphere, prominences (~80K K)",
            "eit_171": "EIT 171A: quiet corona, coronal loops (~1 MK)",
            "eit_284": "EIT 284A: active regions (~2 MK)",
            "hmi_mag": "HMI magnetogram: surface magnetic field polarity",
            "hmi_con": "HMI intensitygram: sunspot structure",
            "lasco_c2": "LASCO C2: inner coronagraph (2-6 Rsun, CME detection)",
            "lasco_c3": "LASCO C3: outer coronagraph (4-30 Rsun, CME tracking)",
        },
    }


@app.get("/api/cosmic_rays")
def get_cosmic_rays():
    """Cosmic ray neutron monitor data for Forbush decrease detection."""
    # Check for cached NMDB data. refresh_data.py writes one file per station
    # per month (cosmic_rays_<STATION>_<YYYYMM>_clean.csv); pick the latest.
    result = {"stations": {}}
    sw_dir = DATA_DIR / "solar_wind"
    for station in ["OULU", "ROME", "NEWK", "THUL"]:
        matches = sorted(sw_dir.glob(f"cosmic_rays_{station}_*_clean.csv")) if sw_dir.exists() else []
        f = matches[-1] if matches else (sw_dir / f"cosmic_rays_{station}_202603_clean.csv")
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


@app.get("/api/cme_predict")
def get_cme_predict(
    v_nose: float = 1689,
    source_lon: float = 45,
    half_angle: float = 46,
    v_sw: float = None,
    launch_time: str = "2026-03-30T03:24:00Z",
):
    """
    Geometric CME transit prediction.

    Uses flank correction + aerodynamic drag, which beats DONKI/ENLIL
    for oblique sources (E30+).

    v_effective = v_nose * cos(source_lon)
    Then integrate drag: dv/dt = -gamma * (v - v_sw) * |v - v_sw|
    """
    from cme_transit import cme_transit, cme_dual_transit

    # Use measured solar wind if not provided
    if v_sw is None:
        sw = cached_fetch("sw_plasma", "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json")
        if sw:
            for row in reversed(sw[1:]):
                try:
                    v_sw = float(row[2])
                    if v_sw and v_sw > 200:
                        break
                except Exception:
                    pass
        if not v_sw:
            v_sw = 400

    dual = cme_dual_transit(v_nose, source_lon, half_angle, v_sw)
    result = dual["ejecta"]

    if not result.get("hit"):
        return {"hit": False, "reason": result.get("reason", "CME misses Earth")}

    launch_dt = datetime.fromisoformat(launch_time.replace("Z", "+00:00"))
    arrival_dt = launch_dt + timedelta(hours=result["transit_hours"])

    shock_result = dual["shock"]
    shock_dt = launch_dt + timedelta(hours=shock_result["transit_hours"]) if shock_result.get("hit") else None

    # Also compute the naive ballistic prediction for comparison
    ballistic_h = (1.496e8 / v_nose) / 3600
    ballistic_dt = launch_dt + timedelta(hours=ballistic_h)

    # DONKI prediction (if available from cache)
    donki_arrival = None
    donki_data = cached_fetch("donki_cme",
        f"https://api.nasa.gov/DONKI/CME?startDate=2026-03-28&endDate=2026-03-31&api_key={NASA_API_KEY}",
        ttl=600)
    if donki_data:
        for cme in donki_data:
            for a in cme.get("cmeAnalyses", []):
                if a.get("isEarthGB") and a.get("estimatedShockArrivalTime"):
                    donki_arrival = a["estimatedShockArrivalTime"]

    return {
        "hit": True,
        "launch": launch_time,
        "v_nose": v_nose,
        "source_lon": source_lon,
        "half_angle": half_angle,
        "v_sw_measured": round(v_sw),
        "shock": {
            "v_arrival": shock_result.get("v_arrival"),
            "transit_hours": shock_result.get("transit_hours"),
            "arrival": shock_dt.isoformat() if shock_dt else None,
            "arrival_readable": shock_dt.strftime("%b %d %H:%M UTC") if shock_dt else None,
            "note": "Shock front: near-ballistic nose speed, brief magnetopause compression",
        },
        "ejecta": {
            "v_flank": result["v_flank"],
            "v_arrival": result["v_arrival"],
            "transit_hours": result["transit_hours"],
            "arrival": arrival_dt.isoformat(),
            "arrival_readable": arrival_dt.strftime("%b %d %H:%M UTC"),
            "drag_deceleration_pct": result["drag_deceleration"],
            "note": "Magnetic ejecta: flank speed + drag, sustained Bz south = geomagnetic storm",
        },
        "separation_hours": dual["separation_hours"],
        "comparison": {
            "ballistic_nose": {"transit_h": round(ballistic_h, 1), "arrival": ballistic_dt.strftime("%b %d %H:%M UTC")},
            "donki_enlil": {"arrival": donki_arrival},
            "ccmc_actual_shock": {"arrival": "2026-03-31T05:53Z", "transit_h": 26.5},
            "swpc_ejecta_forecast": {"arrival": "2026-04-01T03:00-09:00Z"},
            "geometric_model": {
                "shock": shock_dt.strftime("%b %d %H:%M UTC") if shock_dt else None,
                "ejecta": arrival_dt.strftime("%b %d %H:%M UTC"),
                "method": f"shock=nose(weak drag) | ejecta=v*cos({source_lon})+drag(v_sw={v_sw:.0f})",
            },
        },
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
        f"https://api.nasa.gov/DONKI/CME?startDate=2026-03-28&endDate=2026-03-31&api_key={NASA_API_KEY}",
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
        f"https://api.nasa.gov/DONKI/FLR?startDate=2026-03-28&endDate=2026-03-31&api_key={NASA_API_KEY}",
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


@app.get("/api/jellyball/prediction")
def get_jellyball_prediction():
    """
    Jelly Ball coupled oscillator prediction tracker.

    Computes the current state of the KT phase transition model:
    - J (stiffness) from Kp/Dst/Bz
    - Gap to critical threshold J_c = 2/pi
    - Correlation length xi
    - Active scenario (recovery, compression, relaxation burst)
    - Zone-resolved earthquake risk modulation
    - Compound event tracking
    """
    J_C = 2 / math.pi  # 0.6366 — critical threshold

    # Get current data from cache
    kp_data = CACHE.get("kp", {}).get("data")
    sw_data = CACHE.get("sw_mag", {}).get("data")

    # Current Kp
    kp = 2.0
    if kp_data:
        try:
            for row in reversed(kp_data if isinstance(kp_data, list) else []):
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    try:
                        kp = float(row[1])
                        break
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    # Current Bz
    bz = 0.0
    if sw_data:
        try:
            for row in reversed(sw_data[1:]):
                v = row[3] if len(row) > 3 else None
                if v not in (None, "", "null"):
                    bz = float(v)
                    break
        except Exception:
            pass

    # Current solar wind speed
    v_sw = 400.0
    if sw_data:
        try:
            plasma = CACHE.get("sw_plasma", {}).get("data")
            if plasma:
                for row in reversed(plasma[1:]):
                    v = row[2] if len(row) > 2 else None
                    if v not in (None, "", "null"):
                        v_sw = float(v)
                        break
        except Exception:
            pass

    # === Compute J (coupling stiffness) ===
    # J is driven by geomagnetic compression:
    #   - Kp is the primary proxy (Kp=0 -> J~0.50, Kp=9 -> J~0.85)
    #   - Bz southward enhances reconnection (reduces J at magnetopause)
    #   - Bz northward -> closed magnetosphere -> J from compression
    #   - High V_sw -> dynamic pressure -> J increase
    j_from_kp = 0.50 + kp * 0.04  # 0.50 at Kp=0, 0.86 at Kp=9
    j_bz_mod = 0.0
    if bz < -5:
        j_bz_mod = -0.02 * abs(bz + 5) / 15  # southward reduces J (energy dissipated as aurora)
    elif bz > 2:
        j_bz_mod = 0.01 * bz / 10  # northward enhances compression
    j_vsw_mod = max(0, (v_sw - 400) / 600) * 0.05  # high speed stream contribution
    j_current = max(0.40, min(0.90, j_from_kp + j_bz_mod + j_vsw_mod))

    # === Gap to critical ===
    gap = J_C - j_current
    gap_pct = gap / J_C * 100

    # === Correlation length xi ===
    # xi ~ a * exp(b / sqrt(|J/J_c - 1|))
    # Diverges as J -> J_c (critical point)
    j_ratio = j_current / J_C
    delta = abs(j_ratio - 1.0)
    if delta > 0.001:
        xi_km = 1e4 * math.exp(1.5 / math.sqrt(delta))
        xi_km = min(xi_km, 1e8)  # cap at reasonable value
    else:
        xi_km = 1e8  # at criticality, correlation length diverges
    xi_rsun = xi_km / 6.957e5  # in solar radii

    # === Phase determination ===
    above_critical = j_current > J_C
    near_critical = abs(gap_pct) < 10  # within 10% of J_c

    if above_critical:
        if kp >= 6:
            phase = "STORM COMPRESSION"
            phase_detail = "J >> J_c, system ordered, seismicity SUPPRESSED"
        else:
            phase = "ELEVATED"
            phase_detail = "J > J_c, approaching relaxation"
    elif near_critical:
        phase = "CRITICAL TRANSITION"
        phase_detail = f"J within {abs(gap_pct):.1f}% of J_c — maximum sensitivity"
    elif gap_pct > 0 and gap_pct < 20:
        phase = "POST-STORM RECOVERY"
        phase_detail = "J dropping through J_c, relaxation burst possible"
    else:
        phase = "QUIET"
        phase_detail = "J well below J_c, normal seismicity"

    # === Bz shield state ===
    if bz < -5:
        shield = "ON"
        shield_detail = "Southward Bz: reconnection dissipates energy as aurora"
    elif bz > 2:
        shield = "OFF"
        shield_detail = "Northward Bz: closed magnetosphere, compression transmits to crust"
    else:
        shield = "TRANSITIONAL"
        shield_detail = "Bz near zero: shield state uncertain"

    # === Zone risk modulation (backtested from 183K events, 2000-2026) ===
    # Phase-resolved ratios derived from spatial backtest
    ZONE_RATIOS_BY_PHASE = {
        "above_critical": {
            "eye": 1.742, "inner": 1.567, "transition": 1.187,
            "wavefront": 0.780, "wavefront-tail": 0.850, "neutral": 0.859,
            "far-suppress": 1.103, "far-neutral": 0.932,
            "pre-antipodal": 1.144, "antipodal": 0.926,
        },
        "compression": {
            "eye": 1.624, "inner": 1.339, "transition": 1.096,
            "wavefront": 0.818, "wavefront-tail": 0.892, "neutral": 0.946,
            "far-suppress": 0.921, "far-neutral": 1.128,
            "pre-antipodal": 1.067, "antipodal": 1.139,
        },
        "critical_transition": {
            "eye": 1.789, "inner": 1.137, "transition": 1.385,
            "wavefront": 0.907, "wavefront-tail": 0.801, "neutral": 0.798,
            "far-suppress": 0.829, "far-neutral": 0.904,
            "pre-antipodal": 1.394, "antipodal": 1.707,
        },
        "unsettled": {
            "eye": 1.418, "inner": 1.364, "transition": 1.099,
            "wavefront": 0.917, "wavefront-tail": 0.809, "neutral": 0.872,
            "far-suppress": 1.004, "far-neutral": 1.151,
            "pre-antipodal": 1.315, "antipodal": 1.243,
        },
        "quiet": {
            "eye": 1.209, "inner": 1.181, "transition": 1.064,
            "wavefront": 0.883, "wavefront-tail": 0.835, "neutral": 0.909,
            "far-suppress": 1.044, "far-neutral": 1.209,
            "pre-antipodal": 1.291, "antipodal": 1.378,
        },
    }

    # Select phase for zone ratios
    if above_critical:
        phase_key = "above_critical"
    elif near_critical:
        phase_key = "critical_transition"
    elif kp >= 5:
        phase_key = "compression"
    elif kp >= 3:
        phase_key = "unsettled"
    else:
        phase_key = "quiet"

    ratios = ZONE_RATIOS_BY_PHASE[phase_key]
    zone_risk = {}
    for z, factor in ratios.items():
        if factor >= 1.3:
            risk = "ENHANCED"
        elif factor >= 1.1:
            risk = "enhanced"
        elif factor >= 0.95:
            risk = "near-normal"
        elif factor >= 0.85:
            risk = "suppressed"
        else:
            risk = "SUPPRESSED"
        zone_risk[z] = {"factor": round(factor, 3), "risk": risk}

    return {
        "j_current": round(j_current, 4),
        "j_critical": round(J_C, 4),
        "gap": round(gap, 4),
        "gap_pct": round(gap_pct, 1),
        "above_critical": above_critical,
        "phase": phase,
        "phase_detail": phase_detail,
        "correlation_length_km": round(xi_km, 0),
        "correlation_length_rsun": round(xi_rsun, 1),
        "shield": shield,
        "shield_detail": shield_detail,
        "zone_risk": zone_risk,
        "inputs": {
            "kp": round(kp, 2),
            "bz": round(bz, 1),
            "v_sw": round(v_sw, 0),
        },
    }


@app.get("/api/jellyball/neural")
def get_jellyball_neural():
    """
    JellyBallNet neural predictions — zone-resolved seismicity ratios.

    Runs the trained model on current solar wind data to predict
    per-zone earthquake rate modulation for each storm phase.
    """
    try:
        import torch
        from jellyball_net import JellyBallNet, ZONES, J_C, N_MODES
    except ImportError:
        return {"error": "PyTorch or jellyball_net not available"}

    model_path = Path(__file__).parent / "output" / "jellyball_net.pt"
    if not model_path.exists():
        return {"error": "Model not trained yet. Run jellyball_net.py first."}

    import numpy as np

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = JellyBallNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]

    # Get current solar data from cache
    kp = 2.0
    kp_data = CACHE.get("kp", {}).get("data")
    if kp_data:
        try:
            for row in reversed(kp_data if isinstance(kp_data, list) else []):
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    try:
                        kp = float(row[1])
                        break
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    bz = 0.0
    sw_data = CACHE.get("sw_mag", {}).get("data")
    if sw_data:
        try:
            for row in reversed(sw_data[1:]):
                v = row[3] if len(row) > 3 else None
                if v not in (None, "", "null"):
                    bz = float(v)
                    break
        except Exception:
            pass

    dst = -20.0  # default
    solar_input = [kp, dst, bz, 0, 0, 0]
    solar_norm = (np.array(solar_input, dtype=np.float32) - x_mean) / x_std

    # Run model for all 4 phases
    phase_names = ["compression", "peak", "relaxation_early", "relaxation_late"]
    phase_values = [0.0, 0.2, 0.5, 0.8]
    zone_names = [z[0] for z in ZONES]

    predictions = {}
    diagnostics = {}

    with torch.no_grad():
        for phase_name, phase_val in zip(phase_names, phase_values):
            x = torch.tensor(solar_norm, dtype=torch.float32).unsqueeze(0)
            t = torch.tensor([[phase_val]], dtype=torch.float32)
            pred, diag = model(x, t)

            zone_ratios = pred[0].numpy()
            predictions[phase_name] = {
                zone_names[i]: round(float(zone_ratios[i]), 3)
                for i in range(len(zone_names))
            }

        # Get diagnostics from compression phase
        x = torch.tensor(solar_norm, dtype=torch.float32).unsqueeze(0)
        t = torch.tensor([[0.0]], dtype=torch.float32)
        _, diag = model(x, t)
        diagnostics = {
            "J": round(float(diag["J"][0, 0]), 4),
            "J_critical": round(J_C, 4),
            "bivector_norm": round(float(diag["bivector_norm"][0, 0]), 4),
            "above_critical": bool(diag["above_critical"][0, 0] > 0.5),
            "mode_amplitudes": {
                f"l{i+1}": round(float(diag["mode_amplitudes"][0, i]), 4)
                for i in range(N_MODES)
            },
        }

    return {
        "predictions": predictions,
        "diagnostics": diagnostics,
        "inputs": {"kp": round(kp, 2), "bz": round(bz, 1), "dst": round(dst, 1)},
        "model": {"params": 253, "val_mse": round(checkpoint["val_loss"], 4)},
    }


@app.get("/api/cosmic_rays_global")
async def get_cosmic_rays_global():
    """
    Global cosmic ray neutron monitor data from Nagoya WDC-CR + NMDB.

    Multiple stations for better Forbush decrease detection and
    geographic coverage of atmospheric ionization changes.
    """
    stations = {
        "OULU": {"lat": 65.05, "lon": 25.47, "alt": 15, "cutoff": 0.81},
        "MOSCOW": {"lat": 55.47, "lon": 37.32, "alt": 200, "cutoff": 2.43},
        "KIEL": {"lat": 54.34, "lon": 10.12, "alt": 54, "cutoff": 2.36},
        "ROME": {"lat": 41.86, "lon": 12.47, "alt": 60, "cutoff": 6.27},
        "CLIMAX": {"lat": 39.37, "lon": -106.18, "alt": 3400, "cutoff": 2.97},
        "THULE": {"lat": 76.50, "lon": -68.70, "alt": 260, "cutoff": 0.30},
        "JUNGFR": {"lat": 46.55, "lon": 7.98, "alt": 3475, "cutoff": 4.48},
        "MCMURD": {"lat": -77.85, "lon": 166.72, "alt": 48, "cutoff": 0.30},
    }

    # Try NMDB API for real-time data
    result = {"stations": {}, "global_mean": None, "forbush": False}

    try:
        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%S")
        end = now.strftime("%Y-%m-%dT%H:%M:%S")

        for code, info in list(stations.items())[:4]:  # limit to 4 for speed
            cache_key = f"cr_global_{code}"
            if cache_key in CACHE and time.time() - CACHE[cache_key]["ts"] < 600:
                result["stations"][code] = CACHE[cache_key]["data"]
                continue

            try:
                url = f"https://www.nmdb.eu/nest/draw_graph.php?stations[]={code}&tabchoice=revori&dtype=corr_for_efficiency&tresolution=60&force=1&startdate={start}&enddate={end}&output=ascii"
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        lines = resp.text.strip().split("\n")
                        data_lines = [l for l in lines if l and not l.startswith("#") and ";" in l]
                        values = []
                        for line in data_lines[-72:]:  # last 72 hours
                            parts = line.split(";")
                            if len(parts) >= 2:
                                try:
                                    values.append(float(parts[1].strip()))
                                except ValueError:
                                    pass

                        if values:
                            mean_val = sum(values) / len(values)
                            last_val = values[-1]
                            dev = (last_val - mean_val) / mean_val * 100 if mean_val > 0 else 0
                            station_data = {
                                "current": round(last_val, 1),
                                "mean_72h": round(mean_val, 1),
                                "deviation_pct": round(dev, 2),
                                "n_points": len(values),
                                **info,
                            }
                            result["stations"][code] = station_data
                            CACHE[cache_key] = {"data": station_data, "ts": time.time()}
            except Exception:
                pass

    except Exception:
        pass

    # Compute global mean and Forbush detection
    if result["stations"]:
        devs = [s["deviation_pct"] for s in result["stations"].values()]
        result["global_mean"] = round(sum(devs) / len(devs), 2)
        result["forbush"] = any(d < -3 for d in devs)
        result["n_stations"] = len(result["stations"])

    return result


@app.get("/api/tec")
async def get_tec():
    """
    Total Electron Content from NOAA GLOTEC/USTEC models.

    TEC anomalies over seismically active regions are a potential
    pre-earthquake ionospheric signature (Freund p-hole mechanism).
    """
    # NCEI HAPI endpoint for GLOTEC
    base = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1"

    try:
        # Get USTEC parameters
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{base}/hapi/info?dataset=ustec")
            if resp.status_code == 200:
                info = resp.json()
                params = [p["name"] for p in info.get("parameters", []) if p["name"] != "time"]
                return {
                    "available": True,
                    "dataset": "ustec",
                    "parameters": params[:10],
                    "start_date": info.get("startDate"),
                    "stop_date": info.get("stopDate"),
                    "note": "USTEC provides US Total Electron Content maps. Use /api/tec/data for time series.",
                }
    except Exception as e:
        pass

    return {"available": False, "note": "NCEI TEC data not accessible. Check https://www.ncei.noaa.gov/cloud-access/space-weather-portal/"}


@app.get("/api/enlil")
async def get_enlil():
    """
    WSA-Enlil solar wind prediction model output from SWPC.

    Provides predicted solar wind speed, density, and IMF at Earth
    for the next 4 days — useful for CME arrival prediction.
    """
    base = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Get ENLIL background parameters
            resp = await client.get(f"{base}/hapi/info?dataset=swpc_wsaenlil_bkg")
            if resp.status_code == 200:
                info = resp.json()
                params = [p["name"] for p in info.get("parameters", []) if p["name"] != "time"]
                return {
                    "available": True,
                    "datasets": {
                        "background": "swpc_wsaenlil_bkg",
                        "cme": "swpc_wsaenlil_cme",
                    },
                    "parameters": params[:15],
                    "start_date": info.get("startDate"),
                    "stop_date": info.get("stopDate"),
                    "api_base": base,
                    "note": "WSA-Enlil solar wind prediction. Use HAPI /data endpoint for time series.",
                }
    except Exception:
        pass

    return {"available": False}


@app.get("/api/precipitation")
async def get_precipitation():
    """
    Global precipitation from Open-Meteo (free, no API key).

    Returns 72h precipitation history for key seismic/atmospheric zones.
    Precipitation is relevant because:
    - Rainfall increases pore pressure in shallow faults
    - Heavy rain correlates with shallow landslide-triggered seismicity
    - The global electric circuit couples thunderstorms -> ionosphere -> Jz
    """
    stations = [
        {"name": "Indonesia (Molucca)", "lat": 1.0, "lon": 125.0},
        {"name": "Japan (Kanto)", "lat": 36.0, "lon": 140.0},
        {"name": "Chile (Santiago)", "lat": -33.4, "lon": -70.6},
        {"name": "California", "lat": 34.0, "lon": -118.2},
        {"name": "Vanuatu", "lat": -17.7, "lon": 168.3},
        {"name": "Turkey", "lat": 39.9, "lon": 32.9},
        {"name": "Central US (Tornado)", "lat": 35.0, "lon": -97.0},
        {"name": "India (Monsoon)", "lat": 20.0, "lon": 77.0},
    ]

    results = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for st in stations:
                cache_key = f"precip_{st['name']}"
                if cache_key in CACHE and time.time() - CACHE[cache_key]["ts"] < 600:
                    results.append(CACHE[cache_key]["data"])
                    continue
                try:
                    url = (f"https://api.open-meteo.com/v1/forecast?"
                           f"latitude={st['lat']}&longitude={st['lon']}"
                           f"&hourly=precipitation,weathercode,windspeed_10m,winddirection_10m"
                           f"&past_days=3&forecast_days=0")
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        d = resp.json()
                        hourly = d.get("hourly", {})
                        precip = hourly.get("precipitation", [])
                        codes = hourly.get("weathercode", [])
                        wind_speeds = hourly.get("windspeed_10m", [])
                        wind_dirs = hourly.get("winddirection_10m", [])
                        total_72h = sum(p for p in precip if p) if precip else 0
                        current = precip[-1] if precip else 0
                        wind_speed = wind_speeds[-1] if wind_speeds else 0
                        wind_dir = wind_dirs[-1] if wind_dirs else 0
                        # Thunderstorm detection: WMO codes 95-99
                        thunder_hours = sum(1 for c in codes if c and c >= 95)
                        entry = {
                            **st,
                            "total_72h_mm": round(total_72h, 1),
                            "current_mm": current,
                            "thunder_hours": thunder_hours,
                            "n_hours": len(precip),
                            "wind_speed_kmh": round(wind_speed, 1) if wind_speed is not None else None,
                            "wind_dir_deg": round(wind_dir, 0) if wind_dir is not None else None,
                        }
                        results.append(entry)
                        CACHE[cache_key] = {"data": entry, "ts": time.time()}
                except Exception:
                    pass
    except Exception:
        pass

    # Global summary
    total_global = sum(r.get("total_72h_mm", 0) for r in results)
    thunder_total = sum(r.get("thunder_hours", 0) for r in results)

    return {
        "stations": results,
        "global_precip_72h": round(total_global, 1),
        "global_thunder_hours": thunder_total,
        "n_stations": len(results),
    }


@app.get("/api/wind_field")
async def get_wind_field():
    """
    Global wind field from Open-Meteo (batch multi-point requests).

    Returns coarse global grid of 10m wind (u,v) for animation.
    """
    cache_key = "wind_field"
    if cache_key in CACHE and time.time() - CACHE[cache_key]["ts"] < 1200:
        return CACHE[cache_key]["data"]

    step = 7.5  # degrees
    lats = [round(-82.5 + i * step, 1) for i in range(int(165 / step) + 1)]
    lons = [round(-180 + i * step, 1) for i in range(int(360 / step))]
    async def fetch_lat_row(client, lat):
        lat_list = ",".join(f"{lat:.1f}" for _ in lons)
        lon_list = ",".join(f"{lon:.1f}" for lon in lons)
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat_list}&longitude={lon_list}"
               f"&hourly=windspeed_10m,winddirection_10m"
               f"&past_days=0&forecast_days=1")
        resp = await client.get(url)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data if isinstance(data, list) else [data]

    results = {}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            tasks = [fetch_lat_row(client, lat) for lat in lats]
            batches = await asyncio.gather(*tasks, return_exceptions=True)
            for batch in batches:
                if isinstance(batch, Exception):
                    continue
                for item in batch:
                    lat = round(float(item.get("latitude")), 1)
                    lon = round(float(item.get("longitude")), 1)
                    hourly = item.get("hourly", {})
                    speeds = hourly.get("windspeed_10m", [])
                    dirs = hourly.get("winddirection_10m", [])
                    if not speeds or not dirs:
                        continue
                    speed_kmh = speeds[-1]
                    dir_from = dirs[-1]
                    if speed_kmh is None or dir_from is None:
                        continue
                    speed_mps = speed_kmh / 3.6
                    dir_to = (dir_from + 180) % 360
                    rad = math.radians(dir_to)
                    u = speed_mps * math.sin(rad)
                    v = speed_mps * math.cos(rad)
                    results[(lat, lon)] = (round(u, 3), round(v, 3), round(speed_mps, 3))
    except Exception:
        pass

    u_list = []
    v_list = []
    s_list = []
    for lat in lats:
        for lon in lons:
            val = results.get((lat, lon))
            if not val:
                u_list.append(None)
                v_list.append(None)
                s_list.append(None)
            else:
                u_list.append(val[0])
                v_list.append(val[1])
                s_list.append(val[2])

    u_list, v_list, s_list = fill_vector_grid(u_list, v_list, len(lats), len(lons))

    payload = {
        "source": "Open-Meteo",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid": {
            "resolution_deg": step,
            "lats": lats,
            "lons": lons,
            "u": u_list,
            "v": v_list,
            "speed": s_list,
            "units": {"u": "m/s", "v": "m/s", "speed": "m/s"},
        },
    }
    CACHE[cache_key] = {"data": payload, "ts": time.time()}
    return payload


@app.get("/api/ocean_currents")
def get_ocean_currents():
    """
    Global surface ocean currents (geostrophic) from NOAA CoastWatch ERDDAP.
    """
    cache_key = "ocean_currents"
    if cache_key in CACHE and time.time() - CACHE[cache_key]["ts"] < 6 * 3600:
        return CACHE[cache_key]["data"]

    base = "https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDNRTcurrentsDaily"
    time_str = None
    step = 5.0
    stride = int(step / 0.25)
    lat_min, lat_max = -80, 80
    lon_min, lon_max = -180, 180

    try:
        with httpx.Client(timeout=25) as client:
            das = client.get(f"{base}.das")
            if das.status_code == 200:
                for line in das.text.splitlines():
                    if "time_coverage_end" in line:
                        parts = line.split('"')
                        if len(parts) >= 2:
                            time_str = parts[1].strip()
                            break

            if not time_str:
                time_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")

            query = (
                f"u_current[({time_str})][({lat_min}):{stride}:({lat_max})][({lon_min}):{stride}:({lon_max})],"
                f"v_current[({time_str})][({lat_min}):{stride}:({lat_max})][({lon_min}):{stride}:({lon_max})]"
            )
            data_url = f"{base}.csv?{query}"
            resp = client.get(data_url)
            if resp.status_code != 200:
                return CACHE.get(cache_key, {}).get("data") or {"error": "ocean_current_fetch_failed"}

            rows = resp.text.splitlines()
            if len(rows) < 3:
                return CACHE.get(cache_key, {}).get("data") or {"error": "ocean_current_empty"}

            import csv
            reader = csv.reader(rows[2:])
            results = {}
            lats_set = set()
            lons_set = set()
            for r in reader:
                if len(r) < 5:
                    continue
                lat = round(float(r[1]), 3)
                lon = round(float(r[2]), 3)
                u = float(r[3])
                v = float(r[4])
                if abs(u) > 1e5 or abs(v) > 1e5:
                    continue
                if not math.isfinite(u) or not math.isfinite(v):
                    continue
                if u == -214748.3648 or v == -214748.3648:
                    continue
                results[(lat, lon)] = (round(u, 3), round(v, 3))
                lats_set.add(lat)
                lons_set.add(lon)

            lats = sorted(lats_set)
            lons = sorted(lons_set)
            u_list = []
            v_list = []
            for lat in lats:
                for lon in lons:
                    val = results.get((lat, lon))
                    if not val:
                        u_list.append(None)
                        v_list.append(None)
                    else:
                        u_list.append(val[0])
                        v_list.append(val[1])

            payload = {
                "source": "NOAA CoastWatch ERDDAP (noaacwBLENDEDNRTcurrentsDaily)",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time": time_str,
                "grid": {
                    "resolution_deg": step,
                    "lats": lats,
                    "lons": lons,
                    "u": u_list,
                    "v": v_list,
                    "units": {"u": "m/s", "v": "m/s"},
                },
            }
            CACHE[cache_key] = {"data": payload, "ts": time.time()}
            return payload
    except Exception:
        return CACHE.get(cache_key, {}).get("data") or {"error": "ocean_current_exception"}


@app.get("/api/lightning")
def get_lightning():
    """
    Lightning activity from WWLLN climatology + thunderstorm detection.

    Uses WWLLN 2010-2022 monthly climatology for baseline comparison,
    plus Open-Meteo thunderstorm codes for real-time storm detection.

    Lightning is the primary driver of Schumann resonances:
    ~2000 active thunderstorms globally at any time drive the cavity.
    """
    # Load WWLLN monthly climatology for current month
    lightning_file = DATA_DIR / "lightning" / "wglc_climatology_30m_monthly.nc"
    result = {"source": "WWLLN climatology + Open-Meteo thunderstorm codes"}

    if lightning_file.exists():
        try:
            import xarray as xr
            ds = xr.open_dataset(str(lightning_file))
            now = datetime.now(timezone.utc)
            month = now.month

            # Get climatological flash density for this month
            if "time" in ds.dims:
                month_data = ds["density"].isel(time=month - 1)
            else:
                month_data = ds["density"]

            # Global statistics
            total = float(month_data.sum())
            mean_density = float(month_data.mean())

            # Regional hotspots
            regions = [
                ("Central Africa", -5, 5, 15, 35),
                ("Amazon", -10, 5, -70, -45),
                ("Maritime Continent", -10, 10, 95, 140),
                ("Central US", 30, 40, -100, -85),
                ("India", 10, 30, 70, 90),
            ]
            hotspots = []
            for name, la1, la2, lo1, lo2 in regions:
                regional = month_data.sel(lat=slice(la1, la2), lon=slice(lo1, lo2))
                hotspots.append({
                    "name": name,
                    "mean_density": round(float(regional.mean()), 4),
                })

            result["climatology"] = {
                "month": now.strftime("%B"),
                "global_total": round(total, 0),
                "global_mean_density": round(mean_density, 6),
                "hotspots": hotspots,
            }
            ds.close()
        except Exception as e:
            result["climatology_error"] = str(e)

    # Real-time thunderstorm count from cached precipitation data
    thunder_total = 0
    for key, val in CACHE.items():
        if key.startswith("precip_") and "data" in val:
            thunder_total += val["data"].get("thunder_hours", 0)
    result["realtime_thunder_hours"] = thunder_total

    return result


@app.get("/api/cloud_charge")
async def get_cloud_charge():
    """
    Cloud cover and charge gradient model.

    Cloud layers are the charge SOURCE for the global electric circuit:
    - Cumulonimbus: dipole (negative base ~3km, positive top ~10km)
    - Stratiform: weak negative sheet
    - Clear: no local charge (fair weather Ez from distant storms)

    Returns cloud cover at 3 altitudes + estimated charge density +
    Carnegie curve Ez + convective energy (CAPE) for each station.
    """
    stations = [
        {"name": "Indonesia", "lat": 1.0, "lon": 125.0},
        {"name": "Japan", "lat": 36.0, "lon": 140.0},
        {"name": "Chile", "lat": -33.4, "lon": -70.6},
        {"name": "California", "lat": 34.0, "lon": -118.2},
        {"name": "Vanuatu", "lat": -17.7, "lon": 168.3},
        {"name": "Central US", "lat": 35.0, "lon": -97.0},
        {"name": "C. Africa", "lat": 0.0, "lon": 25.0},
        {"name": "Amazon", "lat": -3.0, "lon": -60.0},
        {"name": "India", "lat": 20.0, "lon": 77.0},
        {"name": "Australia", "lat": -25.0, "lon": 134.0},
    ]

    results = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for st in stations:
                cache_key = f"cloud_{st['name']}"
                if cache_key in CACHE and time.time() - CACHE[cache_key]["ts"] < 600:
                    results.append(CACHE[cache_key]["data"])
                    continue
                try:
                    url = (f"https://api.open-meteo.com/v1/forecast?"
                           f"latitude={st['lat']}&longitude={st['lon']}"
                           f"&hourly=cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,"
                           f"precipitation,weathercode,cape"
                           f"&past_days=1&forecast_days=0")
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        continue
                    d = resp.json().get("hourly", {})
                    cc = d.get("cloudcover", [])
                    cc_low = d.get("cloudcover_low", [])
                    cc_mid = d.get("cloudcover_mid", [])
                    cc_high = d.get("cloudcover_high", [])
                    precip = d.get("precipitation", [])
                    codes = d.get("weathercode", [])
                    cape_vals = d.get("cape", [])

                    # Current values (last hour)
                    cur_cc = cc[-1] if cc else 0
                    cur_low = cc_low[-1] if cc_low else 0
                    cur_mid = cc_mid[-1] if cc_mid else 0
                    cur_high = cc_high[-1] if cc_high else 0
                    cur_precip = precip[-1] if precip else 0
                    cur_code = codes[-1] if codes else 0
                    cur_cape = cape_vals[-1] if cape_vals else 0

                    # Cloud charge estimation
                    # Thunderstorm (WMO 95-99): ~30 C dipole per active cell
                    # Heavy rain (80-84): ~5 C weak charge separation
                    # Stratiform (61-67): ~0.5 C per layer
                    # Fair weather: 0 C local
                    is_thunder = cur_code and cur_code >= 95
                    is_heavy = cur_code and 80 <= cur_code <= 84
                    is_rain = cur_code and 60 <= cur_code <= 67

                    if is_thunder:
                        charge_c = 30 * max(1, cur_precip / 10)
                        charge_type = "Cb dipole"
                    elif is_heavy:
                        charge_c = 5 * max(1, cur_precip / 5)
                        charge_type = "convective"
                    elif is_rain:
                        charge_c = 0.5
                        charge_type = "stratiform"
                    else:
                        charge_c = 0
                        charge_type = "fair weather"

                    # Ez estimate: Carnegie curve + cloud modulation
                    now_utc = datetime.now(timezone.utc)
                    hour = now_utc.hour + now_utc.minute / 60
                    ez_carnegie = 130 * (1 + 0.15 * math.cos(2 * math.pi * (hour - 19) / 24))
                    # Thunderstorms enhance local Ez by 10-100x
                    ez_local = ez_carnegie
                    if is_thunder:
                        ez_local = ez_carnegie * (5 + cur_precip)
                    elif is_heavy:
                        ez_local = ez_carnegie * 2
                    # Overcast reduces Ez (conduction through cloud layer)
                    elif cur_cc and cur_cc > 80:
                        ez_local = ez_carnegie * 0.7

                    # Charge gradient dQ/dz (C/m per km altitude)
                    # Low clouds: 1-3 km, Mid: 3-6 km, High: 6-12 km
                    gradient = {
                        "low_1_3km": round(-charge_c * cur_low / 100 * 0.3, 2),   # negative base
                        "mid_3_6km": round(charge_c * cur_mid / 100 * 0.1, 2),    # transition
                        "high_6_12km": round(charge_c * cur_high / 100 * 0.5, 2), # positive top
                    }

                    entry = {
                        **st,
                        "cloud_cover": {"total": cur_cc, "low": cur_low, "mid": cur_mid, "high": cur_high},
                        "charge_c": round(charge_c, 1),
                        "charge_type": charge_type,
                        "charge_gradient": gradient,
                        "ez_v_m": round(ez_local, 1),
                        "ez_carnegie": round(ez_carnegie, 1),
                        "cape_j_kg": cur_cape,
                        "precip_mm_hr": cur_precip,
                        "weathercode": cur_code,
                    }
                    results.append(entry)
                    CACHE[cache_key] = {"data": entry, "ts": time.time()}
                except Exception:
                    pass
    except Exception:
        pass

    # Global charge budget
    total_charge = sum(r.get("charge_c", 0) for r in results)
    thunder_count = sum(1 for r in results if r.get("charge_type") == "Cb dipole")

    return {
        "stations": results,
        "global_charge_c": round(total_charge, 1),
        "active_thunderstorms": thunder_count,
        "n_stations": len(results),
        "carnegie_hour": round(datetime.now(timezone.utc).hour + datetime.now(timezone.utc).minute / 60, 1),
    }


@app.get("/api/pore_pressure")
def get_pore_pressure():
    """
    Pore pressure model: rainfall + telluric + tidal -> effective stress.

    Computes pore pressure change at fault depths from:
    1. Rainfall infiltration (1D diffusion from cached precipitation)
    2. Telluric current Jz (from Kp/Bz via global electric circuit)
    3. Lunar tidal body force (fortnightly M2)

    Output: delta_sigma_eff / sigma_tectonic at each monitoring station.
    """
    import math as m

    rho_w = 1000  # kg/m3
    g = 9.81
    sigma_tectonic = 1e6  # 1 MPa typical

    def erfc_approx(x):
        if x > 5: return 0
        if x < -5: return 2
        t = 1 / (1 + 0.3275911 * abs(x))
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        result = poly * m.exp(-x * x)
        return result if x >= 0 else 2 - result

    # Get current inputs
    kp = 2.0
    kp_data = CACHE.get("kp", {}).get("data")
    if kp_data:
        for row in reversed(kp_data if isinstance(kp_data, list) else []):
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                try: kp = float(row[1]); break
                except: pass

    bz = 0.0
    sw_data = CACHE.get("sw_mag", {}).get("data")
    if sw_data:
        for row in reversed(sw_data[1:]):
            v = row[3] if len(row) > 3 else None
            if v not in (None, "", "null"):
                try: bz = float(v); break
                except: pass

    # Lunar tidal force
    now = datetime.now(timezone.utc)
    ref = datetime(2000, 1, 6, tzinfo=timezone.utc)
    days_since = (now - ref).total_seconds() / 86400
    lunar_phase = (days_since % 29.53059) / 29.53059
    tidal_force = m.cos(2 * m.pi * lunar_phase)  # +1 at new/full, -1 at quarters
    tidal_stress_pa = 1000 * abs(tidal_force)  # ~1kPa max body tide stress

    # Telluric Jz contribution (from Kp/Bz)
    telluric_j = 2.0 * m.exp(kp * 0.4)  # mA/km
    if bz < -10: telluric_j *= 1.5
    # Lorentz force on pore fluid: F = J x B, integrated over depth
    B_surface = 50e-6  # T
    telluric_pressure_pa = telluric_j * 1e-3 * B_surface * 10000  # rough integral over 10km

    stations = []
    depths = [10, 50, 100, 500, 1000, 5000]  # meters
    D_default = 0.1  # fractured rock

    for key, val in CACHE.items():
        if not key.startswith("precip_") or "data" not in val:
            continue
        st = val["data"]
        rain_mm = st.get("total_72h_mm", 0)
        rain_m = rain_mm / 1000
        P0_rain = rho_w * g * rain_m  # surface head from rain
        t_sec = 72 * 3600  # 72h

        depth_profile = {}
        for depth in depths:
            # Rain pore pressure at depth
            arg = depth / m.sqrt(4 * D_default * t_sec) if t_sec > 0 else 999
            p_rain = P0_rain * erfc_approx(arg)
            # Total pore pressure change
            p_total = p_rain + telluric_pressure_pa + tidal_stress_pa
            # Fraction of tectonic stress
            frac = p_total / sigma_tectonic

            depth_label = f"{depth}m" if depth < 1000 else f"{depth // 1000}km"
            depth_profile[depth_label] = {
                "rain_pa": round(p_rain, 2),
                "telluric_pa": round(telluric_pressure_pa, 2),
                "tidal_pa": round(tidal_stress_pa, 2),
                "total_pa": round(p_total, 2),
                "pct_tectonic": round(frac * 100, 5),
            }

        stations.append({
            "name": st.get("name", "?"),
            "lat": st.get("lat"), "lon": st.get("lon"),
            "rain_mm_72h": rain_mm,
            "depth_profile": depth_profile,
        })

    return {
        "stations": stations,
        "inputs": {
            "kp": round(kp, 2),
            "bz": round(bz, 1),
            "lunar_phase": round(lunar_phase, 3),
            "tidal_force": round(tidal_force, 3),
            "telluric_j_mA_km": round(telluric_j, 2),
        },
        "model": {
            "diffusivity_m2s": D_default,
            "surface_B_uT": 50,
            "tectonic_stress_MPa": 1,
        },
    }


@app.get("/api/ocean_light_phenomena")
def get_ocean_light_phenomena():
    """Historical reports of te lapa, St. Elmo's fire, and related
    ocean electromagnetic light phenomena, mapped to ocean currents."""
    import math

    reports = [
        # Te lapa observations
        {"name": "Te lapa — Solomon Islands", "lat": -9.4, "lon": 160.0,
         "type": "te_lapa", "year": "traditional", "observer": "Kaveia & George",
         "current": "South Equatorial Current",
         "desc": "Navigation lights pointing toward islands, 0.5-1.8m depth, up to 130 km range"},
        {"name": "Te lapa — Tonga", "lat": -21.2, "lon": -175.2,
         "type": "te_lapa", "year": "traditional", "observer": "Tongan navigators",
         "current": "South Equatorial / Pacific gyre",
         "desc": "Called 'te tapa' (to burst forth with light) or 'ulo aetahi' (Glory of the Seas)"},
        {"name": "Te lapa — Nikunau (Kiribati)", "lat": -1.35, "lon": 176.45,
         "type": "te_lapa", "year": "traditional", "observer": "Gilbertese navigators",
         "current": "Equatorial Counter-Current",
         "desc": "Called 'te mata' — navigation flashes near reef islands"},
        {"name": "Te lapa — Tikopia", "lat": -12.3, "lon": 168.8,
         "type": "te_lapa", "year": "1972", "observer": "David Lewis",
         "current": "South Equatorial Current",
         "desc": "Lewis documented in 'We, the Navigators' — Tikopians reportedly unaware"},
        {"name": "Te lapa — Vaeakau-Taumako", "lat": -9.8, "lon": 167.0,
         "type": "te_lapa", "year": "1993-2003", "observer": "Marianne George",
         "current": "South Equatorial Current",
         "desc": "George observed with Kaveia — streaks, flashes, glowing plaques"},

        # St. Elmo's fire at sea
        {"name": "Columbus, 1492", "lat": 28.0, "lon": -65.0,
         "type": "st_elmo", "year": "1492", "observer": "Christopher Columbus",
         "current": "Gulf Stream / N Atlantic gyre",
         "desc": "Observed crossing the North Atlantic — crew terrified then reassured"},
        {"name": "Magellan / Pigafetta, 1519", "lat": -35.0, "lon": -52.0,
         "type": "st_elmo", "year": "1519", "observer": "Antonio Pigafetta",
         "current": "Brazil-Falkland convergence",
         "desc": "Documented off South America during circumnavigation"},
        {"name": "Bligh, HMS Bounty, 1788", "lat": -42.6, "lon": -34.6,
         "type": "st_elmo", "year": "1788", "observer": "William Bligh",
         "current": "Brazil-Falkland + ACC",
         "desc": "Corpo-Sant on yard arms, 42°34'S 34°38'W, tropical squalls"},
        {"name": "Noah, Hillsborough #1, 1799", "lat": -45.0, "lon": 30.0,
         "type": "st_elmo", "year": "1799", "observer": "William Noah",
         "current": "Antarctic Circumpolar Current",
         "desc": "Southern Ocean between Cape Town and Sydney"},
        {"name": "Noah, Hillsborough #2, 1799", "lat": -35.0, "lon": 155.0,
         "type": "st_elmo", "year": "1799", "observer": "William Noah",
         "current": "East Australian Current",
         "desc": "Tasman Sea near Port Jackson (Sydney)"},
        {"name": "Darwin, HMS Beagle, 1832", "lat": -35.0, "lon": -56.0,
         "type": "st_elmo", "year": "1832", "observer": "Charles Darwin",
         "current": "Brazil-Falkland convergence",
         "desc": "Masts pointed with blue flame; sea luminous, penguin tracks fiery"},
        {"name": "Air France 447, 2009", "lat": 2.0, "lon": -30.0,
         "type": "st_elmo", "year": "2009", "observer": "Flight crew",
         "current": "N Equatorial Counter-Current / ITCZ",
         "desc": "Appeared 23 min before Atlantic crash; ITCZ thunderstorm"},

        # Earthquake lights at sea / coastal
        {"name": "Japan Trench EQ lights", "lat": 38.3, "lon": 142.4,
         "type": "eq_light", "year": "2011", "observer": "Multiple",
         "current": "Kuroshio extension",
         "desc": "Blue-white flashes reported offshore before Tōhoku M9.1"},
        {"name": "Chile coastal lights", "lat": -36.1, "lon": -72.9,
         "type": "eq_light", "year": "2010", "observer": "Multiple",
         "current": "Humboldt Current",
         "desc": "Lights reported before Maule M8.8 earthquake"},
    ]

    # Add ocean current paths for rendering
    currents = [
        {"name": "Gulf Stream", "color": "#ff4444",
         "path": [[-80,25],[-75,35],[-60,40],[-40,45],[-20,50]]},
        {"name": "Kuroshio", "color": "#ff6644",
         "path": [[125,25],[130,30],[140,35],[155,38],[170,40]]},
        {"name": "S. Equatorial (Pacific)", "color": "#44aaff",
         "path": [[-120,-10],[-140,-8],[-160,-8],[-180,-10],[175,-10],[165,-10]]},
        {"name": "Brazil Current", "color": "#ff8844",
         "path": [[-40,-10],[-42,-15],[-45,-20],[-48,-25],[-50,-30],[-52,-35]]},
        {"name": "Falkland Current", "color": "#4488ff",
         "path": [[-65,-55],[-62,-50],[-58,-45],[-55,-40],[-52,-35]]},
        {"name": "ACC", "color": "#88ccff",
         "path": [[-60,-55],[-30,-55],[0,-55],[30,-55],[60,-55],[90,-55],[120,-55],[150,-55],[180,-55]]},
        {"name": "Agulhas", "color": "#ff4488",
         "path": [[35,-30],[32,-33],[28,-35],[25,-36],[22,-37]]},
        {"name": "E. Australian", "color": "#ff6688",
         "path": [[154,-25],[153,-30],[152,-33],[153,-35],[155,-38]]},
        {"name": "Equatorial Counter-Current", "color": "#44ccaa",
         "path": [[-170,5],[-150,5],[-130,5],[-110,5]]},
    ]

    return {"reports": reports, "currents": currents}


@app.get("/api/magnetic_anomalies")
def get_magnetic_anomalies():
    """Major crustal magnetic anomalies from ore deposits and BIFs.

    Each anomaly has: lat, lon, strength_nT, conductivity, area_km2,
    deposit type, and Schumann interaction regime.
    """
    import math
    PI = math.pi
    MU0 = 4 * PI * 1e-7
    deposits = {
        "Bayan Obo": {
            "lat": 41.8, "lon": 109.97, "strength_nT": 1500,
            "conductivity_Sm": 0.1, "area_km2": 48,
            "type": "REE-Fe carbonatite", "country": "China",
            "ore_Mt": 1500, "ree_Mt": 48, "magnetite_pct": 35,
        },
        "Kiruna": {
            "lat": 67.86, "lon": 20.22, "strength_nT": 5000,
            "conductivity_Sm": 1.0, "area_km2": 80,
            "type": "Apatite iron ore", "country": "Sweden",
            "ore_Mt": 2500, "ree_Mt": 1.0, "magnetite_pct": 65,
        },
        "Kursk Magnetic Anomaly": {
            "lat": 51.7, "lon": 37.5, "strength_nT": 3000,
            "conductivity_Sm": 0.5, "area_km2": 120000,
            "type": "Banded iron formation", "country": "Russia",
            "ore_Mt": 30000, "ree_Mt": 0, "magnetite_pct": 40,
        },
        "Bushveld Complex": {
            "lat": -25.5, "lon": 29.0, "strength_nT": 1000,
            "conductivity_Sm": 0.3, "area_km2": 65000,
            "type": "Layered mafic intrusion", "country": "South Africa",
            "ore_Mt": 5000, "ree_Mt": 0.1, "magnetite_pct": 25,
        },
        "Palabora": {
            "lat": -23.68, "lon": 31.12, "strength_nT": 800,
            "conductivity_Sm": 0.08, "area_km2": 20,
            "type": "Carbonatite complex", "country": "South Africa",
            "ore_Mt": 400, "ree_Mt": 0.5, "magnetite_pct": 20,
        },
        "Lovozero": {
            "lat": 67.83, "lon": 34.75, "strength_nT": 500,
            "conductivity_Sm": 0.05, "area_km2": 650,
            "type": "Alkaline intrusion", "country": "Russia",
            "ore_Mt": 180, "ree_Mt": 7.0, "magnetite_pct": 12,
        },
        "Mountain Pass": {
            "lat": 35.48, "lon": -115.53, "strength_nT": 200,
            "conductivity_Sm": 0.01, "area_km2": 3,
            "type": "Carbonatite REE", "country": "USA",
            "ore_Mt": 20, "ree_Mt": 2.4, "magnetite_pct": 5,
        },
        "Mount Weld": {
            "lat": -28.77, "lon": 122.55, "strength_nT": 300,
            "conductivity_Sm": 0.02, "area_km2": 5,
            "type": "Laterite over carbonatite", "country": "Australia",
            "ore_Mt": 24, "ree_Mt": 2.5, "magnetite_pct": 8,
        },
        "Ilimaussaq": {
            "lat": 60.95, "lon": -46.0, "strength_nT": 150,
            "conductivity_Sm": 0.01, "area_km2": 136,
            "type": "Agpaitic intrusion", "country": "Greenland",
            "ore_Mt": 60, "ree_Mt": 6.6, "magnetite_pct": 3,
        },
        "Carajas": {
            "lat": -6.07, "lon": -50.17, "strength_nT": 2000,
            "conductivity_Sm": 0.4, "area_km2": 400,
            "type": "Iron oxide (BIF)", "country": "Brazil",
            "ore_Mt": 18000, "ree_Mt": 0, "magnetite_pct": 50,
        },
        "Pilbara (Hamersley)": {
            "lat": -22.3, "lon": 118.3, "strength_nT": 1500,
            "conductivity_Sm": 0.3, "area_km2": 6000,
            "type": "Banded iron formation", "country": "Australia",
            "ore_Mt": 20000, "ree_Mt": 0, "magnetite_pct": 35,
        },
        "Labrador Trough": {
            "lat": 55.0, "lon": -66.5, "strength_nT": 1200,
            "conductivity_Sm": 0.3, "area_km2": 3000,
            "type": "BIF iron ore", "country": "Canada",
            "ore_Mt": 5000, "ree_Mt": 0, "magnetite_pct": 30,
        },
    }

    anomalies = []
    for name, d in deposits.items():
        sigma = d["conductivity_Sm"]
        body_km = math.sqrt(d["area_km2"])
        delta_km = math.sqrt(2 / (2 * PI * 7.83 * MU0 * sigma)) / 1000
        body_over_delta = body_km / delta_km
        if body_over_delta > 5:
            schumann = "scatterer"
        elif body_over_delta > 1:
            schumann = "absorber"
        else:
            schumann = "transparent"

        anomalies.append({
            "name": name,
            "lat": d["lat"], "lon": d["lon"],
            "strength_nT": d["strength_nT"],
            "conductivity_Sm": sigma,
            "area_km2": d["area_km2"],
            "type": d["type"],
            "country": d["country"],
            "ore_Mt": d["ore_Mt"],
            "ree_Mt": d.get("ree_Mt", 0),
            "magnetite_pct": d["magnetite_pct"],
            "skin_depth_7Hz_km": round(delta_km, 2),
            "body_over_skin": round(body_over_delta, 1),
            "schumann_regime": schumann,
        })

    return {"anomalies": sorted(anomalies, key=lambda a: -a["strength_nT"])}


@app.get("/api/paleomag")
def get_paleomag():
    """Bronze Age paleomagnetic field data from pfm9k.2."""
    json_file = Path(__file__).parent.parent / "frontend" / "assets" / "bronze_age_field.json"
    if json_file.exists():
        return json.loads(json_file.read_text())
    return {"error": "Run paleomag_plots.py first to generate data"}


@app.get("/api/plates")
def get_plates():
    """Labeled tectonic plate boundaries as GeoJSON FeatureCollection."""
    plates_file = Path(__file__).parent.parent / "frontend" / "src" / "plates.json"
    if not plates_file.exists():
        return {"type": "FeatureCollection", "features": []}

    segs = json.loads(plates_file.read_text())

    # Major plate boundary regions: (name, color, lon_min, lat_min, lon_max, lat_max, boundary_type)
    REGIONS = [
        ("Mid-Atlantic Ridge (N)", "#4466cc", -50, 10, -10, 70, "divergent"),
        ("Mid-Atlantic Ridge (S)", "#4488aa", -30, -60, 10, 10, "divergent"),
        ("East Pacific Rise", "#cc4466", -130, -60, -80, 20, "divergent"),
        ("Cascadia / Juan de Fuca", "#ff6644", -135, 38, -120, 52, "convergent"),
        ("San Andreas", "#ff8844", -125, 30, -115, 42, "transform"),
        ("Peru-Chile Trench", "#ff4444", -85, -55, -65, 0, "convergent"),
        ("Central America Trench", "#ff6622", -110, 5, -75, 22, "convergent"),
        ("Caribbean Arc", "#ff8866", -80, 10, -55, 22, "convergent"),
        ("Aleutian Trench", "#dd4444", 165, 48, -165, 58, "convergent"),
        ("Japan Trench", "#ee4444", 135, 25, 150, 45, "convergent"),
        ("Mariana Trench", "#dd2244", 140, 8, 150, 25, "convergent"),
        ("Philippine Trench", "#cc3344", 120, 5, 135, 22, "convergent"),
        ("Tonga-Kermadec", "#dd4466", 172, -38, -175, -12, "convergent"),
        ("Sunda Trench", "#ee4422", 90, -12, 125, 8, "convergent"),
        ("Himalayan Front", "#ff8800", 68, 24, 100, 38, "convergent"),
        ("Alpine-Mediterranean", "#ff9944", -10, 32, 50, 48, "convergent"),
        ("East African Rift", "#44aa66", 25, -20, 45, 15, "divergent"),
        ("Red Sea Rift", "#44cc66", 30, 10, 48, 30, "divergent"),
        ("Mid-Indian Ridge", "#4488cc", 50, -55, 100, -10, "divergent"),
        ("Southeast Indian Ridge", "#4466aa", 80, -65, 150, -35, "divergent"),
        ("Pacific-Antarctic Ridge", "#4488bb", -180, -68, -80, -50, "divergent"),
        ("Scotia Arc", "#886644", -70, -62, -25, -52, "transform"),
        ("Antarctic Plate (S)", "#668888", -180, -80, 180, -60, "divergent"),
        ("Ring of Fire (W Pacific)", "#ff5533", 105, -50, 170, 60, "convergent"),
        ("North American (W)", "#cc6644", -170, 50, -120, 72, "transform"),
    ]

    def classify_segment(seg):
        """Assign a name/color to a segment based on nearest region centroid."""
        lons = [p[0] for p in seg]
        lats = [p[1] for p in seg]
        clon = sum(lons) / len(lons)
        clat = sum(lats) / len(lats)

        # First pass: check if centroid is inside any bounding box
        for name, color, lo0, la0, lo1, la1, btype in REGIONS:
            if lo0 > lo1:
                in_lon = clon > lo0 or clon < lo1
            else:
                in_lon = lo0 <= clon <= lo1
            if in_lon and la0 <= clat <= la1:
                return (name, color, btype)

        # Second pass: nearest region centroid (no segment left behind)
        best = ("Other Boundary", "#556677", "unknown")
        best_dist = 1e9
        for name, color, lo0, la0, lo1, la1, btype in REGIONS:
            cx = (lo0 + lo1) / 2 if lo0 < lo1 else ((lo0 + lo1 + 360) / 2) % 360 - 180
            cy = (la0 + la1) / 2
            # Longitude distance handling wrap-around
            dlon = abs(clon - cx)
            if dlon > 180:
                dlon = 360 - dlon
            d = math.sqrt(dlon ** 2 + (clat - cy) ** 2)
            if d < best_dist:
                best_dist = d
                best = (name, color, btype)
        return best

    features = []
    for i, seg in enumerate(segs):
        if len(seg) < 2:
            continue
        name, color, btype = classify_segment(seg)
        features.append({
            "type": "Feature",
            "properties": {
                "name": name,
                "color": color,
                "boundary_type": btype,
                "segment_id": i,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": seg,
            },
        })

    # Summary of plates loaded
    plate_names = sorted(set(f["properties"]["name"] for f in features))

    return {
        "type": "FeatureCollection",
        "features": features,
        "summary": {
            "total_segments": len(features),
            "named_boundaries": plate_names,
        },
    }


@app.get("/api/seismic/waveform")
async def get_seismic_waveform(station: str = "ANMO", network: str = "IU",
                                channel: str = "BHZ", duration: int = 600):
    """Live seismogram data from IRIS FDSN."""
    now = datetime.now(timezone.utc)
    # IRIS data has ~5-30 min latency; offset window back to ensure data exists
    end = now - timedelta(minutes=5)
    start = end - timedelta(seconds=duration)
    url = (
        f"https://service.iris.edu/irisws/timeseries/1/query?"
        f"net={network}&sta={station}&loc=00&cha={channel}"
        f"&starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&endtime={end.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&output=ascii"
    )
    cache_key = f"seismo_{station}_{channel}"
    now_ts = time.time()
    if cache_key in CACHE and now_ts - CACHE[cache_key]["ts"] < 120:
        return CACHE[cache_key]["data"]

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return {"error": "No data available", "station": station}
            text = resp.text
    except Exception as e:
        return {"error": str(e), "station": station}

    # Parse IRIS ASCII: TSPAIR format (timestamp  value) or simple columns
    lines = text.strip().split("\n")
    samples = []
    sample_rate = 20.0
    start_time = ""
    for line in lines:
        if line.startswith("TIMESERIES"):
            parts = line.split(",")
            for p in parts:
                p = p.strip()
                if "sps" in p:
                    try:
                        sample_rate = float(p.split()[0])
                    except Exception:
                        pass
                if "20" in p and "T" in p and "-" in p:
                    start_time = p.strip()
        else:
            try:
                vals = line.strip().split()
                # TSPAIR: "2026-04-04T14:00:00.019538  164"
                if len(vals) == 2 and "T" in vals[0]:
                    samples.append(int(vals[1]))
                else:
                    # Plain integer columns
                    for v in vals:
                        samples.append(int(v))
            except (ValueError, IndexError):
                pass

    # Downsample for frontend (target ~500 points)
    target = 500
    step = max(1, len(samples) // target)
    decimated = samples[::step] if len(samples) > target else samples

    result = {
        "station": f"{network}.{station}.{channel}",
        "start_time": start_time,
        "sample_rate": sample_rate,
        "samples": decimated,
        "raw_count": len(samples),
        "duration_s": duration,
    }
    CACHE[cache_key] = {"data": result, "ts": now_ts}
    return result


@app.get("/api/field_strengths")
def get_field_strengths():
    """
    Computed electromagnetic field strengths at Earth's surface.

    Fair Weather Field: ~130 V/m baseline, modulated by cosmic rays and Kp
    Telluric Currents: proportional to dB/dt (geomagnetic rate of change)
    Mansurov Effect: Bz-dependent atmospheric potential modulation
    Schumann Resonance: ~7.83 Hz, shifts with ionospheric conductivity
    GIC Risk: dB/dt based geomagnetically induced current risk
    """
    # Get latest data from cache
    kp_data = CACHE.get("kp", {}).get("data")
    sw_data = CACHE.get("sw_mag", {}).get("data")
    dst_data = None  # would need Dst cache
    cr_stations = CACHE.get("cosmic_rays", {}).get("data", {})

    # Current Kp
    kp = 2.0
    if kp_data:
        try:
            entries = []
            for row in kp_data:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    try:
                        entries.append(float(row[1]))
                    except (ValueError, TypeError):
                        pass
            if entries:
                kp = entries[-1]
        except Exception:
            pass

    # Current Bz
    bz = 0.0
    if sw_data:
        try:
            for row in reversed(sw_data[1:]):
                v = row[3] if len(row) > 3 else None
                if v not in (None, "", "null"):
                    bz = float(v)
                    break
        except Exception:
            pass

    # Cosmic ray deviation
    cr_dev = 0.0
    if isinstance(cr_stations, dict) and "stations" in cr_stations:
        devs = [s.get("deviation_pct", 0) for s in cr_stations["stations"].values()]
        if devs:
            cr_dev = sum(devs) / len(devs)

    # === Fair Weather Field (Ez) ===
    # Baseline ~130 V/m, suppressed by cosmic ray increase, enhanced by Forbush decrease
    # Kp disturbs it: high Kp -> more variation
    fwf_baseline = 130.0
    fwf = fwf_baseline * (1 - cr_dev / 100 * 0.3)  # cosmic ray modulation
    fwf *= (1 + kp * 0.02)  # slight Kp enhancement
    fwf = max(50, min(250, fwf))

    # === Telluric Currents (J) ===
    # Proportional to dB/dt; Kp is a rough proxy
    # Quiet: ~1-5 mA/km, Storm: 50-500 mA/km
    telluric = 2.0 * math.exp(kp * 0.4)  # exponential with Kp
    if bz < -10:
        telluric *= 1.5  # southward Bz enhances substorms

    # === Mansurov Effect (dB/dt from Kp transitions) ===
    # Rate of Kp change drives atmospheric E-field modulation
    mansurov_dbdt = kp * 3.5  # rough nT/hr from Kp
    if bz < 0:
        mansurov_dbdt *= (1 + abs(bz) / 20)

    # === Schumann Resonance ===
    # f1 ~ 7.83 Hz, shifts slightly with ionospheric conductivity
    # Solar X-rays increase conductivity -> slight frequency increase
    schumann = 7.83 + kp * 0.005 + (0.02 if bz < -5 else 0)

    # === GIC Risk ===
    # Based on dB/dt proxy and Kp
    gic_score = min(1.0, (kp - 3) / 6 + (1 if bz < -15 else 0) * 0.3)
    gic_score = max(0, gic_score)
    gic_label = "LOW" if gic_score < 0.2 else "MODERATE" if gic_score < 0.5 else "HIGH" if gic_score < 0.8 else "EXTREME"

    return {
        "fair_weather_ez": {"value": round(fwf, 1), "unit": "V/m", "baseline": 130},
        "telluric_j": {"value": round(telluric, 1), "unit": "mA/km"},
        "mansurov_dbdt": {"value": round(mansurov_dbdt, 1), "unit": "nT/hr"},
        "schumann_f1": {"value": round(schumann, 3), "unit": "Hz"},
        "gic_risk": {"score": round(gic_score, 2), "label": gic_label},
        "inputs": {"kp": kp, "bz": round(bz, 1), "cr_dev_pct": round(cr_dev, 1)},
    }


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


# ========== Solar Monitor Proxy ==========
# Proxies to the Rust solar-monitor running on SOLAR_MONITOR_URL (default localhost:3000)

SOLAR_MONITOR_URL = os.environ.get("SOLAR_MONITOR_URL", "http://localhost:8089")


async def _solar_proxy(path: str):
    """Proxy a GET request to the solar monitor."""
    url = f"{SOLAR_MONITOR_URL}/api/solar/{path}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            return resp.json()
    except Exception as e:
        return {"error": str(e), "source": url}


@app.get("/api/solar/status")
async def solar_status():
    """Stressor index + escalation + fused detector score."""
    return await _solar_proxy("status")


@app.get("/api/solar/detectors")
async def solar_detectors():
    """Per-detector raw scores + agreement count."""
    return await _solar_proxy("detectors")


@app.get("/api/solar/pathways")
async def solar_pathways():
    """All 5 coupling pathway statuses."""
    return await _solar_proxy("pathways")


@app.get("/api/solar/escalation")
async def solar_escalation():
    """Current escalation level + precursor status."""
    return await _solar_proxy("escalation")


@app.get("/api/solar/feeds")
async def solar_feeds():
    """Latest values from all data streams."""
    return await _solar_proxy("feeds")


@app.get("/api/solar/feeds/{feed_name}")
async def solar_feed(feed_name: str):
    """Specific feed ring buffer (xray, electrons, solar-wind, kp-dst)."""
    return await _solar_proxy(f"feeds/{feed_name}")


@app.get("/api/solar/health")
async def solar_health():
    """Feed freshness check."""
    return await _solar_proxy("health")


@app.get("/api/solar/state")
async def solar_state():
    """Complete solar state snapshot (DONKI + SWPC fetch)."""
    return await _solar_proxy("state")


async def _sse_proxy(path: str):
    """Proxy an SSE stream from solar monitor."""
    url = f"{SOLAR_MONITOR_URL}/api/solar/{path}"

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as resp:
                    async for line in resp.aiter_lines():
                        yield line + "\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/solar/metrics")
async def solar_metrics_sse():
    """SSE stream of metrics (per poll cycle)."""
    return await _sse_proxy("metrics")


@app.get("/api/solar/alerts")
async def solar_alerts_sse():
    """SSE stream of alert events."""
    return await _sse_proxy("alerts")


# Serve frontend if available. Prefer the Vite-built output (frontend/dist),
# fall back to raw source (only usable behind the Vite dev server).
_frontend_root = Path(__file__).parent.parent / "frontend"
_frontend_dist = _frontend_root / "dist"
frontend_dir = _frontend_dist if _frontend_dist.exists() else _frontend_root
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
