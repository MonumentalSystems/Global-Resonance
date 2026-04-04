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

            # Paper XXV 10-zone spatial response (earth_jelly_ball.py)
            if ang_dist < 15: zone = "eye"              # 0.85x suppression
            elif ang_dist < 30: zone = "inner"           # 0.92x compression
            elif ang_dist < 60: zone = "transition"      # 0.98x near-neutral
            elif ang_dist < 75: zone = "wavefront"       # 1.36x PEAK
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
            # Paper XXV spatial response pattern (earth_jelly_ball.py)
            # Each ring is drawn at its outer radius
            {"name": "eye",            "radius_deg": 15,  "color": "#4444ff", "ratio": 0.85, "effect": "suppression"},
            {"name": "inner",          "radius_deg": 30,  "color": "#6666cc", "ratio": 0.92, "effect": "compression"},
            {"name": "transition",     "radius_deg": 60,  "color": "#44aa44", "ratio": 0.98, "effect": "near-neutral"},
            {"name": "wavefront",      "radius_deg": 75,  "color": "#ff4444", "ratio": 1.36, "effect": "PEAK enhancement"},
            {"name": "wavefront-tail", "radius_deg": 100, "color": "#ff8844", "ratio": 1.09, "effect": "enhancement"},
            {"name": "neutral",        "radius_deg": 120, "color": "#888844", "ratio": 0.95, "effect": "neutral"},
            {"name": "far-suppress",   "radius_deg": 135, "color": "#446688", "ratio": 0.82, "effect": "suppression"},
            {"name": "far-neutral",    "radius_deg": 155, "color": "#666666", "ratio": 0.90, "effect": "far neutral"},
            {"name": "pre-antipodal",  "radius_deg": 165, "color": "#886688", "ratio": 1.00, "effect": "neutral"},
            {"name": "antipodal",      "radius_deg": 180, "color": "#cc88cc", "ratio": 1.16, "effect": "enhancement"},
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
        "https://api.nasa.gov/DONKI/CME?startDate=2026-03-28&endDate=2026-03-31&api_key=DEMO_KEY",
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

    # === Zone risk modulation ===
    # When J > J_c: suppression near subsolar, enhancement at wavefront
    # When J < J_c: normal background rates
    if above_critical or near_critical:
        zone_risk = {
            "eye":            {"factor": 0.85, "risk": "SUPPRESSED"},
            "inner":          {"factor": 0.92, "risk": "suppressed"},
            "transition":     {"factor": 0.98, "risk": "near-normal"},
            "wavefront":      {"factor": 1.36, "risk": "ENHANCED"},
            "wavefront-tail": {"factor": 1.09, "risk": "enhanced"},
            "neutral":        {"factor": 0.95, "risk": "normal"},
            "far-suppress":   {"factor": 0.82, "risk": "SUPPRESSED"},
            "far-neutral":    {"factor": 0.90, "risk": "slightly suppressed"},
            "pre-antipodal":  {"factor": 1.00, "risk": "normal"},
            "antipodal":      {"factor": 1.16, "risk": "enhanced"},
        }
    else:
        zone_risk = {z: {"factor": 1.00, "risk": "baseline"} for z in
            ["eye", "inner", "transition", "wavefront", "wavefront-tail",
             "neutral", "far-suppress", "far-neutral", "pre-antipodal", "antipodal"]}

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
        """Assign a name/color to a segment based on its centroid."""
        lons = [p[0] for p in seg]
        lats = [p[1] for p in seg]
        clon = sum(lons) / len(lons)
        clat = sum(lats) / len(lats)

        best = ("Unknown Boundary", "#445566", "unknown")
        best_dist = 1e9
        for name, color, lo0, la0, lo1, la1, btype in REGIONS:
            # Handle wrap-around for Pacific
            if lo0 > lo1:
                in_lon = clon > lo0 or clon < lo1
            else:
                in_lon = lo0 <= clon <= lo1
            in_lat = la0 <= clat <= la1
            if in_lon and in_lat:
                cx = (lo0 + lo1) / 2 if lo0 < lo1 else ((lo0 + lo1 + 360) / 2) % 360 - 180
                cy = (la0 + la1) / 2
                d = math.sqrt((clon - cx) ** 2 + (clat - cy) ** 2)
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
    start = now - timedelta(seconds=duration)
    url = (
        f"https://service.iris.edu/irisws/timeseries/1/query?"
        f"net={network}&sta={station}&loc=00&cha={channel}"
        f"&starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&endtime={now.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&output=ascii&nodata=404"
    )
    cache_key = f"seismo_{station}_{channel}"
    now_ts = time.time()
    if cache_key in CACHE and now_ts - CACHE[cache_key]["ts"] < 60:
        return CACHE[cache_key]["data"]

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return {"error": "No data available", "station": station}
            text = resp.text
    except Exception as e:
        return {"error": str(e), "station": station}

    # Parse IRIS ASCII: header line starting with TIMESERIES, then data lines
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
                if p.startswith("20") and "T" in p:
                    start_time = p
        else:
            try:
                vals = line.strip().split()
                for v in vals:
                    samples.append(int(v))
            except Exception:
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


# Serve frontend if available
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
