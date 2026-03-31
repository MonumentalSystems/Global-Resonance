#!/usr/bin/env python3
"""
Download and cache ALL data sources for earthquake-solar analysis.
Run this once — everything saves to data/ directory.
"""

import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
import datetime as dt
import requests
import json
import sys
import os
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
INIT_DATE = dt.datetime(2000, 1, 1)


def save_if_new(df, name, force=False):
    path = DATA_DIR / name
    if path.exists() and not force:
        print(f"  Already cached: {name} ({path.stat().st_size / 1024:.0f} KB)")
        return False
    df.to_csv(path, index=False)
    print(f"  Saved: {name} ({len(df)} rows, {path.stat().st_size / 1024:.0f} KB)")
    return True


def save_raw(content, name):
    path = DATA_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved: {name} ({len(content)} bytes)")


# ═══════════════════════════════════════════════════════════════════════
# 1. EARTHQUAKES (USGS ComCat)
# ═══════════════════════════════════════════════════════════════════════

def download_earthquakes():
    for min_mag, label in [(4.5, "m4.5"), (5.0, "m5.0"), (4.0, "m4.0_vanuatu")]:
        cache = DATA_DIR / f"earthquakes_{label}.csv"
        if cache.exists():
            print(f"  Already cached: earthquakes_{label}.csv")
            continue

        if "vanuatu" in label:
            extra = {"minlatitude": -25, "maxlatitude": -10,
                     "minlongitude": 160, "maxlongitude": 180}
            min_mag = 4.0
        else:
            extra = {}

        print(f"\n  Downloading earthquakes M>={min_mag} {'(Vanuatu)' if 'vanuatu' in label else '(global)'}...")
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        all_dfs = []
        for year in range(2000, 2027):
            try:
                params = {"format": "csv", "starttime": f"{year}-01-01",
                          "endtime": f"{year}-12-31", "minmagnitude": min_mag,
                          "orderby": "time-asc", "limit": 20000, **extra}
                resp = requests.get(url, params=params, timeout=120)
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text))
                all_dfs.append(df)
                print(f"    {year}: {len(df)}")
            except Exception as e:
                print(f"    {year}: FAILED ({e})")
            time.sleep(0.5)  # be nice to USGS

        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            df["time_parsed"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
            df["magnitude"] = df["mag"]
            df["day_number"] = ((df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
            # Save full USGS columns plus our additions
            df.to_csv(cache, index=False)
            print(f"  Saved: earthquakes_{label}.csv ({len(df)} events)")

    # Also yearly M5+ going back to 1980
    cache = DATA_DIR / "earthquakes_yearly_m5.csv"
    if not cache.exists():
        print("\n  Downloading yearly M5+ earthquake counts (1980-2026)...")
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        yearly = []
        for year in range(1980, 2027):
            try:
                resp = requests.get(url, params={
                    "format": "csv", "starttime": f"{year}-01-01",
                    "endtime": f"{year}-12-31", "minmagnitude": 5.0,
                    "orderby": "time-asc", "limit": 20000,
                }, timeout=60)
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text))
                n = len(df)
                energy = np.sum(10**(1.5 * df["mag"])) if n > 0 else 0
                yearly.append({"year": year, "n_m5": n,
                              "n_m6": len(df[df["mag"]>=6.0]),
                              "n_m7": len(df[df["mag"]>=7.0]),
                              "max_mag": df["mag"].max() if n > 0 else 0,
                              "log_energy": np.log10(energy + 1)})
                print(f"    {year}: {n} M5+")
            except Exception as e:
                print(f"    {year}: FAILED ({e})")
            time.sleep(0.5)
        save_if_new(pd.DataFrame(yearly), "earthquakes_yearly_m5.csv")


# ═══════════════════════════════════════════════════════════════════════
# 2. GEOMAGNETIC INDICES (GFZ Potsdam)
# ═══════════════════════════════════════════════════════════════════════

def download_kp():
    cache_daily = DATA_DIR / "kp_daily.csv"
    cache_3h = DATA_DIR / "kp_3hourly.csv"
    cache_raw = DATA_DIR / "kp_raw.txt"

    if cache_daily.exists() and cache_3h.exists():
        print("  Already cached: kp_daily.csv, kp_3hourly.csv")
        return

    print("\n  Downloading Kp/Ap/F10.7 from GFZ Potsdam...")
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    # Save raw file
    save_raw(resp.text, "kp_raw.txt")

    daily_records = []
    hourly_records = []

    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 26:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 1980:
                continue
            kp_vals = [float(parts[7+i]) for i in range(8)]
            ap_vals = [float(parts[15+i]) for i in range(8)]
            daily_ap = float(parts[23])
            sn = float(parts[24])
            f107_obs = float(parts[25])
            f107_adj = float(parts[26]) if len(parts) > 26 else np.nan

            daily_records.append({
                "year": y, "month": m, "day": d,
                "kp_mean": np.mean(kp_vals), "kp_max": max(kp_vals), "kp_min": min(kp_vals),
                "kp_sum": sum(kp_vals),
                "ap": daily_ap, "sn": sn if sn >= 0 else np.nan,
                "f107": f107_obs if f107_obs > 0 else np.nan,
                "f107_adj": f107_adj if f107_adj > 0 else np.nan,
            })

            for slot in range(8):
                hourly_records.append({
                    "year": y, "month": m, "day": d, "hour": slot * 3,
                    "kp": kp_vals[slot], "ap": ap_vals[slot],
                })
        except (ValueError, IndexError):
            continue

    daily_df = pd.DataFrame(daily_records)
    daily_df["day_number"] = ((pd.to_datetime(daily_df[["year","month","day"]]) -
                                pd.Timestamp(INIT_DATE)).dt.days).values
    save_if_new(daily_df, "kp_daily.csv", force=True)

    hourly_df = pd.DataFrame(hourly_records)
    hourly_df["datetime"] = pd.to_datetime(hourly_df[["year","month","day","hour"]])
    hourly_df["day_number"] = ((hourly_df["datetime"] - pd.Timestamp(INIT_DATE)).dt.days).values
    hourly_df["dkp_dt"] = hourly_df["kp"].diff().fillna(0)
    save_if_new(hourly_df, "kp_3hourly.csv", force=True)


# ═══════════════════════════════════════════════════════════════════════
# 3. SUNSPOT DATA (SILSO)
# ═══════════════════════════════════════════════════════════════════════

def download_sunspots():
    # Daily
    cache = DATA_DIR / "sunspots_daily.csv"
    if not cache.exists():
        print("\n  Downloading daily sunspot numbers from SILSO...")
        url = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
        resp = requests.get(url, timeout=60)
        df = pd.read_csv(StringIO(resp.text), sep=";", header=None,
                         names=["year","month","day","dec_year","ssn","std","nobs","definitive"],
                         skipinitialspace=True)
        df = df[df["year"] >= 1980].copy()
        df.loc[df["ssn"] < 0, "ssn"] = np.nan
        df["ssn"] = df["ssn"].interpolate()
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        df["day"] = df["day"].astype(int)
        df["day_number"] = ((pd.to_datetime(df[["year","month","day"]]) -
                              pd.Timestamp(INIT_DATE)).dt.days).values
        save_if_new(df, "sunspots_daily.csv", force=True)
    else:
        print("  Already cached: sunspots_daily.csv")

    # Monthly
    cache = DATA_DIR / "sunspots_monthly.csv"
    if not cache.exists():
        print("  Downloading monthly sunspot numbers...")
        url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
        resp = requests.get(url, timeout=60)
        df = pd.read_csv(StringIO(resp.text), sep=";", header=None,
                         names=["year","month","dec_year","ssn","std","nobs","definitive"],
                         skipinitialspace=True)
        df = df[df["year"] >= 1980].copy()
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        save_if_new(df, "sunspots_monthly.csv", force=True)
    else:
        print("  Already cached: sunspots_monthly.csv")


# ═══════════════════════════════════════════════════════════════════════
# 4. SOLAR FLARES (DONKI)
# ═══════════════════════════════════════════════════════════════════════

def download_flares():
    cache = DATA_DIR / "solar_flares.csv"
    if cache.exists():
        print("  Already cached: solar_flares.csv")
        return

    print("\n  Downloading solar flare catalog from DONKI (2010-2026)...")
    base_url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR"
    all_flares = []
    for year in range(2010, 2027):
        for half in [0, 1]:
            start = f"{year}-{'01' if half==0 else '07'}-01"
            end = f"{year}-{'06-30' if half==0 else '12-31'}"
            try:
                resp = requests.get(base_url, params={"startDate": start, "endDate": end}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                for flare in data:
                    all_flares.append({
                        "beginTime": flare.get("beginTime"),
                        "peakTime": flare.get("peakTime"),
                        "endTime": flare.get("endTime"),
                        "classType": flare.get("classType", ""),
                        "sourceLocation": flare.get("sourceLocation", ""),
                        "activeRegionNum": flare.get("activeRegionNum"),
                    })
                print(f"    {start}: {len(data)} flares")
            except Exception as e:
                print(f"    {start}: FAILED ({e})")
            time.sleep(0.3)

    df = pd.DataFrame(all_flares)
    for col in ["beginTime", "peakTime", "endTime"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)

    def parse_class(c):
        if not isinstance(c, str) or len(c) < 2: return np.nan
        letter = c[0]
        try: num = float(c[1:])
        except: return np.nan
        mult = {"X": 1.0, "M": 0.1, "C": 0.01, "B": 0.001, "A": 0.0001}
        return mult.get(letter, 0) * num

    df["class_numeric"] = df["classType"].apply(parse_class)
    df["day_number"] = ((df["peakTime"] - pd.Timestamp(INIT_DATE)).dt.days).values
    save_if_new(df, "solar_flares.csv", force=True)


# ═══════════════════════════════════════════════════════════════════════
# 5. CME CATALOG (DONKI)
# ═══════════════════════════════════════════════════════════════════════

def download_cmes():
    cache = DATA_DIR / "cmes.csv"
    if cache.exists():
        print("  Already cached: cmes.csv")
        return

    print("\n  Downloading CME catalog from DONKI (2010-2026)...")
    base_url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"
    all_cmes = []
    for year in range(2010, 2027):
        for half in [0, 1]:
            start = f"{year}-{'01' if half==0 else '07'}-01"
            end = f"{year}-{'06-30' if half==0 else '12-31'}"
            try:
                resp = requests.get(base_url, params={"startDate": start, "endDate": end}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                for cme in data:
                    analyses = cme.get("cmeAnalyses", [])
                    best = None
                    for a in analyses:
                        if a.get("speed") and a.get("speed") > 0:
                            best = a; break
                    if best is None and analyses:
                        best = analyses[0]

                    src = cme.get("sourceLocation", "")
                    src_lat, src_lon = np.nan, np.nan
                    if src and len(src) >= 4:
                        try:
                            lat_sign = 1 if src[0] == 'N' else -1
                            ew_idx = None
                            for i, c in enumerate(src[1:], 1):
                                if c in ('E', 'W'): ew_idx = i; break
                            if ew_idx:
                                src_lat = lat_sign * float(src[1:ew_idx])
                                src_lon = (1 if src[ew_idx]=='E' else -1) * float(src[ew_idx+1:])
                        except: pass

                    all_cmes.append({
                        "datetime": cme.get("startTime"),
                        "speed": best.get("speed") if best else None,
                        "halfAngle": best.get("halfAngle") if best else None,
                        "latitude": best.get("latitude") if best else None,
                        "longitude": best.get("longitude") if best else None,
                        "sourceLocation": src,
                        "src_lat": src_lat, "src_lon": src_lon,
                        "note": cme.get("note", "")[:200],  # truncate long notes
                    })
                print(f"    {start}: {len(data)} CMEs")
            except Exception as e:
                print(f"    {start}: FAILED ({e})")
            time.sleep(0.3)

    df = pd.DataFrame(all_cmes)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors='coerce').dt.tz_localize(None)
    df["day_number"] = ((df["datetime"] - pd.Timestamp(INIT_DATE)).dt.days).values
    df["obliquity"] = np.sqrt(df["src_lat"].fillna(0)**2 + df["src_lon"].fillna(0)**2)
    save_if_new(df, "cmes.csv", force=True)


# ═══════════════════════════════════════════════════════════════════════
# 6. DSCOVR / ACE SOLAR WIND (SWPC JSON — last 7 days + archive info)
# ═══════════════════════════════════════════════════════════════════════

def download_solar_wind():
    # SWPC provides 7-day real-time data in JSON
    # For longer archives, NCEI DSCOVR portal needed (manual download)
    cache = DATA_DIR / "solar_wind_recent.json"
    if not cache.exists():
        print("\n  Downloading recent solar wind from SWPC (7-day JSON)...")
        urls = {
            "mag": "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
            "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
        }
        wind_data = {}
        for key, url in urls.items():
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                wind_data[key] = resp.json()
                print(f"    {key}: {len(wind_data[key])} records")
            except Exception as e:
                print(f"    {key}: FAILED ({e})")

        with open(cache, "w") as f:
            json.dump(wind_data, f)
        print(f"  Saved: solar_wind_recent.json")

        # Parse into CSV
        if "mag" in wind_data and len(wind_data["mag"]) > 1:
            headers = wind_data["mag"][0]
            rows = wind_data["mag"][1:]
            mag_df = pd.DataFrame(rows, columns=headers)
            save_if_new(mag_df, "solar_wind_mag_7day.csv", force=True)

        if "plasma" in wind_data and len(wind_data["plasma"]) > 1:
            headers = wind_data["plasma"][0]
            rows = wind_data["plasma"][1:]
            plasma_df = pd.DataFrame(rows, columns=headers)
            save_if_new(plasma_df, "solar_wind_plasma_7day.csv", force=True)
    else:
        print("  Already cached: solar_wind_recent.json")

    # DSCOVR archive pointer
    print("  Note: Full DSCOVR archive at https://www.ngdc.noaa.gov/dscovr/portal/index.html")
    print("        ACE archive at https://izw1.caltech.edu/ACE/ASC/level2/lvl2DATA_MAG-SWEPAM.html")


# ═══════════════════════════════════════════════════════════════════════
# 7. GEOMAGNETIC STORM SUDDEN COMMENCEMENTS (GFZ)
# ═══════════════════════════════════════════════════════════════════════

def download_ssc():
    cache = DATA_DIR / "ssc_list.csv"
    if cache.exists():
        print("  Already cached: ssc_list.csv")
        return

    print("\n  Downloading Sudden Storm Commencements from GFZ...")
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    # We already have this data — extract SSCs from the Kp jumps
    # SSC = sudden increase in H component, proxy: dKp >= 3 in one 3-hour step
    kp_3h = pd.read_csv(DATA_DIR / "kp_3hourly.csv")
    if "dkp_dt" not in kp_3h.columns:
        kp_3h["dkp_dt"] = kp_3h["kp"].diff().fillna(0)

    sscs = kp_3h[kp_3h["dkp_dt"] >= 3.0].copy()
    # Deduplicate: keep first per 3-day window
    if len(sscs) > 0:
        days = sscs["day_number"].values
        filtered = [0]
        for i in range(1, len(days)):
            if days[i] - days[filtered[-1]] >= 3:
                filtered.append(i)
        sscs = sscs.iloc[filtered]

    save_if_new(sscs, "ssc_list.csv", force=True)


# ═══════════════════════════════════════════════════════════════════════
# 8. Dst INDEX (Kyoto WDC — monthly pages)
# ═══════════════════════════════════════════════════════════════════════

def download_dst():
    cache = DATA_DIR / "dst_hourly.csv"
    if cache.exists():
        print("  Already cached: dst_hourly.csv")
        return

    print("\n  Downloading Dst index from Kyoto WDC (provisional, 2000-2026)...")
    print("  Note: this scrapes monthly pages — may take a few minutes")

    all_records = []
    base_url = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional"

    for year in range(2000, 2027):
        for month in range(1, 13):
            ym = f"{year}{month:02d}"
            url = f"{base_url}/{ym}/index.html"
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue
                # Parse the fixed-format Dst data
                # The format has day numbers and 24 hourly values per line
                lines = resp.text.split('\n')
                for line in lines:
                    # Dst lines have format: DD followed by 24 values
                    # Look for lines that start with day numbers
                    stripped = line.strip()
                    if not stripped or len(stripped) < 50:
                        continue
                    # Try to parse as Dst data line
                    try:
                        # Check if first 2 chars are a day number
                        day = int(stripped[:2])
                        if day < 1 or day > 31:
                            continue
                        # Extract 24 hourly values (each 4 chars wide after the day)
                        vals = []
                        for h in range(24):
                            start = 3 + h * 4
                            end = start + 4
                            if end <= len(stripped):
                                val_str = stripped[start:end].strip()
                                if val_str and val_str != '9999':
                                    vals.append(int(val_str))
                                else:
                                    vals.append(np.nan)
                            else:
                                vals.append(np.nan)

                        if len(vals) == 24 and not all(np.isnan(v) for v in vals):
                            for h, v in enumerate(vals):
                                all_records.append({
                                    "year": year, "month": month, "day": day,
                                    "hour": h, "dst": v
                                })
                    except (ValueError, IndexError):
                        continue

                if month == 1:
                    print(f"    {year}...", end=" ", flush=True)
            except Exception as e:
                continue
            time.sleep(0.2)

    print()
    if all_records:
        df = pd.DataFrame(all_records)
        df["datetime"] = pd.to_datetime(df[["year","month","day","hour"]], errors='coerce')
        df = df.dropna(subset=["datetime"])
        df["day_number"] = ((df["datetime"] - pd.Timestamp(INIT_DATE)).dt.days).values
        save_if_new(df, "dst_hourly.csv", force=True)
    else:
        print("  WARNING: No Dst data parsed. Format may have changed.")
        print("  Manual download: https://wdc.kugi.kyoto-u.ac.jp/dstdir/")


# ═══════════════════════════════════════════════════════════════════════
# 9. INTERPLANETARY MAGNETIC FIELD (OMNI dataset — combined ACE/DSCOVR)
# ═══════════════════════════════════════════════════════════════════════

def download_omni():
    """OMNI hourly data — merged solar wind from multiple spacecraft."""
    cache = DATA_DIR / "omni_hourly.csv"
    if cache.exists():
        print("  Already cached: omni_hourly.csv")
        return

    print("\n  Downloading OMNI hourly solar wind data (NASA CDAWeb)...")
    # OMNI2 hourly data is available as annual text files
    # https://omniweb.gsfc.nasa.gov/form/dx1.html
    base_url = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"

    all_records = []
    for year in range(2000, 2027):
        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"    {year}: not available")
                continue

            for line in resp.text.splitlines():
                parts = line.split()
                if len(parts) < 55:
                    continue
                try:
                    yr = int(parts[0])
                    doy = int(parts[1])
                    hr = int(parts[2])

                    # Key columns (1-indexed in OMNI format):
                    # 17: IMF magnitude (nT)
                    # 18: Bx GSE (nT)
                    # 19: By GSE (nT)
                    # 20: Bz GSE (nT)  <-- THIS IS THE KEY ONE
                    # 24: Solar wind speed (km/s)
                    # 25: Proton density (n/cc)
                    # 38: Dst index (nT)
                    # 39: AE index (nT)
                    # 40: Ap index

                    bz = float(parts[14])   # Bz GSE (signed, nT)
                    bz_gsm = float(parts[16])  # Bz GSM (better for reconnection)
                    by = float(parts[13])   # By GSE (signed, nT)
                    b_mag = float(parts[8])  # |B| (nT)
                    v_sw = float(parts[24])  # Flow speed km/s
                    n_p = float(parts[23])   # Proton density n/cc
                    dst = float(parts[40]) if len(parts) > 40 else 99999  # Dst nT
                    ae = float(parts[41]) if len(parts) > 41 else 99999   # AE nT

                    # OMNI uses 9999 or 999.9 as fill values
                    def clean(v, fill=999.9):
                        return v if abs(v) < fill else np.nan

                    date = dt.datetime(yr, 1, 1) + dt.timedelta(days=doy-1, hours=hr)

                    all_records.append({
                        "year": yr, "doy": doy, "hour": hr,
                        "datetime": date,
                        "bz_gse": bz if abs(bz) < 999 else np.nan,
                        "bz_gsm": bz_gsm if abs(bz_gsm) < 999 else np.nan,
                        "by_gse": by if abs(by) < 999 else np.nan,
                        "b_mag": b_mag if b_mag < 999 else np.nan,
                        "v_sw": v_sw if v_sw < 9999 else np.nan,
                        "n_proton": n_p if n_p < 999 else np.nan,
                        "dst": dst if abs(dst) < 99999 else np.nan,
                        "ae": ae if ae < 9999 else np.nan,
                    })
                except (ValueError, IndexError):
                    continue

            print(f"    {year}: OK")
        except Exception as e:
            print(f"    {year}: FAILED ({e})")
        time.sleep(0.3)

    if all_records:
        df = pd.DataFrame(all_records)
        df["day_number"] = ((pd.to_datetime(df["datetime"]) - pd.Timestamp(INIT_DATE)).dt.days).values
        save_if_new(df, "omni_hourly.csv", force=True)
        print(f"  OMNI dataset: {len(df)} hourly records with Bz, V_sw, Dst, AE")
    else:
        print("  WARNING: No OMNI data downloaded")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("DOWNLOADING ALL DATA SOURCES")
    print(f"Saving to: {DATA_DIR}")
    print("=" * 70)

    print("\n[1/9] EARTHQUAKES (USGS ComCat)")
    download_earthquakes()

    print("\n[2/9] GEOMAGNETIC INDICES (GFZ)")
    download_kp()

    print("\n[3/9] SUNSPOT DATA (SILSO)")
    download_sunspots()

    print("\n[4/9] SOLAR FLARES (DONKI)")
    download_flares()

    print("\n[5/9] CME CATALOG (DONKI)")
    download_cmes()

    print("\n[6/9] SOLAR WIND (SWPC real-time)")
    download_solar_wind()

    print("\n[7/9] SUDDEN STORM COMMENCEMENTS")
    download_ssc()

    print("\n[8/9] Dst INDEX (Kyoto WDC)")
    download_dst()

    print("\n[9/9] OMNI HOURLY (NASA — Bz, V_sw, Dst, AE)")
    download_omni()

    # Summary
    print("\n" + "=" * 70)
    print("DATA INVENTORY")
    print("=" * 70)
    total_size = 0
    for f in sorted(DATA_DIR.glob("*")):
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name:40s}  {size/1024:>8.0f} KB")
    print(f"  {'TOTAL':40s}  {total_size/1024:>8.0f} KB ({total_size/1024/1024:.1f} MB)")

    print(f"\nAll data cached in: {DATA_DIR}")
    print("Run analyses with cached data — no re-downloading needed.")


if __name__ == "__main__":
    main()
