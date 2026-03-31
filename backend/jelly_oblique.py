#!/usr/bin/env python3
"""
Jelly Ball v2: CME Obliquity Analysis
=======================================
Uses NASA DONKI CME catalog to get:
  - Source location on solar disk (lat, lon)
  - Half-angle (angular width)
  - Speed
  - Earth-directed flag

The non-planar force is [F, nabla F] ~ sin(alpha) where alpha
is the angle between the CME velocity vector and the
Earth-Sun line. Head-on CMEs have alpha ~ 0 (weak commutator),
oblique CMEs have alpha > 0 (strong commutator).

Prediction: oblique CMEs should show stronger seismic enhancement
at 60-90 degrees from subsolar point than head-on CMEs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO
from pathlib import Path
import datetime as dt
import requests
import json
import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
INIT_DATE = dt.datetime(2000, 1, 1)


def download_donki_cmes(start_year=2010, end_year=2026):
    """
    Download CMEs from NASA DONKI API.
    DONKI has good coverage from ~2010 onward.
    """
    print(f"Downloading CME catalog from DONKI ({start_year}-{end_year})...")
    base_url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"

    all_cmes = []
    for year in range(start_year, end_year + 1):
        for half in [0, 1]:
            start = f"{year}-{'01' if half == 0 else '07'}-01"
            end = f"{year}-{'06-30' if half == 0 else '12-31'}"
            try:
                resp = requests.get(base_url, params={
                    "startDate": start, "endDate": end
                }, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                all_cmes.extend(data)
                print(f"  {start} to {end}: {len(data)} CMEs")
            except Exception as e:
                print(f"  {start} to {end}: failed ({e})")

    print(f"  Total raw CMEs: {len(all_cmes)}")
    return all_cmes


def parse_cmes(raw_cmes):
    """Extract structured data from DONKI CME records."""
    records = []

    for cme in raw_cmes:
        start_time = cme.get("startTime")
        source_loc = cme.get("sourceLocation", "")
        ar_num = cme.get("activeRegionNum")

        # Parse source location (e.g., "N05E59" -> lat=5, lon=59)
        src_lat, src_lon = None, None
        if source_loc and len(source_loc) >= 4:
            try:
                lat_sign = 1 if source_loc[0] == 'N' else -1
                # Find where the E/W character is
                ew_idx = None
                for i, c in enumerate(source_loc[1:], 1):
                    if c in ('E', 'W'):
                        ew_idx = i
                        break
                if ew_idx:
                    src_lat = lat_sign * float(source_loc[1:ew_idx])
                    lon_sign = 1 if source_loc[ew_idx] == 'E' else -1
                    src_lon = lon_sign * float(source_loc[ew_idx+1:])
            except (ValueError, IndexError):
                pass

        # Get best analysis (prefer "S" type with speed)
        analyses = cme.get("cmeAnalyses", [])
        best = None
        for a in analyses:
            if a.get("speed") and a.get("speed") > 0:
                if best is None or (a.get("isMostAccurate", False)):
                    best = a

        if best is None and analyses:
            best = analyses[0]

        speed = best.get("speed") if best else None
        half_angle = best.get("halfAngle") if best else None
        cme_lat = best.get("latitude") if best else None
        cme_lon = best.get("longitude") if best else None

        # Use analysis lat/lon if source location parsing failed
        if src_lat is None and cme_lat is not None:
            src_lat = cme_lat
        if src_lon is None and cme_lon is not None:
            src_lon = cme_lon

        # Check if Earth-directed
        is_earth = False
        if best and best.get("enlilList"):
            for enlil in best["enlilList"]:
                if enlil.get("isEarthGB") or enlil.get("isEarthMinorImpact"):
                    is_earth = True
                    break

        if start_time and src_lon is not None:
            dt_parsed = pd.to_datetime(start_time, utc=True).tz_localize(None)
            # Obliquity: angle from disk center
            # CME at (lat, lon) on solar disk -> obliquity ~ sqrt(lat^2 + lon^2)
            obliquity = np.sqrt((src_lat or 0)**2 + (src_lon or 0)**2)

            records.append({
                "datetime": dt_parsed,
                "src_lat": src_lat,
                "src_lon": src_lon,
                "obliquity": obliquity,
                "speed": speed,
                "half_angle": half_angle,
                "is_earth_directed": is_earth,
                "day_number": (dt_parsed - pd.Timestamp(INIT_DATE)).days,
                "source_location": source_loc,
            })

    df = pd.DataFrame(records)
    print(f"  Parsed: {len(df)} CMEs with source locations")
    print(f"  Earth-directed: {df['is_earth_directed'].sum()}")
    print(f"  With speed: {df['speed'].notna().sum()}")
    return df


def download_global_earthquakes(min_mag=5.0, start_year=2010):
    """Download global earthquake catalog."""
    print(f"Downloading global earthquakes (M>={min_mag}, {start_year}+)...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_dfs = []
    for year in range(start_year, 2027):
        try:
            resp = requests.get(url, params={
                "format": "csv", "starttime": f"{year}-01-01",
                "endtime": f"{year}-12-31", "minmagnitude": min_mag,
                "orderby": "time-asc", "limit": 20000,
            }, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            all_dfs.append(df)
            print(f"  {year}: {len(df)} events")
        except Exception as e:
            print(f"  {year}: failed ({e})")

    df = pd.concat(all_dfs, ignore_index=True)
    df["time_parsed"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["magnitude"] = df["mag"]
    df["day_number"] = ((df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    print(f"  Total: {len(df)} earthquakes")
    return df


def subsolar_point(dt_utc):
    """Approximate subsolar point."""
    doy = dt_utc.timetuple().tm_yday
    decl = -23.44 * np.cos(np.radians(360 / 365 * (doy + 10)))
    hour = dt_utc.hour + dt_utc.minute / 60.0
    lon = 180.0 - 15.0 * hour
    if lon < -180: lon += 360
    if lon > 180: lon -= 360
    return decl, lon


def angular_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def obliquity_analysis(cme_df, eq_df):
    """
    Split CMEs into head-on vs oblique and compare seismic response.

    Head-on: source near disk center (obliquity < 30 deg)
      -> sin(alpha) ~ 0, weak commutator, mostly compressive
    Oblique: source near limb (obliquity > 45 deg)
      -> sin(alpha) > 0.7, strong commutator, lateral wave
    """
    print("\n=== CME Obliquity vs Seismic Response ===")

    # Only use CMEs with speed data (real events, not data gaps)
    cme_valid = cme_df[cme_df["speed"].notna() & (cme_df["speed"] > 200)].copy()
    print(f"CMEs with speed > 200 km/s: {len(cme_valid)}")

    # Split by obliquity
    head_on = cme_valid[cme_valid["obliquity"] < 30]
    oblique = cme_valid[cme_valid["obliquity"] >= 45]
    print(f"Head-on (obliquity < 30 deg): {len(head_on)}")
    print(f"Oblique (obliquity >= 45 deg): {len(oblique)}")

    # Angular distance bins
    bins = np.arange(0, 181, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    solid_angles = 2 * np.pi * np.abs(
        np.cos(np.radians(bins[:-1])) - np.cos(np.radians(bins[1:]))
    )

    def compute_profile(cme_subset, eq_df, day_offset_range=(1, 3)):
        """Compute seismicity profile for a set of CMEs."""
        all_counts = np.zeros((len(cme_subset), len(bins) - 1))
        bg_counts = np.zeros((len(cme_subset), len(bins) - 1))

        for idx, (_, cme) in enumerate(cme_subset.iterrows()):
            cme_day = cme["day_number"]
            ss_lat, ss_lon = subsolar_point(cme["datetime"])

            # Offset the subsolar point by CME source location
            # CME from E30 means the impact is shifted 30 deg east
            # This is approximate — the real geometry involves the Parker spiral
            impact_lat = ss_lat + (cme["src_lat"] or 0) * 0.3  # partial coupling
            impact_lon = ss_lon + (cme["src_lon"] or 0) * 0.3

            # Wave arrival window
            eq_wave = eq_df[(eq_df["day_number"] >= cme_day + day_offset_range[0]) &
                            (eq_df["day_number"] <= cme_day + day_offset_range[1])]
            if len(eq_wave) > 0:
                dists = angular_distance(impact_lat, impact_lon,
                                         eq_wave["latitude"].values,
                                         eq_wave["longitude"].values)
                counts, _ = np.histogram(dists, bins=bins)
                n_days = day_offset_range[1] - day_offset_range[0] + 1
                all_counts[idx] = counts / n_days

            # Background
            eq_bg = eq_df[(eq_df["day_number"] >= cme_day - 10) &
                          (eq_df["day_number"] <= cme_day - 4)]
            if len(eq_bg) > 0:
                dists = angular_distance(impact_lat, impact_lon,
                                         eq_bg["latitude"].values,
                                         eq_bg["longitude"].values)
                counts, _ = np.histogram(dists, bins=bins)
                bg_counts[idx] = counts / 7.0

        mean_wave = np.mean(all_counts, axis=0) / solid_angles
        mean_bg = np.mean(bg_counts, axis=0) / solid_angles
        ratio = np.where(mean_bg > 0, mean_wave / mean_bg, 1.0)
        return mean_wave, mean_bg, ratio

    print("\nComputing head-on CME profile...")
    wave_ho, bg_ho, ratio_ho = compute_profile(head_on, eq_df)
    print("Computing oblique CME profile...")
    wave_ob, bg_ob, ratio_ob = compute_profile(oblique, eq_df)

    # Also split by speed
    fast = cme_valid[cme_valid["speed"] >= 800]
    slow = cme_valid[(cme_valid["speed"] >= 200) & (cme_valid["speed"] < 500)]
    print(f"\nFast CMEs (>= 800 km/s): {len(fast)}")
    print(f"Slow CMEs (200-500 km/s): {len(slow)}")

    print("Computing fast CME profile...")
    wave_fast, bg_fast, ratio_fast = compute_profile(fast, eq_df)
    print("Computing slow CME profile...")
    wave_slow, bg_slow, ratio_slow = compute_profile(slow, eq_df)

    # Print results
    print(f"\n{'Angle':>6s}  {'Head-on':>8s}  {'Oblique':>8s}  {'Fast':>8s}  {'Slow':>8s}")
    for i, c in enumerate(bin_centers):
        print(f"  {c:>4.0f}   {ratio_ho[i]:>7.2f}x  {ratio_ob[i]:>7.2f}x  "
              f"{ratio_fast[i]:>7.2f}x  {ratio_slow[i]:>7.2f}x")

    # Zone summaries
    print("\nZone summary (wave arrival day +1 to +3, ratio to background):")
    for label, ratio in [("Head-on", ratio_ho), ("Oblique", ratio_ob),
                          ("Fast", ratio_fast), ("Slow", ratio_slow)]:
        near = np.mean(ratio[:3])    # 0-60
        mid = np.mean(ratio[3:6])    # 60-120
        far = np.mean(ratio[6:])     # 120-180
        print(f"  {label:>10s}:  Near(0-60)={near:.3f}  Mid(60-120)={mid:.3f}  Far(120-180)={far:.3f}")

    # ─── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Obliquity comparison
    ax = axes[0]
    ax.plot(bin_centers, ratio_ho, 'o-', color="#fc8d62", linewidth=2,
            label=f"Head-on (<30 deg, N={len(head_on)})")
    ax.plot(bin_centers, ratio_ob, 's-', color="#8da0cb", linewidth=2,
            label=f"Oblique (>45 deg, N={len(oblique)})")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(bin_centers, 0.8, 1.2, alpha=0.1, color="gray")
    ax.set_ylabel("Seismicity ratio (wave / background)")
    ax.set_xlabel("Angular distance from impact point (degrees)")
    ax.set_title("CME Obliquity: Head-On vs Oblique Impact\n"
                 "Prediction: oblique should show stronger wave front at 60-90 deg")
    ax.legend()
    ax.set_ylim(0.4, 2.0)

    # Speed comparison
    ax = axes[1]
    ax.plot(bin_centers, ratio_fast, 'o-', color="#e78ac3", linewidth=2,
            label=f"Fast (>= 800 km/s, N={len(fast)})")
    ax.plot(bin_centers, ratio_slow, 's-', color="#66c2a5", linewidth=2,
            label=f"Slow (200-500 km/s, N={len(slow)})")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(bin_centers, 0.8, 1.2, alpha=0.1, color="gray")
    ax.set_ylabel("Seismicity ratio (wave / background)")
    ax.set_xlabel("Angular distance from impact point (degrees)")
    ax.set_title("CME Speed: Fast vs Slow Impact\n"
                 "Prediction: fast CMEs = stronger impulse = larger [F, nabla F]")
    ax.legend()
    ax.set_ylim(0.4, 2.0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "jelly_oblique.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'jelly_oblique.png'}")

    return ratio_ho, ratio_ob, ratio_fast, ratio_slow


def main():
    print("=" * 70)
    print("JELLY BALL v2: CME Obliquity and Non-Planar Force")
    print("[F, nabla F] ~ sin(alpha): oblique hits should trigger more")
    print("=" * 70)

    raw_cmes = download_donki_cmes(start_year=2010, end_year=2026)
    cme_df = parse_cmes(raw_cmes)
    eq_df = download_global_earthquakes(min_mag=5.0, start_year=2010)

    ratio_ho, ratio_ob, ratio_fast, ratio_slow = obliquity_analysis(cme_df, eq_df)

    print("\n" + "=" * 70)
    print("FRAMEWORK PREDICTIONS vs DATA")
    print("=" * 70)

    # Check key predictions
    mid_ho = np.mean(ratio_ho[3:6])
    mid_ob = np.mean(ratio_ob[3:6])
    near_ho = np.mean(ratio_ho[:3])
    near_ob = np.mean(ratio_ob[:3])

    print(f"""
Prediction 1: Oblique CMEs trigger more at 60-120 deg than head-on
  Head-on mid-range ratio: {mid_ho:.3f}
  Oblique mid-range ratio: {mid_ob:.3f}
  Result: {'CONFIRMED' if mid_ob > mid_ho else 'NOT CONFIRMED'}

Prediction 2: Head-on CMEs suppress more at 0-60 deg (eye of storm)
  Head-on near ratio: {near_ho:.3f}
  Oblique near ratio: {near_ob:.3f}
  Result: {'CONFIRMED' if near_ho < near_ob else 'NOT CONFIRMED'}

Prediction 3: The commutator [F, nabla F] scales as sin(alpha)
  If confirmed, this is direct evidence that the seismic coupling
  goes through the non-planar component of the geomagnetic field
  perturbation, not the scalar (compressive) component.
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
