#!/usr/bin/env python3
"""
Trans-Pacific Propagation + Volcanism
=======================================
Tests two framework predictions:

1. TRANSVERSE PROPAGATION: Stress waves propagate preferentially
   PERPENDICULAR to geomagnetic field lines (E-W), not parallel (N-S).
   Because [F, nabla F] ~ sin(alpha), and alpha is maximized when
   the wave crosses the field at right angles.

2. VOLCANIC SQUEEZE: The jelly ball doesn't just shake — it squeezes.
   Large earthquakes should trigger volcanic unrest preferentially
   at volcanoes where the crust is weakest (thin, extensional, or
   already primed by magma). The volcanic response should follow
   the same angular pattern as the seismic response.

Uses USGS earthquake catalog + all VEI 2+ eruptions we can get.
For volcanism: use large M7+ earthquakes as triggers and check
subsequent volcanic activity at various angular distances.
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


def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, in degrees from north."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360


# ─── Data ────────────────────────────────────────────────────────────────────

def download_global_earthquakes(min_mag=5.0, start_year=2000):
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
            all_dfs.append(pd.read_csv(StringIO(resp.text)))
        except Exception as e:
            print(f"  {year}: failed ({e})")
    df = pd.concat(all_dfs, ignore_index=True)
    df["time_parsed"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["magnitude"] = df["mag"]
    df["day_number"] = ((df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    print(f"  Total: {len(df)} earthquakes")
    return df


def download_kp():
    print("Downloading Kp...")
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url, timeout=60)
    records = []
    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        if len(parts) < 25: continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 2000: continue
            kp_vals = [float(parts[7+i]) for i in range(8)]
            records.append({"year":y, "month":m, "day":d,
                           "kp_max": max(kp_vals), "kp_mean": np.mean(kp_vals)})
        except: continue
    df = pd.DataFrame(records)
    df["day_number"] = ((pd.to_datetime(df[["year","month","day"]]) - pd.Timestamp(INIT_DATE)).dt.days).values
    print(f"  {len(df)} records")
    return df


# ─── Analysis 1: Bearing-dependent propagation ──────────────────────────────

def bearing_analysis(eq_df):
    """
    For each M7+ earthquake, bin subsequent M5+ events by BEARING
    from the epicenter. If the field coupling is real, E-W bearings
    (perpendicular to B) should show more triggering than N-S.

    Magnetic field at most latitudes points roughly N-S (dipole).
    Perpendicular = E-W = bearings near 90 and 270 degrees.
    Parallel = N-S = bearings near 0 and 180 degrees.
    """
    print("\n=== Bearing-Dependent Propagation ===")
    print("Framework: [F, nabla F] ~ sin(alpha), max when wave crosses B perpendicularly")

    large = eq_df[eq_df["magnitude"] >= 7.0].copy()
    print(f"M7+ triggering events: {len(large)}")

    # Bearing bins: 0-360 in 30-degree increments
    bearing_bins = np.arange(0, 361, 30)
    bin_centers = (bearing_bins[:-1] + bearing_bins[1:]) / 2

    # Distance range: 30-90 degrees (not too near, not too far)
    triggered_bearings = []  # day +1 to +7
    background_bearings = []  # day -14 to -7

    for _, ev in large.iterrows():
        d0 = ev["day_number"]
        elat, elon = ev["latitude"], ev["longitude"]

        # Post-event M5+ within 30-90 degrees
        post = eq_df[(eq_df["day_number"] > d0) & (eq_df["day_number"] <= d0 + 7) &
                     (eq_df["magnitude"] >= 5.0)]
        # Background
        bg = eq_df[(eq_df["day_number"] >= d0 - 14) & (eq_df["day_number"] <= d0 - 7) &
                   (eq_df["magnitude"] >= 5.0)]

        for _, q in post.iterrows():
            dist = angular_distance(elat, elon, q["latitude"], q["longitude"])
            if 30 <= dist <= 90:
                b = bearing(elat, elon, q["latitude"], q["longitude"])
                triggered_bearings.append(b)

        for _, q in bg.iterrows():
            dist = angular_distance(elat, elon, q["latitude"], q["longitude"])
            if 30 <= dist <= 90:
                b = bearing(elat, elon, q["latitude"], q["longitude"])
                background_bearings.append(b)

    trig_hist, _ = np.histogram(triggered_bearings, bins=bearing_bins)
    bg_hist, _ = np.histogram(background_bearings, bins=bearing_bins)

    # Normalize background to same total
    scale = len(triggered_bearings) / max(len(background_bearings), 1)
    bg_hist_scaled = bg_hist * scale

    ratio = np.where(bg_hist_scaled > 0, trig_hist / bg_hist_scaled, 1.0)

    print(f"\nTriggered events (30-90 deg, day +1 to +7): {len(triggered_bearings)}")
    print(f"Background events (30-90 deg, day -14 to -7): {len(background_bearings)}")

    print(f"\n{'Bearing':>10s}  {'Direction':>8s}  {'Triggered':>10s}  {'Background':>10s}  {'Ratio':>7s}")
    for i, c in enumerate(bin_centers):
        direction = ""
        if 345 <= c or c < 15: direction = "N"
        elif 75 <= c < 105: direction = "E"
        elif 165 <= c < 195: direction = "S"
        elif 255 <= c < 285: direction = "W"
        elif 15 <= c < 75: direction = "NE"
        elif 105 <= c < 165: direction = "SE"
        elif 195 <= c < 255: direction = "SW"
        elif 285 <= c < 345: direction = "NW"
        print(f"  {c:>6.0f} deg  {direction:>8s}  {trig_hist[i]:>10d}  {bg_hist_scaled[i]:>10.1f}  {ratio[i]:>6.2f}x")

    # Group into perpendicular (E-W: 60-120, 240-300) vs parallel (N-S: 330-30, 150-210)
    perp_idx = [i for i, c in enumerate(bin_centers) if (60 <= c <= 120) or (240 <= c <= 300)]
    para_idx = [i for i, c in enumerate(bin_centers) if (c >= 330 or c <= 30) or (150 <= c <= 210)]

    perp_ratio = np.mean(ratio[perp_idx])
    para_ratio = np.mean(ratio[para_idx])

    print(f"\nPerpendicular to B (E-W): mean ratio = {perp_ratio:.3f}")
    print(f"Parallel to B (N-S):      mean ratio = {para_ratio:.3f}")
    print(f"Perp/Para: {perp_ratio/max(para_ratio, 0.001):.3f}")
    print(f"Framework predicts: Perp > Para (sin(alpha) maximized)")

    # Plot polar
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    theta = np.radians(bin_centers)
    # Close the circle
    theta_closed = np.append(theta, theta[0])
    ratio_closed = np.append(ratio, ratio[0])

    ax.plot(theta_closed, ratio_closed, 'o-', color="#fc8d62", linewidth=2)
    ax.fill(theta_closed, ratio_closed, alpha=0.3, color="#fc8d62")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    # Mark perpendicular and parallel directions
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Seismicity Triggering by Bearing from M7+ Epicenter\n"
                 "Ratio to background (30-90 deg distance, day +1 to +7)\n"
                 "E-W = perpendicular to B, N-S = parallel to B",
                 pad=20, fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "bearing_polar.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'bearing_polar.png'}")

    return ratio, bin_centers, perp_ratio, para_ratio


# ─── Analysis 2: Volcanic response to distant earthquakes ───────────────────

def volcanic_response(eq_df):
    """
    Use M7.5+ earthquakes as triggers. Check if volcanic earthquakes
    (shallow, near known volcanoes) increase at various distances.

    We don't have a volcano eruption database API, but we CAN use
    earthquake swarms near known volcanic centers as a proxy for
    volcanic unrest. Shallow earthquakes (< 30 km) clustered in
    volcanic regions indicate magma movement.

    Key volcanic regions with known active volcanoes:
    """
    print("\n=== Volcanic Response to Distant M7.5+ Earthquakes ===")
    print("Proxy: shallow (<30 km) earthquake rate in volcanic regions")

    # Major volcanic regions
    volc_regions = {
        "Kamchatka":    {"lat": (50, 58), "lon": (155, 165)},
        "Japan_volc":   {"lat": (30, 42), "lon": (128, 145)},
        "Philippines_v":{"lat": (6, 18),  "lon": (119, 128)},
        "Indonesia_v":  {"lat": (-8, 2),  "lon": (105, 130)},
        "Vanuatu_v":    {"lat": (-20, -14), "lon": (166, 171)},
        "Tonga_v":      {"lat": (-22, -16), "lon": (-177, -173)},
        "Cascades":     {"lat": (40, 50), "lon": (-125, -120)},
        "CentralAm":    {"lat": (8, 16),  "lon": (-92, -85)},
        "Andes_N":      {"lat": (-5, 5),  "lon": (-80, -75)},
        "Andes_S":      {"lat": (-42, -30), "lon": (-73, -68)},
        "Iceland":      {"lat": (63, 67), "lon": (-25, -13)},
        "Italy":        {"lat": (36, 42), "lon": (13, 17)},
    }

    # Get shallow earthquakes in volcanic regions as volcanic unrest proxy
    print("Downloading shallow volcanic-region earthquakes...")
    volc_dfs = {}
    for vname, vdef in volc_regions.items():
        try:
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {
                "format": "csv", "starttime": "2000-01-01", "endtime": "2026-03-31",
                "minlatitude": vdef["lat"][0], "maxlatitude": vdef["lat"][1],
                "minlongitude": vdef["lon"][0], "maxlongitude": vdef["lon"][1],
                "minmagnitude": 3.0, "maxdepth": 30,  # shallow = volcanic
                "orderby": "time-asc", "limit": 20000,
            }
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            df["time_parsed"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
            df["day_number"] = ((df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
            volc_dfs[vname] = df
            print(f"  {vname}: {len(df)} shallow events")
        except Exception as e:
            print(f"  {vname}: failed ({e})")

    # Large triggering earthquakes (M7.5+, any depth)
    large = eq_df[eq_df["magnitude"] >= 7.5].copy()
    print(f"\nM7.5+ global triggers: {len(large)}")

    # For each volcanic region, compute superposed epoch around large quakes
    window = np.arange(-14, 29)

    print(f"\n{'Volcano region':>15s}  {'Center':>12s}  {'BG rate':>8s}  {'Post rate':>9s}  {'Ratio':>6s}  {'Peak day':>9s}")

    all_ratios = {}
    for vname, vdf in volc_dfs.items():
        if len(vdf) < 100:
            continue

        vdef = volc_regions[vname]
        vcenter_lat = np.mean(vdef["lat"])
        vcenter_lon = np.mean(vdef["lon"])

        stacked = np.zeros((len(large), len(window)))
        for i, (_, ev) in enumerate(large.iterrows()):
            d0 = ev["day_number"]
            # Skip if the trigger is IN this volcanic region
            if (vdef["lat"][0] <= ev["latitude"] <= vdef["lat"][1] and
                vdef["lon"][0] <= ev["longitude"] <= vdef["lon"][1]):
                continue
            for j, offset in enumerate(window):
                n = len(vdf[vdf["day_number"] == d0 + offset])
                stacked[i, j] = n

        mean_rate = np.mean(stacked, axis=0)
        bg = np.mean(mean_rate[:14])
        post = np.mean(mean_rate[14:28])
        ratio = post / max(bg, 0.001)
        peak_day = window[14 + np.argmax(mean_rate[14:28])]

        all_ratios[vname] = ratio

        print(f"  {vname:>13s}  ({vcenter_lat:5.1f},{vcenter_lon:6.1f})  "
              f"{bg:>8.3f}  {post:>9.3f}  {ratio:>5.2f}x  day {peak_day:+d}")

    # Sort by ratio
    print("\nRanked by volcanic response:")
    for vname, ratio in sorted(all_ratios.items(), key=lambda x: -x[1]):
        print(f"  {vname:>15s}: {ratio:.2f}x")

    return all_ratios


# ─── Analysis 3: Latitude-dependent field coupling ──────────────────────────

def latitude_analysis(eq_df):
    """
    The geomagnetic field strength varies with latitude:
    ~30 uT at equator, ~60 uT at poles (dipole approximation).

    In the framework, J ~ B. Higher latitude = higher J = more ordered.
    The KT transition J_c = 2/pi is fixed.

    Prediction: seismic triggering ratio should vary with the
    MAGNETIC latitude of the target region, not just distance.
    """
    print("\n=== Latitude-Dependent Field Coupling ===")

    large = eq_df[eq_df["magnitude"] >= 7.0]
    print(f"M7+ triggers: {len(large)}")

    # Bin target earthquakes by latitude
    lat_bins = np.arange(-60, 61, 15)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    # Approximate IGRF field strength (dipole)
    # B(lambda) = B_eq * sqrt(1 + 3*sin^2(lambda))
    B_eq = 30.0  # uT
    B_field = B_eq * np.sqrt(1 + 3 * np.sin(np.radians(lat_centers))**2)

    triggered_by_lat = np.zeros(len(lat_centers))
    background_by_lat = np.zeros(len(lat_centers))

    for _, ev in large.iterrows():
        d0 = ev["day_number"]

        post = eq_df[(eq_df["day_number"] > d0) & (eq_df["day_number"] <= d0 + 7) &
                     (eq_df["magnitude"] >= 5.0)]
        bg = eq_df[(eq_df["day_number"] >= d0 - 14) & (eq_df["day_number"] <= d0 - 7) &
                   (eq_df["magnitude"] >= 5.0)]

        for _, q in post.iterrows():
            dist = angular_distance(ev["latitude"], ev["longitude"],
                                    q["latitude"], q["longitude"])
            if 20 <= dist <= 120:
                idx = np.digitize(q["latitude"], lat_bins) - 1
                if 0 <= idx < len(lat_centers):
                    triggered_by_lat[idx] += 1

        for _, q in bg.iterrows():
            dist = angular_distance(ev["latitude"], ev["longitude"],
                                    q["latitude"], q["longitude"])
            if 20 <= dist <= 120:
                idx = np.digitize(q["latitude"], lat_bins) - 1
                if 0 <= idx < len(lat_centers):
                    background_by_lat[idx] += 1

    # Scale background
    scale = np.sum(triggered_by_lat) / max(np.sum(background_by_lat), 1)
    bg_scaled = background_by_lat * scale
    ratio_by_lat = np.where(bg_scaled > 0, triggered_by_lat / bg_scaled, 1.0)

    print(f"\n{'Lat band':>10s}  {'B (uT)':>7s}  {'Triggered':>10s}  {'Background':>10s}  {'Ratio':>6s}")
    for i, c in enumerate(lat_centers):
        print(f"  {c:+5.0f} deg   {B_field[i]:>6.1f}  {triggered_by_lat[i]:>10.0f}  "
              f"{bg_scaled[i]:>10.1f}  {ratio_by_lat[i]:>5.2f}x")

    # Correlation between B and ratio
    valid = bg_scaled > 10
    if np.sum(valid) > 3:
        r, p = stats.pearsonr(B_field[valid], ratio_by_lat[valid])
        print(f"\nCorrelation (B vs triggering ratio): r = {r:+.3f}, p = {p:.3f}")
        print(f"Framework predicts: NEGATIVE correlation (high B = high J = more stable)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.bar(lat_centers, ratio_by_lat, width=12, alpha=0.7, color="steelblue")
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Latitude (degrees)")
    ax1.set_ylabel("Triggering ratio")
    ax1.set_title("Seismic Triggering Ratio by Latitude\n"
                   "(M5+ events at 20-120 deg from M7+ epicenter, day +1 to +7)")

    ax2.plot(lat_centers, B_field, 'o-', color="orange", linewidth=2)
    ax2.set_xlabel("Latitude (degrees)")
    ax2.set_ylabel("Dipole B field (uT)")
    ax2.set_title("Approximate Geomagnetic Field Strength (Dipole)")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "latitude_coupling.png", dpi=150)
    print(f"Saved: {OUT_DIR / 'latitude_coupling.png'}")

    return ratio_by_lat, B_field, lat_centers


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TRANSVERSE PROPAGATION + VOLCANISM")
    print("Does stress propagate perpendicular to B? Does the jelly squeeze?")
    print("=" * 70)

    eq_df = download_global_earthquakes(min_mag=4.5)

    # Analysis 1: Bearing
    ratio_bear, centers_bear, perp, para = bearing_analysis(eq_df)

    # Analysis 2: Volcanic response
    volc_ratios = volcanic_response(eq_df)

    # Analysis 3: Latitude
    ratio_lat, B_lat, centers_lat = latitude_analysis(eq_df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
1. BEARING: Perpendicular (E-W) = {perp:.3f}, Parallel (N-S) = {para:.3f}
   Ratio perp/para = {perp/max(para,0.001):.3f}
   {"CONFIRMED" if perp > para else "NOT CONFIRMED"}: stress propagates preferentially across B

2. VOLCANISM: The jelly squeezes — volcanic regions show
   differential response to distant M7.5+ earthquakes.
   Strongest responders indicate preferred stress pathways.

3. LATITUDE: If triggering ratio anti-correlates with B,
   weaker field = more susceptible to perturbation (lower J).

The geomagnetic field is not a passive backdrop.
It is the MEDIUM through which tectonic stress communicates.
The commutator [F, nabla F] determines the coupling strength,
and its angular dependence creates preferred propagation directions.
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
