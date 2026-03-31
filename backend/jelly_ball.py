#!/usr/bin/env python3
"""
Jelly Ball Analysis
====================
The CME hits the magnetosphere at the subsolar point.
The direct impact SUPPRESSES earthquakes (stress relief, J > J_c).
The wave propagates outward through the crust.
At some angular distance, the wave triggers earthquakes at critical faults.

Test: bin global seismicity by angular distance from subsolar point
at the time of geomagnetic sudden impulses. We should see:
  - Suppression near 0° (eye of the storm)
  - Enhancement at some characteristic angle (the wave front)
  - Return to background at large angles (wave dissipated)

The subsolar point at time t is:
  lat = solar declination (~±23.5° with season)
  lon = 180° - 15°×(hour_UT)  (the sun moves 15°/hour westward)
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
import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
INIT_DATE = dt.datetime(2000, 1, 1)


def subsolar_point(dt_utc):
    """Approximate subsolar point (lat, lon) at a given UTC datetime."""
    # Solar declination (simplified, ignores equation of time)
    doy = dt_utc.timetuple().tm_yday
    decl = -23.44 * np.cos(np.radians(360 / 365 * (doy + 10)))

    # Subsolar longitude: noon at Greenwich = lon 0, moves west at 15°/hr
    hour = dt_utc.hour + dt_utc.minute / 60.0
    lon = 180.0 - 15.0 * hour  # at UT=0, subsolar is at 180°E (dateline)
    if lon < -180:
        lon += 360
    if lon > 180:
        lon -= 360

    return decl, lon


def angular_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in degrees between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def download_global_earthquakes(min_mag=5.0):
    """Download global earthquake catalog (M>=5 to keep it manageable)."""
    print(f"Downloading global earthquakes (M>={min_mag})...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    all_dfs = []
    # Download year by year to avoid hitting the 20K limit
    for year in range(2000, 2027):
        params = {
            "format": "csv",
            "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31",
            "minmagnitude": min_mag,
            "orderby": "time-asc",
            "limit": 20000,
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
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
    print(f"  Total: {len(df)} global earthquakes")
    return df


def download_kp_3hourly():
    """Download 3-hourly Kp from GFZ."""
    print("Downloading 3-hourly Kp from GFZ...")
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    records = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 25:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if year < 2000:
                continue
            kp_vals = [float(parts[7 + i]) for i in range(8)]
            for slot, kp in enumerate(kp_vals):
                hour = slot * 3
                records.append({
                    "year": year, "month": month, "day": day, "hour": hour, "kp": kp,
                })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["day_number"] = ((df["datetime"] - pd.Timestamp(INIT_DATE)).dt.days).values
    df["dkp_dt"] = df["kp"].diff().fillna(0)
    print(f"  Got {len(df)} 3-hourly records")
    return df


def find_impulses(kp_df, threshold=2.0, min_gap_days=3):
    """Find sudden geomagnetic impulses."""
    impulses = kp_df[kp_df["dkp_dt"] >= threshold].copy()
    days = impulses["day_number"].unique()
    filtered = [days[0]]
    for d in days[1:]:
        if d - filtered[-1] >= min_gap_days:
            filtered.append(d)

    # Get the datetime of each impulse (first big jump that day)
    result = []
    for d in filtered:
        day_data = impulses[impulses["day_number"] == d].iloc[0]
        result.append({
            "day_number": d,
            "datetime": day_data["datetime"],
            "hour": day_data["hour"],
            "kp": day_data["kp"],
            "dkp": day_data["dkp_dt"],
        })

    return pd.DataFrame(result)


def jelly_ball_analysis(eq_df, impulses_df):
    """
    For each impulse, compute angular distance of every earthquake
    (within ±3 days) from the subsolar point at impulse time.
    Bin by angular distance and compare to background.
    """
    print("\n=== Jelly Ball: Seismicity vs Angular Distance from Subsolar Point ===")

    # Angular distance bins (degrees)
    bins = np.arange(0, 181, 15)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Collect earthquake counts in each angular bin
    # For impulse days (day 0, +1, +2) and background (day -7 to -3)
    impulse_counts = np.zeros((len(impulses_df), len(bins) - 1))
    background_counts = np.zeros((len(impulses_df), len(bins) - 1))

    # Also collect for the "wave arrival" window: +1 to +3 days
    wave_counts = np.zeros((len(impulses_df), len(bins) - 1))

    for idx, imp in impulses_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing impulse {idx}/{len(impulses_df)}", flush=True)

        imp_dt = imp["datetime"]
        imp_day = imp["day_number"]

        # Subsolar point at impulse time
        ss_lat, ss_lon = subsolar_point(imp_dt)

        # Impulse-day earthquakes (day 0)
        eq_impulse = eq_df[eq_df["day_number"] == imp_day]
        if len(eq_impulse) > 0:
            dists = angular_distance(ss_lat, ss_lon,
                                     eq_impulse["latitude"].values,
                                     eq_impulse["longitude"].values)
            counts, _ = np.histogram(dists, bins=bins)
            impulse_counts[idx] = counts

        # Wave arrival: day +1 to +3
        eq_wave = eq_df[(eq_df["day_number"] >= imp_day + 1) &
                        (eq_df["day_number"] <= imp_day + 3)]
        if len(eq_wave) > 0:
            dists = angular_distance(ss_lat, ss_lon,
                                     eq_wave["latitude"].values,
                                     eq_wave["longitude"].values)
            counts, _ = np.histogram(dists, bins=bins)
            wave_counts[idx] = counts / 3.0  # per-day rate

        # Background: day -7 to -3 (5 days, normalize to per-day)
        eq_bg = eq_df[(eq_df["day_number"] >= imp_day - 7) &
                      (eq_df["day_number"] <= imp_day - 3)]
        if len(eq_bg) > 0:
            dists = angular_distance(ss_lat, ss_lon,
                                     eq_bg["latitude"].values,
                                     eq_bg["longitude"].values)
            counts, _ = np.histogram(dists, bins=bins)
            background_counts[idx] = counts / 5.0  # per-day rate

    # Average across all impulses
    mean_impulse = np.mean(impulse_counts, axis=0)
    mean_wave = np.mean(wave_counts, axis=0)
    mean_bg = np.mean(background_counts, axis=0)
    sem_impulse = stats.sem(impulse_counts, axis=0)
    sem_wave = stats.sem(wave_counts, axis=0)
    sem_bg = stats.sem(background_counts, axis=0)

    # Normalize by solid angle (bins near equator have more area)
    # Solid angle of a spherical zone: 2*pi*(cos(theta1) - cos(theta2))
    solid_angles = 2 * np.pi * np.abs(
        np.cos(np.radians(bins[:-1])) - np.cos(np.radians(bins[1:]))
    )
    # Normalize to density (quakes per steradian per day)
    density_impulse = mean_impulse / solid_angles
    density_wave = mean_wave / solid_angles
    density_bg = mean_bg / solid_angles

    # Ratio to background
    ratio_impulse = np.where(density_bg > 0, density_impulse / density_bg, 1.0)
    ratio_wave = np.where(density_bg > 0, density_wave / density_bg, 1.0)

    print(f"\nAngular distance from subsolar point:")
    print(f"  {'Angle':>8s}  {'BG rate':>8s}  {'Day 0':>8s}  {'Ratio 0':>8s}  {'Day+1-3':>8s}  {'Ratio+':>8s}")
    for i, center in enumerate(bin_centers):
        print(f"  {center:>6.0f} deg  {density_bg[i]:>8.4f}  {density_impulse[i]:>8.4f}  "
              f"{ratio_impulse[i]:>7.2f}x  {density_wave[i]:>8.4f}  {ratio_wave[i]:>7.2f}x")

    # ─── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top: absolute rates
    ax = axes[0]
    w = 4
    ax.bar(bin_centers - w, density_bg, width=w, alpha=0.7, color="#66c2a5", label="Background (day -7 to -3)")
    ax.bar(bin_centers, density_impulse, width=w, alpha=0.7, color="#fc8d62", label="Impulse day (day 0)")
    ax.bar(bin_centers + w, density_wave, width=w, alpha=0.7, color="#8da0cb", label="Wave arrival (day +1 to +3)")
    ax.set_ylabel("Earthquake density\n(per steradian per day)")
    ax.set_xlabel("Angular distance from subsolar point (degrees)")
    ax.set_title("Jelly Ball: Seismicity vs Distance from CME Impact Point\n"
                 f"Global M>=5.0, {len(impulses_df)} sudden impulses stacked, 2000-2026")
    ax.legend()

    # Bottom: ratio to background
    ax = axes[1]
    ax.plot(bin_centers, ratio_impulse, 'o-', color="#fc8d62", label="Day 0 / background", linewidth=2)
    ax.plot(bin_centers, ratio_wave, 's-', color="#8da0cb", label="Day +1-3 / background", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(bin_centers, 0.8, 1.2, alpha=0.1, color="gray")
    ax.set_ylabel("Ratio to background")
    ax.set_xlabel("Angular distance from subsolar point (degrees)")
    ax.set_title("Ratio: impulse/wave seismicity relative to background")
    ax.legend()
    ax.set_ylim(0.4, 2.0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "jelly_ball.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'jelly_ball.png'}")

    # ─── Antipodal analysis ──────────────────────────────────────────────
    # The jelly ball should also show signal at the ANTIPODAL point (180°)
    # where the wave reconverges
    near = ratio_wave[:4]     # 0-60°
    mid = ratio_wave[4:8]     # 60-120°
    far = ratio_wave[8:]      # 120-180°

    print(f"\nZone summary (wave arrival, day +1 to +3):")
    print(f"  Near subsolar (0-60 deg):    mean ratio = {np.mean(near):.3f}")
    print(f"  Mid-range (60-120 deg):      mean ratio = {np.mean(mid):.3f}")
    print(f"  Far / antipodal (120-180 deg): mean ratio = {np.mean(far):.3f}")

    return density_impulse, density_wave, density_bg, ratio_impulse, ratio_wave


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("JELLY BALL: Non-Planar Seismic Response to CME Impact")
    print("The eye of the storm should be quiet.")
    print("The wave front should be loud.")
    print("=" * 70)

    eq_df = download_global_earthquakes(min_mag=5.0)
    kp_df = download_kp_3hourly()
    impulses = find_impulses(kp_df, threshold=2.0, min_gap_days=3)
    print(f"\nFound {len(impulses)} sudden impulses")

    results = jelly_ball_analysis(eq_df, impulses)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
KT Framework prediction:
  - CME impact at subsolar point pushes J ABOVE J_c (ordered, stable)
  - Stress wave propagates outward through crustal oscillator network
  - At characteristic angle (~60-90 deg?), wave arrives at critical faults
  - Those faults are pushed THROUGH J_c -> vortex unbinding -> earthquake
  - Eye of storm (0-30 deg from subsolar): suppressed
  - Wave front (60-120 deg): enhanced
  - Antipodal reconvergence (150-180 deg): possibly enhanced again

If the ratio plot shows suppression near 0 and enhancement at mid-range,
the non-planar coupling is confirmed.
""")
    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
