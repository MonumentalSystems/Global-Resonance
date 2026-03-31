#!/usr/bin/env python3
"""
Two-Grade Arrival Test
========================
The commutator [F, nabla F] has three grade projections:
  Grade-0 (EM): travels at c, arrives in 8 minutes after flare
  Grade-2 (mechanical): travels with CME, arrives hours-days later
  Grade-4 (vacuum/pressure): the sustained background shift

If the solar-seismic coupling is real AND goes through the
commutator, we should see TWO distinct seismic responses
to the same solar event:
  1. An INSTANT response (minutes-hours) to the X-ray/EUV flare
     (grade-0 projection: ionospheric SID -> Schumann shift ->
      telluric current -> crustal stress)
  2. A DELAYED response (1-3 days) to the CME arrival
     (grade-2 projection: magnetospheric compression ->
      ring current -> sustained crustal perturbation)

This script downloads:
  - NOAA GOES solar flare list (X-class flares with precise timing)
  - Global earthquake catalog (already cached)
  - Kp index (already cached)

And tests: is there a seismic signal at BOTH timescales?
Also downloads and caches all data for future local analysis.
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
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
INIT_DATE = dt.datetime(2000, 1, 1)


# ─── Cached loaders ─────────────────────────────────────────────────────────

def get_earthquakes():
    cache = DATA_DIR / "earthquakes_m4.5.csv"
    if cache.exists():
        print(f"Loading cached earthquakes...")
        df = pd.read_csv(cache, parse_dates=["time_parsed"])
        print(f"  {len(df)} events")
        return df
    print("ERROR: Run local_analysis.py first to cache earthquake data")
    sys.exit(1)

def get_kp():
    cache = DATA_DIR / "kp_daily.csv"
    if cache.exists():
        print(f"Loading cached Kp...")
        return pd.read_csv(cache)
    print("ERROR: Run local_analysis.py first to cache Kp data")
    sys.exit(1)


def get_solar_flares():
    """Download DONKI flare list — has precise timing and class."""
    cache = DATA_DIR / "solar_flares.csv"
    if cache.exists():
        print(f"Loading cached solar flares...")
        df = pd.read_csv(cache, parse_dates=["beginTime", "peakTime", "endTime"])
        print(f"  {len(df)} flares")
        return df

    print("Downloading solar flare catalog from DONKI...")
    base_url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR"
    all_flares = []
    for year in range(2010, 2027):
        for half in [0, 1]:
            start = f"{year}-{'01' if half==0 else '07'}-01"
            end = f"{year}-{'06-30' if half==0 else '12-31'}"
            try:
                resp = requests.get(base_url, params={
                    "startDate": start, "endDate": end
                }, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                for flare in data:
                    rec = {
                        "beginTime": flare.get("beginTime"),
                        "peakTime": flare.get("peakTime"),
                        "endTime": flare.get("endTime"),
                        "classType": flare.get("classType", ""),
                        "sourceLocation": flare.get("sourceLocation", ""),
                        "activeRegionNum": flare.get("activeRegionNum"),
                    }
                    all_flares.append(rec)
                print(f"  {start}: {len(data)} flares")
            except Exception as e:
                print(f"  {start}: failed ({e})")

    df = pd.DataFrame(all_flares)
    for col in ["beginTime", "peakTime", "endTime"]:
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)

    # Parse class into numeric (X10 = 10, X1 = 1, M5 = 0.5, etc.)
    def parse_class(c):
        if not isinstance(c, str) or len(c) < 2:
            return np.nan
        letter = c[0]
        try:
            num = float(c[1:])
        except:
            return np.nan
        multipliers = {"X": 1.0, "M": 0.1, "C": 0.01, "B": 0.001, "A": 0.0001}
        return multipliers.get(letter, 0) * num

    df["class_numeric"] = df["classType"].apply(parse_class)
    df["day_number"] = ((df["peakTime"] - pd.Timestamp(INIT_DATE)).dt.days).values
    df["hour"] = df["peakTime"].dt.hour

    df.to_csv(cache, index=False)
    print(f"  Cached {len(df)} flares to {cache}")
    return df


def get_dscovr_solar_wind():
    """Download DSCOVR/ACE solar wind summary — daily Bz, speed, density."""
    cache = DATA_DIR / "solar_wind_daily.csv"
    if cache.exists():
        print(f"Loading cached solar wind...")
        return pd.read_csv(cache)

    # SWPC JSON service for recent data
    print("Downloading solar wind data from SWPC...")
    # The SWPC provides 7-day JSON; for longer archives we'd need DSCOVR NCEI
    # For now, derive solar wind proxy from Kp/Ap (which we already have)
    print("  Using Kp/Ap as solar wind proxy (full archive available)")
    print("  For detailed Bz/speed/density: see https://www.ngdc.noaa.gov/dscovr/")
    return None


# ─── Two-Grade Analysis ─────────────────────────────────────────────────────

def two_grade_test(eq_df, flare_df, kp_df):
    """
    For each X-class flare:
    1. Count earthquakes in the 0-6 hour window after the flare
       (grade-0: EM pulse arrives at c)
    2. Count earthquakes in the 24-72 hour window
       (grade-2: CME arrives mechanically)
    3. Compare both to a background rate
    """
    print("\n=== Two-Grade Arrival Test ===")
    print("Grade-0 (EM, speed of light): 0-6 hours after flare")
    print("Grade-2 (mechanical, CME): 24-72 hours after flare")

    # Use M and X class flares
    x_flares = flare_df[flare_df["class_numeric"] >= 0.5].copy()  # M5+
    print(f"\nM5+ flares: {len(x_flares)}")

    # Also separate by class
    x_only = flare_df[flare_df["class_numeric"] >= 1.0]  # X1+
    print(f"X1+ flares: {len(x_only)}")

    # For each flare, compute earthquake rate in windows
    # Need hourly resolution — use time_parsed
    eq_df = eq_df.copy()
    eq_times = eq_df["time_parsed"].values.astype('datetime64[h]')

    def count_in_window(flare_time, hours_start, hours_end):
        """Count M4.5+ earthquakes in a time window after flare."""
        t0 = np.datetime64(flare_time, 'h')
        t_start = t0 + np.timedelta64(hours_start, 'h')
        t_end = t0 + np.timedelta64(hours_end, 'h')
        return np.sum((eq_times >= t_start) & (eq_times < t_end))

    results = []
    for _, flare in x_flares.iterrows():
        ft = flare["peakTime"]
        if pd.isna(ft):
            continue

        # Grade-0 window: 0-6 hours
        n_grade0 = count_in_window(ft, 0, 6)
        # Grade-2 window: 24-72 hours (CME arrival)
        n_grade2 = count_in_window(ft, 24, 72)
        # Background: -168 to -24 hours (1 week before, excluding 24h pre-flare)
        n_bg = count_in_window(ft, -168, -24)
        # Normalize background to 6-hour rate
        bg_rate_6h = n_bg / (144 / 6)
        # Normalize background to 48-hour rate
        bg_rate_48h = n_bg / (144 / 48)

        results.append({
            "flare_time": ft,
            "class": flare["classType"],
            "class_num": flare["class_numeric"],
            "n_grade0": n_grade0,
            "n_grade2": n_grade2,
            "bg_rate_6h": bg_rate_6h,
            "bg_rate_48h": bg_rate_48h,
        })

    rdf = pd.DataFrame(results)

    # Statistics
    print(f"\nResults across {len(rdf)} M5+ flares:")
    print(f"  Grade-0 (0-6h):   mean = {rdf['n_grade0'].mean():.2f} quakes, "
          f"bg = {rdf['bg_rate_6h'].mean():.2f}, "
          f"ratio = {rdf['n_grade0'].mean()/max(rdf['bg_rate_6h'].mean(), 0.001):.3f}")
    print(f"  Grade-2 (24-72h): mean = {rdf['n_grade2'].mean():.2f} quakes, "
          f"bg = {rdf['bg_rate_48h'].mean():.2f}, "
          f"ratio = {rdf['n_grade2'].mean()/max(rdf['bg_rate_48h'].mean(), 0.001):.3f}")

    # Mann-Whitney tests
    _, p0 = stats.wilcoxon(rdf["n_grade0"] - rdf["bg_rate_6h"])
    _, p2 = stats.wilcoxon(rdf["n_grade2"] - rdf["bg_rate_48h"])
    print(f"  Grade-0 Wilcoxon p = {p0:.4f}")
    print(f"  Grade-2 Wilcoxon p = {p2:.4f}")

    # Split by flare class
    for label, subset in [("X1+ only", rdf[rdf["class_num"] >= 1.0]),
                          ("M5-M9", rdf[(rdf["class_num"] >= 0.5) & (rdf["class_num"] < 1.0)])]:
        if len(subset) > 10:
            r0 = subset["n_grade0"].mean() / max(subset["bg_rate_6h"].mean(), 0.001)
            r2 = subset["n_grade2"].mean() / max(subset["bg_rate_48h"].mean(), 0.001)
            print(f"\n  {label} ({len(subset)} flares):")
            print(f"    Grade-0 ratio: {r0:.3f}")
            print(f"    Grade-2 ratio: {r2:.3f}")

    # Hourly superposed epoch
    print("\n  Hourly superposed epoch around M5+ flares:")
    hour_window = np.arange(-48, 97)  # -48h to +96h
    hourly_counts = np.zeros((len(rdf), len(hour_window)))

    for i, (_, row) in enumerate(rdf.iterrows()):
        ft = row["flare_time"]
        if pd.isna(ft):
            continue
        for j, h in enumerate(hour_window):
            hourly_counts[i, j] = count_in_window(ft, h, h+1)

    mean_hourly = np.mean(hourly_counts, axis=0)
    bg_hourly = np.mean(mean_hourly[:48])  # pre-flare background

    print(f"  Background rate: {bg_hourly:.3f} quakes/hour")
    print(f"  Hour  Rate   Ratio")
    for h_idx in [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                   60, 66, 72, 78, 84, 90, 96]:
        if h_idx < len(hour_window):
            h = hour_window[h_idx]
            r = mean_hourly[h_idx]
            ratio = r / max(bg_hourly, 0.001)
            marker = ""
            if h == 0: marker = " <-- FLARE (grade-0 arrival)"
            if h == 24: marker = " <-- +24h"
            if h == 48: marker = " <-- +48h (typical CME arrival)"
            if h == 72: marker = " <-- +72h"
            print(f"    {h:+4d}h  {r:.3f}  {ratio:.2f}x{marker}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(hour_window, mean_hourly, color="steelblue", linewidth=1)
    ax.axhline(bg_hourly, color="red", linestyle="--", alpha=0.5,
               label=f"Background: {bg_hourly:.3f}/hr")
    ax.axvline(0, color="orange", linewidth=2, alpha=0.8, label="Flare (grade-0)")
    ax.axvspan(24, 72, alpha=0.1, color="green", label="CME window (grade-2)")
    ax.set_xlabel("Hours relative to M5+ solar flare")
    ax.set_ylabel("Mean earthquake rate (M4.5+ per hour)")
    ax.set_title(f"Two-Grade Test: Seismicity After {len(rdf)} M5+ Solar Flares\n"
                 f"Grade-0 (EM, instant) vs Grade-2 (CME, delayed)")
    ax.legend()
    ax.set_xlim(-48, 96)

    # Smoothed version
    ax = axes[1]
    kernel = np.ones(6) / 6  # 6-hour running mean
    smoothed = np.convolve(mean_hourly, kernel, mode='same')
    ax.plot(hour_window, smoothed, color="steelblue", linewidth=2)
    ax.axhline(bg_hourly, color="red", linestyle="--", alpha=0.5)
    ax.axvline(0, color="orange", linewidth=2, alpha=0.8, label="Flare")
    ax.axvspan(24, 72, alpha=0.1, color="green", label="CME window")
    ax.set_xlabel("Hours relative to M5+ solar flare")
    ax.set_ylabel("6-hour smoothed earthquake rate")
    ax.set_title("Smoothed (6-hour running mean)")
    ax.legend()
    ax.set_xlim(-48, 96)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "two_grades.png", dpi=150)
    print(f"\n  Saved: {OUT_DIR / 'two_grades.png'}")

    return rdf


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TWO-GRADE ARRIVAL: EM (instant) vs MECHANICAL (delayed)")
    print("=" * 70)

    eq_df = get_earthquakes()
    kp_df = get_kp()
    flare_df = get_solar_flares()

    rdf = two_grade_test(eq_df, flare_df, kp_df)

    print("\n" + "=" * 70)
    print("FRAMEWORK PREDICTION")
    print("=" * 70)
    print("""
  [F, nabla F] decomposes into three grades:

  Grade 0 (scalar): F . nabla F
    -> Electromagnetic coupling
    -> Travels at c (8 minutes from sun)
    -> Ionospheric SID -> Schumann perturbation -> telluric currents
    -> Expected: subtle signal at 0-6 hours

  Grade 2 (bivector): [F, nabla F] / 2
    -> Mechanical/gravitational coupling
    -> Travels with CME (1-3 days from sun)
    -> Magnetospheric compression -> ring current -> crustal stress
    -> Expected: stronger signal at 24-72 hours

  Grade 4 (pseudoscalar): F ^ nabla F
    -> Vacuum/pressure shift
    -> The sustained background after the storm
    -> Expected: long-term rate change lasting days-weeks

  If BOTH grade-0 and grade-2 show signal, the coupling is
  confirmed as going through the FULL geometric product,
  not just one mechanism.

  EARTHQUAKE LIGHTS are the grade-0 projection of [F, nabla F]
  at the rupture surface. They should occur at Schumann
  harmonics because the local cavity mode IS the Schumann mode
  seen from within the crust.

  VOLCANIC LIGHTNING is the same grade-0 projection but at the
  vent, where the charged ash plume creates a local resonator.
""")

    print("Done. All data cached in:", DATA_DIR)
    print("Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
