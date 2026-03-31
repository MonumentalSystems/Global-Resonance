#!/usr/bin/env python3
"""
Grade-4 Signal: The 18-Hour Peak
==================================
The two-grade test found three timescales:
  Hour 0: suppression (grade-0, EM pulse)
  Hour +18: enhancement peak (grade-4? shock? SEP? relaxation?)
  Hour +24-72: mild enhancement (grade-2, CME body)

What arrives at 18 hours?
  1. Interplanetary shock (if CME speed ~ 2000+ km/s)
  2. SEP flux peak (sustained proton bombardment)
  3. Ionospheric relaxation crossing back through J_c

Test: does the 18h peak correlate with flare properties?
  - Flare duration (longer flare = more EUV = longer relaxation)
  - CME speed (faster CME = earlier shock = earlier peak)
  - Flare class (stronger flare = all effects larger)
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

def get_earthquakes():
    df = pd.read_csv(DATA_DIR / "earthquakes_m4.5.csv", parse_dates=["time_parsed"])
    print(f"Earthquakes: {len(df)}")
    return df

def get_flares():
    df = pd.read_csv(DATA_DIR / "solar_flares.csv", parse_dates=["beginTime","peakTime","endTime"])
    print(f"Flares: {len(df)}")
    return df

def get_cmes():
    """Load CME data from DONKI (cached from obliquity run)."""
    cache = DATA_DIR / "cmes.csv"
    if cache.exists():
        print(f"Loading cached CMEs...")
        return pd.read_csv(cache, parse_dates=["datetime"])

    print("Downloading CMEs from DONKI...")
    base_url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"
    all_cmes = []
    for year in range(2010, 2027):
        for half in [0, 1]:
            start = f"{year}-{'01' if half==0 else '07'}-01"
            end = f"{year}-{'06-30' if half==0 else '12-31'}"
            try:
                resp = requests.get(base_url, params={"startDate":start,"endDate":end}, timeout=60)
                data = resp.json()
                for cme in data:
                    analyses = cme.get("cmeAnalyses", [])
                    best = None
                    for a in analyses:
                        if a.get("speed") and a.get("speed") > 0:
                            best = a
                            break
                    if best is None and analyses:
                        best = analyses[0]
                    speed = best.get("speed") if best else None
                    half_angle = best.get("halfAngle") if best else None
                    all_cmes.append({
                        "datetime": cme.get("startTime"),
                        "speed": speed,
                        "halfAngle": half_angle,
                        "sourceLocation": cme.get("sourceLocation",""),
                    })
            except: pass

    df = pd.DataFrame(all_cmes)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df.to_csv(cache, index=False)
    print(f"  Cached {len(df)} CMEs")
    return df


def count_quakes_in_window(eq_times, ref_time, h_start, h_end):
    t0 = np.datetime64(ref_time, 'h')
    return np.sum((eq_times >= t0 + np.timedelta64(h_start,'h')) &
                  (eq_times < t0 + np.timedelta64(h_end,'h')))


def main():
    print("=" * 70)
    print("GRADE-4 SIGNAL: What Arrives at 18 Hours?")
    print("=" * 70)

    eq_df = get_earthquakes()
    flare_df = get_flares()
    cme_df = get_cmes()

    eq_times = eq_df["time_parsed"].values.astype('datetime64[h]')

    # M5+ flares with duration
    flares = flare_df[flare_df["class_numeric"] >= 0.5].copy()
    flares["duration_min"] = (flares["endTime"] - flares["beginTime"]).dt.total_seconds() / 60
    flares = flares[flares["duration_min"] > 0].copy()
    print(f"\nM5+ flares with duration: {len(flares)}")
    print(f"Duration range: {flares['duration_min'].min():.0f} - {flares['duration_min'].max():.0f} min")
    print(f"Median duration: {flares['duration_min'].median():.0f} min")

    # Match flares to CMEs (within 6 hours)
    flares["cme_speed"] = np.nan
    for idx, flare in flares.iterrows():
        ft = flare["peakTime"]
        if pd.isna(ft):
            continue
        # Find CME within 6 hours of flare
        time_diff = abs((cme_df["datetime"] - ft).dt.total_seconds())
        close = cme_df[time_diff < 6*3600]
        if len(close) > 0 and close["speed"].notna().any():
            flares.loc[idx, "cme_speed"] = close.loc[close["speed"].idxmax(), "speed"]

    n_with_cme = flares["cme_speed"].notna().sum()
    print(f"Flares matched to CME: {n_with_cme}")

    # ─── Test 1: Duration dependence ─────────────────────────────────────
    print("\n=== Test 1: Flare Duration vs 18h Peak ===")

    med_dur = flares["duration_min"].median()
    long_flares = flares[flares["duration_min"] > med_dur]
    short_flares = flares[flares["duration_min"] <= med_dur]
    print(f"Long flares (>{med_dur:.0f} min): {len(long_flares)}")
    print(f"Short flares (<={med_dur:.0f} min): {len(short_flares)}")

    hour_window = np.arange(-24, 73)

    def epoch_for_subset(subset, label):
        hourly = np.zeros((len(subset), len(hour_window)))
        for i, (_, fl) in enumerate(subset.iterrows()):
            ft = fl["peakTime"]
            if pd.isna(ft): continue
            for j, h in enumerate(hour_window):
                hourly[i, j] = count_quakes_in_window(eq_times, ft, h, h+1)
        mean_rate = np.mean(hourly, axis=0)
        bg = np.mean(mean_rate[:24])
        # Find peak in 12-24h window
        window_12_24 = mean_rate[36:48]  # indices 36-47 = hours +12 to +23
        peak_hour = hour_window[36 + np.argmax(window_12_24)]
        peak_rate = np.max(window_12_24)
        peak_ratio = peak_rate / max(bg, 0.001)
        print(f"  {label}: bg={bg:.3f}/hr, peak at +{peak_hour}h = {peak_rate:.3f} ({peak_ratio:.2f}x)")
        return mean_rate, bg

    rate_long, bg_long = epoch_for_subset(long_flares, "Long flares")
    rate_short, bg_short = epoch_for_subset(short_flares, "Short flares")

    # ─── Test 2: CME speed dependence ────────────────────────────────────
    print("\n=== Test 2: CME Speed vs Peak Timing ===")

    flares_with_cme = flares[flares["cme_speed"].notna()].copy()
    if len(flares_with_cme) > 20:
        med_speed = flares_with_cme["cme_speed"].median()
        fast_cme = flares_with_cme[flares_with_cme["cme_speed"] > med_speed]
        slow_cme = flares_with_cme[flares_with_cme["cme_speed"] <= med_speed]

        print(f"Fast CME (>{med_speed:.0f} km/s): {len(fast_cme)}")
        print(f"Slow CME (<={med_speed:.0f} km/s): {len(slow_cme)}")

        # Expected shock arrival time: t = 1AU / (1.5 * v_cme)
        # 1 AU = 1.496e8 km, shock ~ 1.5x CME speed
        au_km = 1.496e8
        fast_arrival = au_km / (1.5 * fast_cme["cme_speed"].median()) / 3600
        slow_arrival = au_km / (1.5 * slow_cme["cme_speed"].median()) / 3600
        print(f"Expected shock arrival: fast={fast_arrival:.0f}h, slow={slow_arrival:.0f}h")

        rate_fast, bg_fast = epoch_for_subset(fast_cme, f"Fast CME (shock ~{fast_arrival:.0f}h)")
        rate_slow, bg_slow = epoch_for_subset(slow_cme, f"Slow CME (shock ~{slow_arrival:.0f}h)")

    # ─── Test 3: X-class only with hourly detail ────────────────────────
    print("\n=== Test 3: X-Class Flares — Hour-by-Hour ===")

    x_flares = flares[flares["class_numeric"] >= 1.0]
    print(f"X1+ flares: {len(x_flares)}")

    hourly_x = np.zeros((len(x_flares), len(hour_window)))
    for i, (_, fl) in enumerate(x_flares.iterrows()):
        ft = fl["peakTime"]
        if pd.isna(ft): continue
        for j, h in enumerate(hour_window):
            hourly_x[i, j] = count_quakes_in_window(eq_times, ft, h, h+1)

    mean_x = np.mean(hourly_x, axis=0)
    bg_x = np.mean(mean_x[:24])

    print(f"\nHour-by-hour X-class response:")
    print(f"Background: {bg_x:.3f}/hr")
    for h_idx, h in enumerate(hour_window):
        if -6 <= h <= 48:
            ratio = mean_x[h_idx] / max(bg_x, 0.001)
            bar = "#" * int(max(0, ratio - 0.5) * 40)
            marker = ""
            if h == 0: marker = " <-- FLARE"
            if h == 18: marker = " <-- 18h PEAK?"
            if h == 24: marker = " <-- +24h"
            if h == 36: marker = " <-- +36h"
            if h == 48: marker = " <-- +48h"
            print(f"    {h:+4d}h: {mean_x[h_idx]:.3f} ({ratio:.2f}x) {bar}{marker}")

    # ─── Test 4: Flare duration vs response timing ──────────────────────
    print("\n=== Test 4: Does Longer Flare = Later Peak? ===")

    # Bin flares by duration quartile
    quartiles = flares["duration_min"].quantile([0.25, 0.5, 0.75]).values
    bins_dur = [0, quartiles[0], quartiles[1], quartiles[2], flares["duration_min"].max() + 1]
    labels_dur = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]

    for i in range(4):
        subset = flares[(flares["duration_min"] >= bins_dur[i]) &
                        (flares["duration_min"] < bins_dur[i+1])]
        if len(subset) < 10:
            continue

        hourly_q = np.zeros((len(subset), len(hour_window)))
        for j, (_, fl) in enumerate(subset.iterrows()):
            ft = fl["peakTime"]
            if pd.isna(ft): continue
            for k, h in enumerate(hour_window):
                hourly_q[j, k] = count_quakes_in_window(eq_times, ft, h, h+1)

        mean_q = np.mean(hourly_q, axis=0)
        bg_q = np.mean(mean_q[:24])

        # Find peak in 6-30h window
        search_start, search_end = 30, 54  # hours +6 to +30
        window = mean_q[search_start:search_end]
        peak_h = hour_window[search_start + np.argmax(window)]
        peak_ratio = np.max(window) / max(bg_q, 0.001)

        dur_range = f"{bins_dur[i]:.0f}-{bins_dur[i+1]:.0f} min"
        print(f"  {labels_dur[i]} ({dur_range}, N={len(subset)}): "
              f"peak at +{peak_h}h ({peak_ratio:.2f}x bg)")

    # ─── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # Panel 1: Long vs short flare duration
    ax = axes[0]
    kernel = np.ones(3) / 3
    ax.plot(hour_window, np.convolve(rate_long/max(bg_long,0.001), kernel, 'same'),
            color="#e78ac3", linewidth=2, label=f"Long flares (>{med_dur:.0f} min)")
    ax.plot(hour_window, np.convolve(rate_short/max(bg_short,0.001), kernel, 'same'),
            color="#66c2a5", linewidth=2, label=f"Short flares (<={med_dur:.0f} min)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="orange", linewidth=2, alpha=0.5)
    ax.axvline(18, color="red", linestyle=":", alpha=0.7, label="+18h peak")
    ax.set_ylabel("Earthquake rate / background")
    ax.set_title("Flare Duration: Does Longer Flare = Stronger/Later Response?")
    ax.legend()

    # Panel 2: Fast vs slow CME
    if len(flares_with_cme) > 20:
        ax = axes[1]
        ax.plot(hour_window, np.convolve(rate_fast/max(bg_fast,0.001), kernel, 'same'),
                color="#fc8d62", linewidth=2, label=f"Fast CME (>{med_speed:.0f} km/s)")
        ax.plot(hour_window, np.convolve(rate_slow/max(bg_slow,0.001), kernel, 'same'),
                color="#8da0cb", linewidth=2, label=f"Slow CME (<={med_speed:.0f} km/s)")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="orange", linewidth=2, alpha=0.5)
        ax.axvline(fast_arrival, color="#fc8d62", linestyle=":", alpha=0.7,
                   label=f"Fast shock: ~{fast_arrival:.0f}h")
        ax.axvline(slow_arrival, color="#8da0cb", linestyle=":", alpha=0.7,
                   label=f"Slow shock: ~{slow_arrival:.0f}h")
        ax.set_ylabel("Earthquake rate / background")
        ax.set_title("CME Speed: Does Faster CME = Earlier Seismic Response?")
        ax.legend()

    # Panel 3: X-class hourly detail
    ax = axes[2]
    ax.bar(hour_window, mean_x / max(bg_x, 0.001), width=0.8, color="steelblue", alpha=0.6)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(0, color="orange", linewidth=2, alpha=0.8, label="Flare")
    ax.axvline(18, color="purple", linestyle=":", linewidth=2, alpha=0.7, label="+18h")
    ax.set_xlabel("Hours relative to X-class solar flare")
    ax.set_ylabel("Earthquake rate / background")
    ax.set_title(f"X-Class Flares ({len(x_flares)} events): Hour-by-Hour Seismic Response")
    ax.legend()
    ax.set_xlim(-24, 72)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "grade4_signal.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'grade4_signal.png'}")


if __name__ == "__main__":
    main()
