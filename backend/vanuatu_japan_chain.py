#!/usr/bin/env python3
"""
Vanuatu → Japan Chain Analysis
================================
Popular myth: big Vanuatu quakes precede big Japan quakes.
No statistical study has confirmed this.

Framework hypothesis: the coupling goes through the Pacific plate
boundary, but is MODULATED by the geomagnetic field geometry.
The stress wave from a Vanuatu rupture propagates along the
subduction zone, but the magnetic field creates preferred
channels (where B is strong = high J = ordered phase) and
barriers (where B is weak = low J = disordered phase).

Tests:
1. Raw: is there a Vanuatu → Japan temporal correlation?
2. Conditioned on geomagnetic state: does the chain only work
   during specific Kp/Dst conditions?
3. Path dependence: do Vanuatu quakes propagate preferentially
   along the magnetically strong side of the Pacific plate?

Also: account for the South Atlantic Anomaly (SAA) as a
region of permanently low J — a KT disordered zone that
should show anomalous seismicity patterns.
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

# ─── Region definitions ─────────────────────────────────────────────────────

REGIONS = {
    "Vanuatu": {"lat": (-22, -12), "lon": (164, 174)},
    "Japan":   {"lat": (28, 46),   "lon": (128, 148)},
    "Tonga":   {"lat": (-24, -15), "lon": (-178, -170)},
    "Philippines": {"lat": (4, 22), "lon": (118, 130)},
    "Indonesia":   {"lat": (-10, 6), "lon": (95, 135)},
    "Chile":   {"lat": (-45, -18), "lon": (-76, -66)},
    "Alaska":  {"lat": (50, 65),   "lon": (-175, -145)},
    # South Atlantic Anomaly center — geomagnetically weak zone
    "SAA":     {"lat": (-35, -15), "lon": (-55, -25)},
}

# Pacific Ring of Fire waypoints (Vanuatu → Japan path)
RING_WAYPOINTS = [
    ("Vanuatu", -16, 167),
    ("Solomon", -8, 157),
    ("PNG", -5, 150),
    ("Mariana", 15, 147),
    ("Izu-Bonin", 28, 142),
    ("Japan", 36, 140),
]

# ─── Data ────────────────────────────────────────────────────────────────────

def download_region_earthquakes(name, lat_range, lon_range, min_mag=5.0):
    """Download earthquakes for a specific region."""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "csv", "starttime": "2000-01-01", "endtime": "2026-03-31",
        "minlatitude": lat_range[0], "maxlatitude": lat_range[1],
        "minlongitude": lon_range[0], "maxlongitude": lon_range[1],
        "minmagnitude": min_mag, "orderby": "time-asc", "limit": 20000,
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df["time_parsed"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["magnitude"] = df["mag"]
    df["day_number"] = ((df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    return df


def download_kp():
    """Download daily Kp."""
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    records = []
    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 25:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 2000: continue
            kp_vals = [float(parts[7+i]) for i in range(8)]
            records.append({"year": y, "month": m, "day": d,
                           "kp_max": max(kp_vals), "kp_mean": np.mean(kp_vals),
                           "ap": float(parts[23])})
        except: continue
    df = pd.DataFrame(records)
    df["day_number"] = ((pd.to_datetime(df[["year","month","day"]]) - pd.Timestamp(INIT_DATE)).dt.days).values
    return df


# ─── Analysis 1: Raw chain test ─────────────────────────────────────────────

def chain_analysis(source_df, target_df, source_name, target_name, kp_df,
                   source_min_mag=6.5, window_days=14):
    """
    Test: after a large earthquake in source region,
    is there elevated seismicity in target region?
    """
    print(f"\n--- {source_name} (M>={source_min_mag}) -> {target_name} (window={window_days} days) ---")

    large_events = source_df[source_df["magnitude"] >= source_min_mag]
    print(f"  Large {source_name} events: {len(large_events)}")

    # For each large event, count target-region earthquakes in window
    triggered = []
    background = []

    for _, event in large_events.iterrows():
        d0 = event["day_number"]

        # Post-event window
        post = target_df[(target_df["day_number"] > d0) &
                         (target_df["day_number"] <= d0 + window_days)]
        triggered.append(len(post))

        # Background: same window length, 30-60 days before
        bg = target_df[(target_df["day_number"] >= d0 - 60) &
                       (target_df["day_number"] <= d0 - 30)]
        background.append(len(bg) * window_days / 30.0)  # normalize to same window

    triggered = np.array(triggered)
    background = np.array(background)

    mean_t = triggered.mean()
    mean_b = background.mean()
    ratio = mean_t / max(mean_b, 0.001)

    if len(triggered) > 5:
        _, p = stats.mannwhitneyu(triggered, background, alternative="greater")
    else:
        p = 1.0

    print(f"  Post-event mean:  {mean_t:.2f} quakes/{window_days}d")
    print(f"  Background mean:  {mean_b:.2f} quakes/{window_days}d")
    print(f"  Ratio: {ratio:.2f}x   p = {p:.4f}")

    return triggered, background, ratio, p


# ─── Analysis 2: Conditioned on geomagnetic state ───────────────────────────

def conditioned_chain(source_df, target_df, source_name, target_name, kp_df,
                      source_min_mag=6.5, window_days=14):
    """
    Same as chain analysis, but split by geomagnetic conditions
    at the time of the source earthquake.
    """
    print(f"\n--- Conditioned: {source_name} -> {target_name} by Kp state ---")

    large_events = source_df[source_df["magnitude"] >= source_min_mag].copy()

    # Merge with Kp
    large_events = pd.merge(large_events, kp_df[["day_number", "kp_max", "kp_mean"]],
                            on="day_number", how="left")

    # Split by Kp
    median_kp = large_events["kp_max"].median()
    high_kp = large_events[large_events["kp_max"] >= median_kp]
    low_kp = large_events[large_events["kp_max"] < median_kp]

    print(f"  Median Kp (max): {median_kp:.1f}")
    print(f"  High Kp events: {len(high_kp)}, Low Kp events: {len(low_kp)}")

    for label, subset in [("High Kp (active)", high_kp), ("Low Kp (quiet)", low_kp)]:
        if len(subset) < 3:
            print(f"  {label}: insufficient data")
            continue

        counts = []
        for _, event in subset.iterrows():
            d0 = event["day_number"]
            post = target_df[(target_df["day_number"] > d0) &
                             (target_df["day_number"] <= d0 + window_days)]
            counts.append(len(post))

        counts = np.array(counts)
        print(f"  {label}: mean {counts.mean():.2f} quakes/{window_days}d "
              f"(N={len(subset)}, max={counts.max()})")

    return high_kp, low_kp


# ─── Analysis 3: Superposed epoch — directional ─────────────────────────────

def directional_epoch(source_df, source_name, regions_dict, kp_df,
                      source_min_mag=6.5):
    """
    After large source events, track seismicity in ALL regions.
    This shows which direction the stress propagates.
    """
    print(f"\n=== Directional Epoch: After {source_name} M>={source_min_mag} ===")

    large_events = source_df[source_df["magnitude"] >= source_min_mag]
    print(f"  {len(large_events)} triggering events")

    window = np.arange(-14, 29)  # -14 to +28 days

    results = {}
    for rname, rdef in regions_dict.items():
        if rname == source_name:
            continue

        try:
            rdf = download_region_earthquakes(rname, rdef["lat"], rdef["lon"], min_mag=5.0)
        except:
            continue

        stacked = np.zeros((len(large_events), len(window)))
        for i, (_, ev) in enumerate(large_events.iterrows()):
            d0 = ev["day_number"]
            for j, offset in enumerate(window):
                day = d0 + offset
                n = len(rdf[rdf["day_number"] == day])
                stacked[i, j] = n

        mean_rate = np.mean(stacked, axis=0)
        bg = np.mean(mean_rate[:14])  # pre-event background
        post = np.mean(mean_rate[14:28])  # days 0-13 after
        ratio = post / max(bg, 0.001)

        results[rname] = {
            "mean_rate": mean_rate, "bg": bg, "post": post,
            "ratio": ratio, "n_events": len(rdf)
        }

        print(f"  -> {rname:>15s}: bg={bg:.3f}, post={post:.3f}, "
              f"ratio={ratio:.2f}x  ({len(rdf)} M5+ events)")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    for (rname, data), color in zip(results.items(), colors):
        ax.plot(window, data["mean_rate"], label=f"{rname} ({data['ratio']:.2f}x)",
                color=color, linewidth=1.5)

    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label=f"{source_name} M>={source_min_mag}")
    ax.set_xlabel(f"Days relative to {source_name} earthquake")
    ax.set_ylabel("Mean daily earthquake count (M>=5)")
    ax.set_title(f"Seismicity in Other Regions After {source_name} M>={source_min_mag}\n"
                 f"{len(large_events)} events, 2000-2026")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"directional_{source_name.lower()}.png", dpi=150)
    print(f"  Saved: {OUT_DIR / f'directional_{source_name.lower()}.png'}")

    return results


# ─── Analysis 4: SAA as KT disordered zone ──────────────────────────────────

def saa_analysis(kp_df):
    """
    The South Atlantic Anomaly: weakest geomagnetic field on Earth.
    In the framework: permanently low J, near or below J_c.
    This should mean:
    1. Seismicity in SAA region responds MORE to geomagnetic storms
       (already near critical, any perturbation crosses J_c)
    2. The SAA acts as a "leak" — stress waves from anywhere
       preferentially dissipate here
    """
    print("\n=== South Atlantic Anomaly: Permanent Low-J Zone ===")

    # SAA region: roughly centered on Brazil/South Atlantic
    # Compare with a magnetically strong region at similar latitude
    saa = download_region_earthquakes("SAA", (-35, -15), (-55, -25), min_mag=4.0)
    # Control: same latitude band but in magnetically strong Pacific
    control = download_region_earthquakes("Control", (-35, -15), (160, 180), min_mag=4.0)

    print(f"  SAA earthquakes (M>=4): {len(saa)}")
    print(f"  Control earthquakes (M>=4): {len(control)}")

    # Correlation with Kp at different lags
    saa_daily = saa.groupby("day_number").agg(n=("magnitude","count")).reset_index()
    ctrl_daily = control.groupby("day_number").agg(n=("magnitude","count")).reset_index()

    merged_saa = pd.merge(kp_df, saa_daily, on="day_number", how="left").fillna(0)
    merged_ctrl = pd.merge(kp_df, ctrl_daily, on="day_number", how="left").fillna(0)

    print(f"\n  Kp-seismicity correlation (lag +1 day):")
    for lag in [0, 1, 2, 3]:
        x = merged_saa["kp_max"].values[:-max(lag,1)]
        y_saa = merged_saa["n"].shift(-lag).values[:-max(lag,1)]
        y_ctrl = merged_ctrl["n"].shift(-lag).values[:-max(lag,1)]

        mask_s = ~(np.isnan(x) | np.isnan(y_saa))
        mask_c = ~(np.isnan(x) | np.isnan(y_ctrl))

        r_saa = np.corrcoef(x[mask_s], y_saa[mask_s])[0,1]
        r_ctrl = np.corrcoef(x[mask_c], y_ctrl[mask_c])[0,1]

        print(f"    Lag +{lag}: SAA r={r_saa:+.4f}, Control r={r_ctrl:+.4f}, "
              f"diff={r_saa-r_ctrl:+.4f}")

    # Storm response comparison
    storm_days = merged_saa[merged_saa["kp_max"] >= 5]["day_number"].values
    quiet_days = merged_saa[merged_saa["kp_max"] < 2]["day_number"].values

    def post_rate(daily_df, trigger_days, lag=1):
        rates = []
        for d in trigger_days:
            row = daily_df[daily_df["day_number"] == d + lag]
            rates.append(row["n"].values[0] if len(row) > 0 else 0)
        return np.array(rates)

    saa_storm = post_rate(merged_saa, storm_days)
    saa_quiet = post_rate(merged_saa, quiet_days)
    ctrl_storm = post_rate(merged_ctrl, storm_days)
    ctrl_quiet = post_rate(merged_ctrl, quiet_days)

    print(f"\n  Next-day seismicity after Kp>=5 storms vs quiet (Kp<2):")
    print(f"    SAA:     storm={saa_storm.mean():.3f}, quiet={saa_quiet.mean():.3f}, "
          f"ratio={saa_storm.mean()/max(saa_quiet.mean(),0.001):.2f}x")
    print(f"    Control: storm={ctrl_storm.mean():.3f}, quiet={ctrl_quiet.mean():.3f}, "
          f"ratio={ctrl_storm.mean()/max(ctrl_quiet.mean(),0.001):.2f}x")
    print(f"    Framework predicts SAA ratio > Control ratio (lower J = more sensitive)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VANUATU -> JAPAN CHAIN + DYNAMO EFFECTS")
    print("Does the geomagnetic field create preferred stress pathways?")
    print("=" * 70)

    # Download Kp
    print("\nDownloading Kp index...")
    kp_df = download_kp()
    print(f"  {len(kp_df)} daily Kp records")

    # Download regional catalogs
    print("\nDownloading regional earthquake catalogs...")
    region_dfs = {}
    for rname, rdef in REGIONS.items():
        try:
            df = download_region_earthquakes(rname, rdef["lat"], rdef["lon"], min_mag=5.0)
            region_dfs[rname] = df
            print(f"  {rname}: {len(df)} M5+ events")
        except Exception as e:
            print(f"  {rname}: failed ({e})")

    # Also download with lower mag for the chain test
    van_full = download_region_earthquakes("Vanuatu", (-22,-12), (164,174), min_mag=4.5)
    jpn_full = download_region_earthquakes("Japan", (28,46), (128,148), min_mag=4.5)

    # ─── Analysis 1: Raw chain ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Raw Temporal Chain")
    print("=" * 70)

    chain_analysis(van_full, jpn_full, "Vanuatu", "Japan", kp_df,
                   source_min_mag=6.5, window_days=14)
    chain_analysis(van_full, jpn_full, "Vanuatu", "Japan", kp_df,
                   source_min_mag=7.0, window_days=30)
    # Reverse direction
    chain_analysis(jpn_full, van_full, "Japan", "Vanuatu", kp_df,
                   source_min_mag=6.5, window_days=14)

    # ─── Analysis 2: Conditioned on Kp ───────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Chain Conditioned on Geomagnetic State")
    print("=" * 70)

    conditioned_chain(van_full, jpn_full, "Vanuatu", "Japan", kp_df,
                      source_min_mag=6.5, window_days=14)

    # ─── Analysis 3: Directional propagation ─────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Directional Stress Propagation")
    print("=" * 70)

    directional_epoch(region_dfs.get("Vanuatu", van_full), "Vanuatu",
                      REGIONS, kp_df, source_min_mag=6.5)

    # Also from Japan
    directional_epoch(region_dfs.get("Japan", jpn_full), "Japan",
                      REGIONS, kp_df, source_min_mag=6.5)

    # ─── Analysis 4: SAA ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS 4: South Atlantic Anomaly")
    print("=" * 70)

    saa_analysis(kp_df)

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FRAMEWORK INTERPRETATION")
    print("=" * 70)
    print("""
The geomagnetic field creates a SPATIALLY VARYING stiffness J(x):
  - Strong field (B large) -> high J -> ordered phase -> stable
  - Weak field (B small, e.g. SAA) -> low J -> near critical -> sensitive

Stress waves from earthquakes propagate through the crust but are
MODULATED by this J(x) landscape:
  - Along magnetically strong paths: wave propagates efficiently
    (ordered phase = good coupling between oscillators)
  - Through magnetically weak zones: wave is absorbed/scattered
    (disordered phase = poor coupling, energy dissipates as heat)

The Vanuatu -> Japan path goes ALONG the Pacific plate boundary
where subduction maintains strong coupling. But the path also
passes through regions of varying magnetic field strength.

The chain should work BETTER when:
  1. Geomagnetic field is quiet (Kp low) -> background J is stable
  2. The path is through magnetically strong crust
  3. No intervening weak zones absorb the wave

And WORSE when:
  1. Geomagnetic storms disrupt the field -> J fluctuates
  2. The path crosses magnetically weak zones
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
