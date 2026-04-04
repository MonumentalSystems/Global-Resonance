#!/usr/bin/env python3
"""
Jelly Ball Backtest — Phase-Resolved Spatial Response

Tests the Paper XXV prediction that seismicity spatial pattern INVERTS
between compression (J > J_c) and relaxation (J dropping through J_c).

Compression phase (Kp rising, Dst dropping):
  - Subsolar: suppressed (0.85x)
  - Wavefront (60-75 deg): enhanced (1.36x)
  - Far-suppress (120-135 deg): suppressed (0.82x)

Relaxation phase (Kp falling, Dst recovering):
  - Far-suppress zone rebounds (stored strain release)
  - Wavefront weakens
  - Pattern inverts

This backtest separates geomagnetic impulses into COMPRESSION and
RELAXATION phases and tests whether the spatial response differs.
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
import json

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
INIT_DATE = dt.datetime(2000, 1, 1)

# Paper XXV zone definitions
ZONES = [
    ("eye",            0,  15, 0.85, "suppression"),
    ("inner",         15,  30, 0.92, "compression"),
    ("transition",    30,  60, 0.98, "near-neutral"),
    ("wavefront",     60,  75, 1.36, "PEAK"),
    ("wavefront-tail",75, 100, 1.09, "enhancement"),
    ("neutral",      100, 120, 0.95, "neutral"),
    ("far-suppress", 120, 135, 0.82, "suppression"),
    ("far-neutral",  135, 155, 0.90, "far neutral"),
    ("pre-antipodal",155, 165, 1.00, "neutral"),
    ("antipodal",    165, 180, 1.16, "enhancement"),
]


def subsolar_point(dt_utc):
    doy = dt_utc.timetuple().tm_yday
    decl = -23.44 * np.cos(np.radians(360 / 365 * (doy + 10)))
    hour = dt_utc.hour + dt_utc.minute / 60.0
    lon = 180.0 - 15.0 * hour
    if lon < -180: lon += 360
    if lon > 180: lon -= 360
    return decl, lon


def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def solid_angle(d1, d2):
    return 2 * np.pi * abs(np.cos(np.radians(d1)) - np.cos(np.radians(d2)))


def download_earthquakes(min_mag=4.5, cache_file=None):
    if cache_file and Path(cache_file).exists():
        print(f"Loading cached earthquakes from {cache_file}")
        return pd.read_csv(cache_file)

    print(f"Downloading global earthquakes M>={min_mag} (2000-2026)...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_dfs = []
    for year in range(2000, 2027):
        params = {
            "format": "csv", "starttime": f"{year}-01-01",
            "endtime": f"{year}-12-31", "minmagnitude": min_mag,
            "orderby": "time-asc", "limit": 20000,
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
    df["day_number"] = ((df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    if cache_file:
        df.to_csv(cache_file, index=False)
        print(f"  Cached to {cache_file}")
    print(f"  Total: {len(df)} earthquakes")
    return df


def download_kp():
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
    # Compute rolling Kp for phase classification
    df["kp_12h_avg"] = df["kp"].rolling(4, min_periods=1).mean()  # 4 slots = 12h
    df["kp_trend"] = df["kp_12h_avg"].diff(4).fillna(0)  # 12h trend
    print(f"  Got {len(df)} 3-hourly records")
    return df


def find_storm_events(kp_df, min_kp=5, min_gap_days=5):
    """
    Find geomagnetic storms (Kp >= min_kp) and classify each into
    COMPRESSION (Kp rising) and RELAXATION (Kp falling) phases.
    """
    storm_slots = kp_df[kp_df["kp"] >= min_kp].copy()
    if storm_slots.empty:
        return pd.DataFrame()

    # Group consecutive storm slots into events
    storm_slots["gap"] = storm_slots["day_number"].diff().fillna(99)
    events = []
    current_event = []

    for _, row in storm_slots.iterrows():
        if row["gap"] > min_gap_days or not current_event:
            if current_event:
                events.append(current_event)
            current_event = [row]
        else:
            current_event.append(row)
    if current_event:
        events.append(current_event)

    result = []
    for event_rows in events:
        edf = pd.DataFrame(event_rows)
        peak_idx = edf["kp"].idxmax()
        peak_row = edf.loc[peak_idx]

        # Get the full Kp context around the storm
        peak_day = peak_row["day_number"]
        peak_dt = peak_row["datetime"]
        peak_kp = peak_row["kp"]

        result.append({
            "day_number": peak_day,
            "datetime": peak_dt,
            "peak_kp": peak_kp,
            "duration_slots": len(edf),
        })

    return pd.DataFrame(result)


def phase_resolved_analysis(eq_df, storms_df, kp_df):
    """
    For each storm, compute seismicity in Paper XXV zones during:
    - COMPRESSION: day -1 to 0 (Kp rising)
    - PEAK: day 0 (storm maximum)
    - RELAXATION: day +1 to +5 (Kp falling, J dropping through J_c)
    - BACKGROUND: day -10 to -5 (pre-storm baseline)
    """
    zone_bins = np.array([z[1] for z in ZONES] + [180])
    zone_names = [z[0] for z in ZONES]
    zone_expected = np.array([z[3] for z in ZONES])
    n_zones = len(ZONES)

    phases = {
        "background": (-10, -5),
        "compression": (-1, 0),
        "peak": (0, 1),
        "relaxation_early": (1, 3),
        "relaxation_late": (3, 7),
    }

    results = {phase: np.zeros((len(storms_df), n_zones)) for phase in phases}

    print(f"\nProcessing {len(storms_df)} storms...")
    for idx, storm in storms_df.iterrows():
        if idx % 50 == 0:
            print(f"  Storm {idx}/{len(storms_df)} (Kp={storm['peak_kp']:.0f})", flush=True)

        storm_day = storm["day_number"]
        ss_lat, ss_lon = subsolar_point(storm["datetime"])

        for phase_name, (d_start, d_end) in phases.items():
            dur = d_end - d_start
            eq_window = eq_df[
                (eq_df["day_number"] >= storm_day + d_start) &
                (eq_df["day_number"] < storm_day + d_end)
            ]
            if len(eq_window) == 0:
                continue

            dists = angular_distance(ss_lat, ss_lon,
                                     eq_window["latitude"].values,
                                     eq_window["longitude"].values)
            counts, _ = np.histogram(dists, bins=zone_bins)
            results[phase_name][idx] = counts / max(dur, 1)

    # Compute densities normalized by solid angle
    solid_angles = np.array([solid_angle(z[1], z[2]) for z in ZONES])

    print("\n" + "=" * 90)
    print("  PHASE-RESOLVED JELLY BALL SPATIAL RESPONSE")
    print("  (Seismicity density ratio to background, by Paper XXV zone)")
    print("=" * 90)

    mean_bg = np.mean(results["background"], axis=0) / solid_angles
    mean_bg[mean_bg == 0] = 1e-10  # avoid division by zero

    phase_ratios = {}
    for phase_name in phases:
        mean_phase = np.mean(results[phase_name], axis=0) / solid_angles
        ratio = mean_phase / mean_bg
        phase_ratios[phase_name] = ratio

        if phase_name == "background":
            continue

        print(f"\n  --- {phase_name.upper()} ---")
        print(f"  {'Zone':18s} {'Observed':>8s} {'Expected':>8s} {'Match':>6s}")
        for i, (name, _, _, expected, _) in enumerate(ZONES):
            obs = ratio[i]
            match = "YES" if abs(obs - expected) < 0.3 else "no"
            flag = " ***" if abs(obs - expected) > 0.5 else ""
            print(f"  {name:18s} {obs:8.2f}x {expected:8.2f}x {match:>6s}{flag}")

    # Key test: does far-suppress INVERT between compression and relaxation?
    far_sup_idx = [i for i, z in enumerate(ZONES) if z[0] == "far-suppress"][0]
    wf_idx = [i for i, z in enumerate(ZONES) if z[0] == "wavefront"][0]

    print("\n" + "=" * 90)
    print("  KEY TEST: Zone Inversion (Compression vs Relaxation)")
    print("=" * 90)

    for zone_name, zone_idx in [("wavefront", wf_idx), ("far-suppress", far_sup_idx)]:
        print(f"\n  {zone_name} (expected: {ZONES[zone_idx][3]}x):")
        for phase in ["compression", "peak", "relaxation_early", "relaxation_late"]:
            r = phase_ratios[phase][zone_idx]
            bar = "#" * int(min(r, 3) * 15)
            print(f"    {phase:20s} {r:6.2f}x  {bar}")

    # Statistical test: is relaxation_late different from compression in far-suppress?
    comp_far = results["compression"][:, far_sup_idx] / solid_angles[far_sup_idx]
    relax_far = results["relaxation_late"][:, far_sup_idx] / solid_angles[far_sup_idx]
    # Remove zeros for valid comparison
    valid = (comp_far > 0) & (relax_far > 0)
    if valid.sum() > 10:
        t_stat, p_val = stats.ttest_rel(relax_far[valid], comp_far[valid])
        print(f"\n  Paired t-test (relaxation_late vs compression) for far-suppress:")
        print(f"    t = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"    Mean compression:  {np.mean(comp_far[valid]):.4f}")
        print(f"    Mean relaxation:   {np.mean(relax_far[valid]):.4f}")
        print(f"    Ratio (relax/comp): {np.mean(relax_far[valid]) / max(np.mean(comp_far[valid]), 1e-10):.2f}x")
        if p_val < 0.05:
            print(f"    ** SIGNIFICANT at p<0.05: relaxation differs from compression **")

    return phase_ratios, results


def plot_results(phase_ratios, storms_df):
    zone_names = [z[0] for z in ZONES]
    zone_expected = [z[3] for z in ZONES]
    x = np.arange(len(zone_names))
    w = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        "compression": "#ff6644",
        "peak": "#ff4444",
        "relaxation_early": "#44aaff",
        "relaxation_late": "#4444ff",
    }

    for i, (phase, color) in enumerate(colors.items()):
        vals = phase_ratios[phase]
        ax.bar(x + i * w - 1.5 * w, vals, w, alpha=0.8, color=color, label=phase)

    ax.plot(x, zone_expected, 'ko--', alpha=0.5, label='Paper XXV expected', markersize=5)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(zone_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Ratio to background")
    ax.set_title(f"Jelly Ball Phase-Resolved Spatial Response\n"
                 f"{len(storms_df)} storms (Kp>=5), M>=4.5 earthquakes, 2000-2026")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(2.5, max(max(v) for v in phase_ratios.values() if not np.isnan(v).all()) + 0.3))

    plt.tight_layout()
    outfile = OUT_DIR / "jellyball_phase_resolved.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved: {outfile}")


def main():
    print("=" * 70)
    print("JELLY BALL BACKTEST — Phase-Resolved Spatial Response")
    print("Testing compression vs relaxation zone inversion (Paper XXV)")
    print("=" * 70)

    cache = OUT_DIR / "earthquakes_m4.5_cache.csv"
    eq_df = download_earthquakes(min_mag=4.5, cache_file=str(cache))
    kp_df = download_kp()

    storms = find_storm_events(kp_df, min_kp=5, min_gap_days=5)
    print(f"\nFound {len(storms)} geomagnetic storms (Kp >= 5)")

    if storms.empty:
        print("No storms found. Check data.")
        return

    phase_ratios, raw_results = phase_resolved_analysis(eq_df, storms, kp_df)
    plot_results(phase_ratios, storms)

    # Save numerical results
    results_out = {}
    for phase, ratios in phase_ratios.items():
        results_out[phase] = {ZONES[i][0]: float(ratios[i]) for i in range(len(ZONES))}
    with open(OUT_DIR / "jellyball_phase_ratios.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"Saved: {OUT_DIR / 'jellyball_phase_ratios.json'}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
