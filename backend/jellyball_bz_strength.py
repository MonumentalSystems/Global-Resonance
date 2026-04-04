#!/usr/bin/env python3
"""
Jelly Ball Hindcast — Split by Bz Polarity and Storm Strength

Tests whether the ringing bell model predictions sharpen when we split by:
1. Bz polarity at storm onset (northward = shield OFF, southward = shield ON)
2. Storm strength (Kp 5-6 moderate, Kp 7-8 strong, Kp 9 extreme)

Hypothesis: The l=2 sign flip and far-suppress loading should be
STRONGEST for northward Bz (compression transmits to crust) and
for the most intense storms (largest cavity excitation).
"""
import numpy as np
import pandas as pd
from scipy.special import legendre
from scipy.optimize import minimize
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta
import json, requests, sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
INIT_DATE = datetime(2000, 1, 1)

ZONES = [
    ("eye",            0,  15, 0.85),
    ("inner",         15,  30, 0.92),
    ("transition",    30,  60, 0.98),
    ("wavefront",     60,  75, 1.36),
    ("wavefront-tail",75, 100, 1.09),
    ("neutral",      100, 120, 0.95),
    ("far-suppress", 120, 135, 0.82),
    ("far-neutral",  135, 155, 0.90),
    ("pre-antipodal",155, 165, 1.00),
    ("antipodal",    165, 180, 1.16),
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
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

def solid_angle(d1, d2):
    return 2 * np.pi * abs(np.cos(np.radians(d1)) - np.cos(np.radians(d2)))


def download_omni():
    """Download hourly OMNI2 data (Bz, speed, Dst) from NASA SPDF."""
    cache = OUT / "omni2_hourly.csv"
    if cache.exists():
        print(f"Loading cached OMNI from {cache}")
        df = pd.read_csv(cache, parse_dates=["datetime"])
        return df

    print("Downloading OMNI2 hourly data from NASA...")
    url = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    records = []
    text = resp.text
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 55:
            continue
        try:
            year = int(parts[0])
            if year < 2000 or year > 2026:
                continue
            doy = int(parts[1])
            hour = int(parts[2])
            dt = datetime(year, 1, 1) + timedelta(days=doy-1, hours=hour)

            # OMNI2 columns (1-indexed in docs):
            # 17: Bz GSE (nT), fill=9999.9
            # 18: Bz GSM (nT), fill=9999.9
            # 24: V_sw (km/s), fill=99999
            # 25: Vx_gse, 26: Vy_gse, 27: Vz_gse
            # 40: Dst (nT), fill=99999
            # 38: Kp*10
            # OMNI2 columns (0-indexed, verified against Halloween 2003):
            # 15 = Bz GSM (nT), fill=999.9
            # 38 = Kp*10, fill=99
            # 40 = Dst (nT), fill=99999
            bz_gsm = float(parts[15])
            bz_gsm = bz_gsm if abs(bz_gsm) < 900 else np.nan
            v_sw = np.nan  # V_sw not in standard OMNI2 columns we need
            dst = float(parts[40])
            dst = dst if abs(dst) < 9000 else np.nan
            kp = float(parts[38]) / 10.0
            kp = kp if kp < 90 else np.nan

            records.append({
                "datetime": dt,
                "year": year,
                "bz": bz_gsm,
                "v_sw": v_sw,
                "dst": dst,
                "kp": kp,
                "day_number": (dt.date() - INIT_DATE.date()).days,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    df.to_csv(cache, index=False)
    print(f"  Got {len(df)} hourly records, saved to {cache}")
    return df


def load_earthquakes():
    cache = OUT / "earthquakes_m4.5_cache.csv"
    if not cache.exists():
        print("Run jellyball_backtest.py first")
        sys.exit(1)
    eq = pd.read_csv(cache)
    eq["time_parsed"] = pd.to_datetime(eq["time"], utc=True).dt.tz_localize(None)
    eq["day_number"] = ((eq["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    return eq


def find_storms_with_bz(omni_df, min_kp=5, min_gap_days=5):
    """Find storms and classify by Bz polarity at onset and peak Kp."""
    daily = omni_df.groupby("day_number").agg({
        "kp": "max", "bz": "mean", "dst": "min",
        "v_sw": "mean", "datetime": "first"
    }).reset_index()

    storm_days = daily[daily["kp"] >= min_kp].copy()
    storm_days = storm_days.sort_values("day_number")

    events = []
    last_day = -999
    for _, row in storm_days.iterrows():
        if row["day_number"] - last_day >= min_gap_days:
            events.append(row.to_dict())
            last_day = row["day_number"]
        elif row["kp"] > events[-1]["kp"]:
            events[-1] = row.to_dict()
            last_day = row["day_number"]

    df = pd.DataFrame(events)

    # Get Bz around each storm: mean Bz during day 0 and day -1
    for idx, event in df.iterrows():
        d = event["day_number"]
        onset_bz = omni_df[(omni_df["day_number"] >= d-1) & (omni_df["day_number"] <= d)]["bz"]
        df.loc[idx, "onset_bz"] = onset_bz.mean() if len(onset_bz) > 0 else np.nan
        peak_bz = omni_df[omni_df["day_number"] == d]["bz"]
        df.loc[idx, "peak_bz_min"] = peak_bz.min() if len(peak_bz) > 0 else np.nan

    return df


def compute_zone_ratios(eq_df, storms, phase_days):
    """Compute zone ratios for a set of storms in a given time window."""
    zone_bins = np.array([z[1] for z in ZONES] + [180])
    sa = np.array([solid_angle(z[1], z[2]) for z in ZONES])
    n_zones = len(ZONES)

    all_ratios = np.zeros((len(storms), n_zones))

    for idx, (_, storm) in enumerate(storms.iterrows()):
        d = storm["day_number"]
        dt = storm["datetime"]
        ss_lat, ss_lon = subsolar_point(dt)

        # Background
        bg = eq_df[(eq_df["day_number"] >= d-10) & (eq_df["day_number"] < d-5)]
        if len(bg) < 5:
            all_ratios[idx] = np.nan
            continue
        bg_dists = angular_distance(ss_lat, ss_lon, bg["latitude"].values, bg["longitude"].values)
        bg_counts, _ = np.histogram(bg_dists, bins=zone_bins)
        bg_density = bg_counts / 5 / sa
        bg_density[bg_density == 0] = 1e-10

        # Phase window
        d_start, d_end = phase_days
        phase = eq_df[(eq_df["day_number"] >= d+d_start) & (eq_df["day_number"] < d+d_end)]
        if len(phase) == 0:
            all_ratios[idx] = 0
            continue
        p_dists = angular_distance(ss_lat, ss_lon, phase["latitude"].values, phase["longitude"].values)
        p_counts, _ = np.histogram(p_dists, bins=zone_bins)
        p_density = p_counts / max(d_end - d_start, 1) / sa

        ratio = p_density / bg_density
        # Cap extreme ratios from low-count zones
        all_ratios[idx] = np.clip(ratio, 0, 10)

    return all_ratios


def main():
    print("=" * 80)
    print("  JELLY BALL — Bz Polarity + Storm Strength Analysis")
    print("=" * 80)

    eq_df = load_earthquakes()
    omni = download_omni()
    storms = find_storms_with_bz(omni, min_kp=5, min_gap_days=5)

    # Drop storms with no Bz data
    storms = storms.dropna(subset=["peak_bz_min"])
    print(f"\n{len(storms)} storms with Bz data")
    print(f"  Bz min range: {storms['peak_bz_min'].min():.1f} to {storms['peak_bz_min'].max():.1f} nT")

    zone_names = [z[0] for z in ZONES]
    far_idx = [i for i, z in enumerate(ZONES) if z[0] == "far-suppress"][0]
    wf_idx = [i for i, z in enumerate(ZONES) if z[0] == "wavefront"][0]

    # ==========================================
    # SPLIT BY Bz POLARITY (use minimum Bz during storm — captures southward excursions)
    # ==========================================
    # Strong southward: min Bz < -10 nT (deep reconnection, shield ON)
    # Weak/northward: min Bz > -5 nT (closed magneto, shield OFF)
    bz_south = storms[storms["peak_bz_min"] < -10]
    bz_north = storms[storms["peak_bz_min"] > -5]

    print(f"\n{'='*80}")
    print(f"  1. SPLIT BY Bz POLARITY")
    print(f"{'='*80}")
    print(f"  Northward Bz (shield OFF): {len(bz_north)} storms")
    print(f"  Southward Bz (shield ON):  {len(bz_south)} storms")

    for label, subset in [("Bz NORTH (shield OFF)", bz_north), ("Bz SOUTH (shield ON)", bz_south)]:
        if len(subset) < 5:
            print(f"\n  {label}: too few events ({len(subset)})")
            continue

        comp_ratios = compute_zone_ratios(eq_df, subset, (-1, 0))
        relax_ratios = compute_zone_ratios(eq_df, subset, (1, 5))

        # Mean across storms (ignore NaN)
        mean_comp = np.nanmean(comp_ratios, axis=0)
        mean_relax = np.nanmean(relax_ratios, axis=0)

        print(f"\n  --- {label} ({len(subset)} storms) ---")
        print(f"  {'Zone':18s} {'Compress':>8s} {'Relax':>8s} {'Shift':>8s}")
        for i, name in enumerate(zone_names):
            shift = mean_relax[i] - mean_comp[i]
            flag = " ***" if name == "far-suppress" else ""
            print(f"  {name:18s} {mean_comp[i]:8.2f}x {mean_relax[i]:8.2f}x {shift:+7.2f}{flag}")

        # Statistical test on far-suppress
        comp_far = comp_ratios[:, far_idx]
        relax_far = relax_ratios[:, far_idx]
        valid = ~np.isnan(comp_far) & ~np.isnan(relax_far) & (comp_far > 0) & (relax_far > 0)
        if valid.sum() > 5:
            t, p = stats.ttest_rel(relax_far[valid], comp_far[valid])
            print(f"  Far-suppress t-test: t={t:.2f}, p={p:.4f} (n={valid.sum()})")

    # ==========================================
    # SPLIT BY STORM STRENGTH
    # ==========================================
    print(f"\n{'='*80}")
    print(f"  2. SPLIT BY STORM STRENGTH (Kp)")
    print(f"{'='*80}")

    strength_bins = [
        ("Moderate (Kp 5-6)", storms[(storms["kp"] >= 5) & (storms["kp"] < 7)]),
        ("Strong (Kp 7-8)",   storms[(storms["kp"] >= 7) & (storms["kp"] < 9)]),
        ("Extreme (Kp 9)",    storms[storms["kp"] >= 9]),
    ]

    results_by_strength = {}
    for label, subset in strength_bins:
        if len(subset) < 5:
            print(f"\n  {label}: too few events ({len(subset)})")
            continue

        comp_ratios = compute_zone_ratios(eq_df, subset, (-1, 0))
        relax_ratios = compute_zone_ratios(eq_df, subset, (1, 5))

        mean_comp = np.nanmean(comp_ratios, axis=0)
        mean_relax = np.nanmean(relax_ratios, axis=0)

        print(f"\n  --- {label} ({len(subset)} storms) ---")
        print(f"  {'Zone':18s} {'Compress':>8s} {'Relax':>8s} {'Shift':>8s}")
        for i, name in enumerate(zone_names):
            shift = mean_relax[i] - mean_comp[i]
            flag = " ***" if name in ("far-suppress", "wavefront") else ""
            print(f"  {name:18s} {mean_comp[i]:8.2f}x {mean_relax[i]:8.2f}x {shift:+7.2f}{flag}")

        comp_far = comp_ratios[:, far_idx]
        relax_far = relax_ratios[:, far_idx]
        valid = ~np.isnan(comp_far) & ~np.isnan(relax_far) & (comp_far > 0) & (relax_far > 0)
        if valid.sum() > 5:
            t, p = stats.ttest_rel(relax_far[valid], comp_far[valid])
            print(f"  Far-suppress t-test: t={t:.2f}, p={p:.4f} (n={valid.sum()})")

        results_by_strength[label] = {
            "n": len(subset),
            "far_sup_comp": round(float(mean_comp[far_idx]), 3),
            "far_sup_relax": round(float(mean_relax[far_idx]), 3),
            "wavefront_comp": round(float(mean_comp[wf_idx]), 3),
            "wavefront_relax": round(float(mean_relax[wf_idx]), 3),
        }

    # ==========================================
    # COMBINED: Bz x Strength
    # ==========================================
    print(f"\n{'='*80}")
    print(f"  3. COMBINED: Bz POLARITY x STORM STRENGTH")
    print(f"{'='*80}")
    print(f"\n  Far-suppress ratio (compression -> relaxation):")
    print(f"  {'':20s} {'Bz North':>15s} {'Bz South':>15s}")

    for label, kp_min, kp_max in [("Moderate (5-6)", 5, 7), ("Strong (7-8)", 7, 9), ("Extreme (9)", 9, 10)]:
        for bz_label, bz_filter in [("Bz North", storms["onset_bz"] > 0), ("Bz South", storms["onset_bz"] <= 0)]:
            kp_filter = (storms["kp"] >= kp_min) & (storms["kp"] < kp_max)
            subset = storms[kp_filter & bz_filter]
            if len(subset) < 3:
                continue
            comp = compute_zone_ratios(eq_df, subset, (-1, 0))
            relax = compute_zone_ratios(eq_df, subset, (1, 5))
            mc = np.nanmean(comp[:, far_idx])
            mr = np.nanmean(relax[:, far_idx])
            if bz_label == "Bz North":
                north_str = f"{mc:.2f}->{mr:.2f} (n={len(subset)})"
            else:
                south_str = f"{mc:.2f}->{mr:.2f} (n={len(subset)})"
        try:
            print(f"  {label:20s} {north_str:>15s} {south_str:>15s}")
        except:
            pass

    # Save
    with open(OUT / "jellyball_bz_strength.json", "w") as f:
        json.dump(results_by_strength, f, indent=2)
    print(f"\nSaved: {OUT / 'jellyball_bz_strength.json'}")

    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
