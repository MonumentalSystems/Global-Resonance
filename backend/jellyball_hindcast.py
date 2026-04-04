#!/usr/bin/env python3
"""
Jelly Ball Hindcast — Event-by-Event Predictions for Major Storms

For each major geomagnetic storm (Kp >= 7), this script:
1. Computes the subsolar point at storm peak
2. Identifies which Paper XXV zones have active fault systems
3. Predicts which zones should show enhancement/suppression
4. Checks actual seismicity in each zone (compression vs relaxation)
5. Tests whether the l=2 sign flip occurs for individual events
6. Scores each prediction as HIT/MISS

Focuses on the most dramatic events: Halloween 2003, May 2024,
Bastille Day 2000, and the current April 2026 compound event.
"""
import numpy as np
import pandas as pd
from scipy.special import legendre
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
from io import StringIO
import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
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
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def solid_angle(d1, d2):
    return 2 * np.pi * abs(np.cos(np.radians(d1)) - np.cos(np.radians(d2)))


def classify_zone(ang_dist):
    for name, d1, d2, _ in ZONES:
        if d1 <= ang_dist < d2:
            return name
    return "antipodal"


def fit_legendre(ratios, n_modes=4):
    theta = np.array([z[1] + (z[2]-z[1])/2 for z in ZONES])
    cos_t = np.cos(np.radians(theta))
    def model(a):
        r = np.ones(len(theta))
        for l, c in enumerate(a, 1):
            r += c * legendre(l)(cos_t)
        return r
    def cost(a):
        return np.sum((model(a) - ratios)**2)
    res = minimize(cost, np.zeros(n_modes), method='Nelder-Mead')
    return res.x


def load_data():
    cache = OUT_DIR / "earthquakes_m4.5_cache.csv"
    if cache.exists():
        eq_df = pd.read_csv(cache)
        eq_df["time_parsed"] = pd.to_datetime(eq_df["time"], utc=True).dt.tz_localize(None)
    else:
        print("Run jellyball_backtest.py first to cache earthquake data")
        sys.exit(1)

    eq_df["day_number"] = ((eq_df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    return eq_df


def load_kp():
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url, timeout=60)
    records = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        if len(parts) < 25: continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 2000: continue
            for slot in range(8):
                records.append({
                    "datetime": datetime(y, m, d, slot*3),
                    "kp": float(parts[7+slot]),
                    "day_number": (datetime(y, m, d) - INIT_DATE).days,
                })
        except: pass
    return pd.DataFrame(records)


def find_major_storms(kp_df, min_kp=7, min_gap_days=5):
    """Find distinct storm events with Kp >= min_kp."""
    high = kp_df[kp_df["kp"] >= min_kp].copy()
    high = high.sort_values("datetime")

    events = []
    last_day = -999
    for _, row in high.iterrows():
        d = row["day_number"]
        if d - last_day >= min_gap_days:
            events.append({"datetime": row["datetime"], "day_number": d, "peak_kp": row["kp"]})
            last_day = d
        elif row["kp"] > events[-1]["peak_kp"]:
            events[-1] = {"datetime": row["datetime"], "day_number": d, "peak_kp": row["kp"]}
            last_day = d

    return pd.DataFrame(events)


def analyze_event(event, eq_df, label=""):
    """Full hindcast analysis for a single storm event."""
    peak_day = event["day_number"]
    peak_dt = event["datetime"]
    peak_kp = event["peak_kp"]
    ss_lat, ss_lon = subsolar_point(peak_dt)

    zone_bins = np.array([z[1] for z in ZONES] + [180])
    zone_names = [z[0] for z in ZONES]
    zone_expected = np.array([z[3] for z in ZONES])
    sa = np.array([solid_angle(z[1], z[2]) for z in ZONES])

    # Background: day -10 to -5
    bg = eq_df[(eq_df["day_number"] >= peak_day - 10) & (eq_df["day_number"] < peak_day - 5)]
    bg_dists = angular_distance(ss_lat, ss_lon, bg["latitude"].values, bg["longitude"].values) if len(bg) > 0 else np.array([])
    bg_counts, _ = np.histogram(bg_dists, bins=zone_bins) if len(bg_dists) > 0 else (np.zeros(len(ZONES)), None)
    bg_density = bg_counts / 5 / sa  # per day per steradian

    # Compression: day -1 to 0
    comp = eq_df[(eq_df["day_number"] >= peak_day - 1) & (eq_df["day_number"] < peak_day)]
    comp_dists = angular_distance(ss_lat, ss_lon, comp["latitude"].values, comp["longitude"].values) if len(comp) > 0 else np.array([])
    comp_counts, _ = np.histogram(comp_dists, bins=zone_bins) if len(comp_dists) > 0 else (np.zeros(len(ZONES)), None)
    comp_density = comp_counts / 1 / sa

    # Peak: day 0 to +1
    peak = eq_df[(eq_df["day_number"] >= peak_day) & (eq_df["day_number"] < peak_day + 1)]
    peak_dists = angular_distance(ss_lat, ss_lon, peak["latitude"].values, peak["longitude"].values) if len(peak) > 0 else np.array([])
    peak_counts, _ = np.histogram(peak_dists, bins=zone_bins) if len(peak_dists) > 0 else (np.zeros(len(ZONES)), None)
    peak_density = peak_counts / 1 / sa

    # Relaxation: day +1 to +5
    relax = eq_df[(eq_df["day_number"] >= peak_day + 1) & (eq_df["day_number"] < peak_day + 5)]
    relax_dists = angular_distance(ss_lat, ss_lon, relax["latitude"].values, relax["longitude"].values) if len(relax) > 0 else np.array([])
    relax_counts, _ = np.histogram(relax_dists, bins=zone_bins) if len(relax_dists) > 0 else (np.zeros(len(ZONES)), None)
    relax_density = relax_counts / 4 / sa

    bg_density[bg_density == 0] = 1e-10

    comp_ratio = comp_density / bg_density
    peak_ratio = peak_density / bg_density
    relax_ratio = relax_density / bg_density

    # Legendre decomposition
    a_comp = fit_legendre(comp_ratio)
    a_peak = fit_legendre(peak_ratio)
    a_relax = fit_legendre(relax_ratio)

    # l=2 sign flip test
    l2_flip = a_comp[1] * a_relax[1] < 0
    l2_comp_positive = a_comp[1] > 0

    # Zone-level predictions
    far_sup_idx = [i for i, z in enumerate(ZONES) if z[0] == "far-suppress"][0]
    wf_idx = [i for i, z in enumerate(ZONES) if z[0] == "wavefront"][0]

    # Predictions from ringing bell model:
    predictions = {
        "l2_flips": l2_flip,
        "far_sup_enhanced_compression": comp_ratio[far_sup_idx] > 1.0,
        "far_sup_decays_relaxation": relax_ratio[far_sup_idx] < comp_ratio[far_sup_idx],
        "eye_suppressed_peak": peak_ratio[0] < 1.0,
        "wavefront_tail_delayed": relax_ratio[wf_idx + 1] > comp_ratio[wf_idx + 1],
    }

    hits = sum(predictions.values())
    total = len(predictions)

    return {
        "label": label,
        "date": str(peak_dt.date()),
        "peak_kp": peak_kp,
        "subsolar": {"lat": round(ss_lat, 1), "lon": round(ss_lon, 1)},
        "eq_counts": {"bg": int(bg_counts.sum()), "comp": int(comp_counts.sum()),
                      "peak": int(peak_counts.sum()), "relax": int(relax_counts.sum())},
        "l2_coefficients": {"compression": round(a_comp[1], 4), "peak": round(a_peak[1], 4),
                            "relaxation": round(a_relax[1], 4)},
        "l2_flips_sign": l2_flip,
        "far_suppress_ratio": {"compression": round(comp_ratio[far_sup_idx], 2),
                               "peak": round(peak_ratio[far_sup_idx], 2),
                               "relaxation": round(relax_ratio[far_sup_idx], 2)},
        "wavefront_ratio": {"compression": round(comp_ratio[wf_idx], 2),
                            "peak": round(peak_ratio[wf_idx], 2),
                            "relaxation": round(relax_ratio[wf_idx], 2)},
        "predictions": predictions,
        "score": f"{hits}/{total}",
    }


def main():
    print("=" * 80)
    print("  JELLY BALL HINDCAST — Event-by-Event Predictions")
    print("  Testing the ringing bell model on major geomagnetic storms")
    print("=" * 80)

    eq_df = load_data()
    kp_df = load_kp()
    storms = find_major_storms(kp_df, min_kp=7, min_gap_days=5)
    print(f"\nFound {len(storms)} major storms (Kp >= 7)")

    # Notable events to label
    labels = {
        "2000-04-07": "Bastille Day precursor",
        "2000-07-15": "Bastille Day Storm",
        "2001-03-31": "March 2001 Storm",
        "2001-11-06": "Nov 2001 Storm",
        "2003-10-29": "Halloween Storm (1)",
        "2003-10-30": "Halloween Storm (2)",
        "2003-11-20": "Nov 2003 Storm",
        "2004-07-27": "Jul 2004 Storm",
        "2004-11-08": "Nov 2004 Storm (1)",
        "2004-11-09": "Nov 2004 Storm (2)",
        "2005-08-24": "Aug 2005 Storm",
        "2024-05-10": "May 2024 Superstorm (1)",
        "2024-05-11": "May 2024 Superstorm (2)",
        "2024-10-10": "Oct 2024 Storm",
        "2025-11-12": "Nov 2025 Storm",
        "2026-01-19": "Jan 2026 Storm",
    }

    all_results = []
    total_hits = 0
    total_preds = 0
    l2_flip_count = 0

    for idx, storm in storms.iterrows():
        date_str = str(storm["datetime"].date())
        label = labels.get(date_str, "")
        result = analyze_event(storm, eq_df, label)
        all_results.append(result)

        hits = sum(result["predictions"].values())
        total = len(result["predictions"])
        total_hits += hits
        total_preds += total
        if result["l2_flips_sign"]:
            l2_flip_count += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"  RESULTS: {len(all_results)} storms analyzed")
    print(f"{'='*80}")
    print(f"\n  Overall prediction accuracy: {total_hits}/{total_preds} ({total_hits/total_preds*100:.1f}%)")
    print(f"  l=2 sign flip rate: {l2_flip_count}/{len(all_results)} ({l2_flip_count/len(all_results)*100:.1f}%)")

    print(f"\n  {'Date':12s} {'Kp':>3s} {'Score':>6s} {'l2 flip':>8s} {'Far-sup C':>9s} {'Far-sup R':>9s} {'Event':>25s}")
    print("  " + "-" * 85)

    for r in all_results:
        l2 = "YES" if r["l2_flips_sign"] else "no"
        fc = r["far_suppress_ratio"]["compression"]
        fr = r["far_suppress_ratio"]["relaxation"]
        flag = " ***" if r["l2_flips_sign"] and fc > 1.0 else ""
        print(f"  {r['date']:12s} {r['peak_kp']:3.0f} {r['score']:>6s} {l2:>8s} {fc:8.2f}x {fr:8.2f}x {r['label']:>25s}{flag}")

    # Best and worst events
    print(f"\n  NOTABLE EVENTS:")
    for r in all_results:
        if r["label"]:
            fc = r["far_suppress_ratio"]["compression"]
            fr = r["far_suppress_ratio"]["relaxation"]
            l2c = r["l2_coefficients"]["compression"]
            l2r = r["l2_coefficients"]["relaxation"]
            print(f"\n  {r['label']} ({r['date']}, Kp={r['peak_kp']:.0f})")
            print(f"    Subsolar: {r['subsolar']['lat']}N, {r['subsolar']['lon']}E")
            print(f"    EQs: bg={r['eq_counts']['bg']}, comp={r['eq_counts']['comp']}, "
                  f"peak={r['eq_counts']['peak']}, relax={r['eq_counts']['relax']}")
            print(f"    l=2: compression={l2c:+.4f} -> relaxation={l2r:+.4f}  flip={'YES' if r['l2_flips_sign'] else 'no'}")
            print(f"    Far-suppress: {fc:.2f}x (comp) -> {fr:.2f}x (relax)")
            print(f"    Score: {r['score']}")

    # Save
    with open(OUT_DIR / "jellyball_hindcast.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'jellyball_hindcast.json'}")

    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
