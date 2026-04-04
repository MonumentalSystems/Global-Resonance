#!/usr/bin/env python3
"""
Jelly Ball Depth + Telluric Analysis

Tests predictions about HOW the electromagnetic coupling reaches faults:

1. DEPTH DEPENDENCE: EM diffusion time scales with depth
   - Shallow (0-35 km): fast coupling (< 10 min), pore fluid present
   - Mid (35-70 km): moderate coupling (< 30 min)
   - Deep (70-300 km): slow coupling (hours), no pore fluid below ~15 km
   - Very deep (300+ km): should show NO EM coupling (beyond diffusion range)

   Prediction: The far-suppress zone inversion should be STRONGEST for
   shallow earthquakes and ABSENT for deep earthquakes.

2. TELLURIC PORE PRESSURE: The mechanism is:
   CME -> ionospheric current -> telluric current (Jz) -> Lorentz force
   on pore fluid -> pore pressure change -> effective stress change
   -> fault friction modulation

   This only works where PORE FLUID EXISTS (< ~15 km in continental crust,
   deeper in subduction zones due to dehydration reactions).

   Prediction: The signal should correlate with crustal hydration:
   - Subduction zones (wet): stronger coupling
   - Continental interiors (dry): weaker coupling
   - Mid-ocean ridges (hydrothermal): moderate coupling

3. TIME DELAY: EM diffusion to depth d takes tau = sqrt(2*mu0*sigma*d^2)
   - 10 km: ~2.5 min
   - 30 km: ~15 min
   - 70 km: ~60 min

   Prediction: Deeper earthquakes should show the zone pattern shifted
   LATER in time relative to the storm onset.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta
import json, sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT = Path(__file__).parent / "output"
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

DEPTH_BINS = [
    ("Very shallow (0-15km)",    0,  15),
    ("Shallow crust (15-35km)", 15,  35),
    ("Mid-crust (35-70km)",     35,  70),
    ("Upper mantle (70-150km)", 70, 150),
    ("Deep (150-300km)",       150, 300),
    ("Very deep (300+km)",     300, 700),
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

def em_diffusion_time(depth_km, sigma=0.01):
    """EM diffusion time to depth (seconds). sigma in S/m."""
    import math
    mu0 = 4 * math.pi * 1e-7
    d = depth_km * 1000
    return math.sqrt(2 * mu0 * sigma * d * d)


def load_data():
    eq = pd.read_csv(OUT / "earthquakes_m4.5_cache.csv")
    eq["time_parsed"] = pd.to_datetime(eq["time"], utc=True).dt.tz_localize(None)
    eq["day_number"] = ((eq["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    return eq


def load_kp():
    import requests
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
    df = pd.DataFrame(records)
    return df


def find_storms(kp_df, min_kp=5, min_gap=5):
    high = kp_df[kp_df["kp"] >= min_kp].sort_values("datetime")
    events, last_day = [], -999
    for _, r in high.iterrows():
        d = r["day_number"]
        if d - last_day >= min_gap:
            events.append({"datetime": r["datetime"], "day_number": d, "peak_kp": r["kp"]})
            last_day = d
        elif r["kp"] > events[-1]["peak_kp"]:
            events[-1] = {"datetime": r["datetime"], "day_number": d, "peak_kp": r["kp"]}
            last_day = d
    return pd.DataFrame(events)


def compute_zone_ratio_by_depth(eq_df, storms, depth_min, depth_max, phase_days):
    """Compute far-suppress zone ratio for earthquakes in a depth range."""
    zone_bins = np.array([z[1] for z in ZONES] + [180])
    sa = np.array([solid_angle(z[1], z[2]) for z in ZONES])
    far_idx = [i for i, z in enumerate(ZONES) if z[0] == "far-suppress"][0]
    wf_idx = [i for i, z in enumerate(ZONES) if z[0] == "wavefront"][0]

    # Filter by depth
    depth_eq = eq_df[(eq_df["depth"] >= depth_min) & (eq_df["depth"] < depth_max)].copy()
    if len(depth_eq) < 100:
        return None

    comp_ratios = []
    relax_ratios = []

    for _, storm in storms.iterrows():
        d = storm["day_number"]
        ss_lat, ss_lon = subsolar_point(storm["datetime"])

        # Background
        bg = depth_eq[(depth_eq["day_number"] >= d-10) & (depth_eq["day_number"] < d-5)]
        if len(bg) < 3: continue
        bg_dists = angular_distance(ss_lat, ss_lon, bg["latitude"].values, bg["longitude"].values)
        bg_counts, _ = np.histogram(bg_dists, bins=zone_bins)
        bg_density = bg_counts / 5 / sa
        bg_density[bg_density == 0] = 1e-10

        # Compression
        d_start, d_end = -1, 0
        comp = depth_eq[(depth_eq["day_number"] >= d+d_start) & (depth_eq["day_number"] < d+d_end)]
        if len(comp) > 0:
            c_dists = angular_distance(ss_lat, ss_lon, comp["latitude"].values, comp["longitude"].values)
            c_counts, _ = np.histogram(c_dists, bins=zone_bins)
            c_density = c_counts / max(d_end-d_start, 1) / sa
            comp_ratios.append(np.clip(c_density / bg_density, 0, 10))

        # Relaxation
        d_start, d_end = 1, 5
        relax = depth_eq[(depth_eq["day_number"] >= d+d_start) & (depth_eq["day_number"] < d+d_end)]
        if len(relax) > 0:
            r_dists = angular_distance(ss_lat, ss_lon, relax["latitude"].values, relax["longitude"].values)
            r_counts, _ = np.histogram(r_dists, bins=zone_bins)
            r_density = r_counts / max(d_end-d_start, 1) / sa
            relax_ratios.append(np.clip(r_density / bg_density, 0, 10))

    if not comp_ratios or not relax_ratios:
        return None

    comp_arr = np.array(comp_ratios)
    relax_arr = np.array(relax_ratios)

    # Return far-suppress and wavefront stats
    return {
        "far_sup_comp": np.nanmean(comp_arr[:, far_idx]),
        "far_sup_relax": np.nanmean(relax_arr[:, far_idx]),
        "wavefront_comp": np.nanmean(comp_arr[:, wf_idx]),
        "wavefront_relax": np.nanmean(relax_arr[:, wf_idx]),
        "n_comp": len(comp_ratios),
        "n_relax": len(relax_ratios),
        "far_sup_comp_all": comp_arr[:, far_idx],
        "far_sup_relax_all": relax_arr[:, far_idx],
    }


def main():
    print("=" * 80)
    print("  JELLY BALL — Depth Dependence + Telluric Pore Pressure Test")
    print("=" * 80)

    eq_df = load_data()
    kp_df = load_kp()
    storms = find_storms(kp_df, min_kp=5, min_gap=5)
    print(f"\n{len(storms)} storms, {len(eq_df)} earthquakes")

    # EM diffusion times
    print(f"\n{'='*80}")
    print(f"  EM DIFFUSION TIMES (sigma=0.01 S/m)")
    print(f"{'='*80}")
    for label, d_min, d_max in DEPTH_BINS:
        d_mid = (d_min + d_max) / 2
        tau = em_diffusion_time(d_mid)
        print(f"  {label:30s}  tau = {tau/60:.1f} min  ({tau:.0f} s)")

    # Test each depth bin
    print(f"\n{'='*80}")
    print(f"  DEPTH-RESOLVED FAR-SUPPRESS ZONE RESPONSE")
    print(f"  (Compression vs Relaxation, 474 storms)")
    print(f"{'='*80}")
    print(f"\n  {'Depth bin':30s} {'N':>5s} {'Far-sup C':>9s} {'Far-sup R':>9s} {'Shift':>8s} {'p-value':>8s} {'Signal':>8s}")
    print("  " + "-" * 80)

    results = {}
    for label, d_min, d_max in DEPTH_BINS:
        r = compute_zone_ratio_by_depth(eq_df, storms, d_min, d_max, (-1, 0))
        if r is None:
            print(f"  {label:30s} {'too few events':>40s}")
            continue

        shift = r["far_sup_relax"] - r["far_sup_comp"]

        # T-test (independent, since storms may differ between phases)
        c_vals = r["far_sup_comp_all"][~np.isnan(r["far_sup_comp_all"]) & (r["far_sup_comp_all"] > 0)]
        r_vals = r["far_sup_relax_all"][~np.isnan(r["far_sup_relax_all"]) & (r["far_sup_relax_all"] > 0)]
        if len(c_vals) > 5 and len(r_vals) > 5:
            t, p = stats.ttest_ind(r_vals, c_vals)
        else:
            t, p = 0, 1

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        print(f"  {label:30s} {r['n_comp']:5d} {r['far_sup_comp']:8.2f}x {r['far_sup_relax']:8.2f}x {shift:+7.2f} {p:8.4f} {sig:>8s}")

        results[label] = {
            "depth_range": [d_min, d_max],
            "far_sup_comp": round(r["far_sup_comp"], 3),
            "far_sup_relax": round(r["far_sup_relax"], 3),
            "shift": round(shift, 3),
            "p_value": round(p, 6),
            "n": r["n_comp"],
            "wavefront_comp": round(r["wavefront_comp"], 3),
            "wavefront_relax": round(r["wavefront_relax"], 3),
        }

    # Wavefront analysis by depth
    print(f"\n{'='*80}")
    print(f"  DEPTH-RESOLVED WAVEFRONT ZONE RESPONSE")
    print(f"{'='*80}")
    print(f"\n  {'Depth bin':30s} {'WF Compress':>11s} {'WF Relax':>11s} {'Expected':>8s}")
    print("  " + "-" * 65)
    for label, data in results.items():
        print(f"  {label:30s} {data['wavefront_comp']:10.2f}x {data['wavefront_relax']:10.2f}x {'1.36x':>8s}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  INTERPRETATION")
    print(f"{'='*80}")

    depths = [(label, data) for label, data in results.items()]
    if len(depths) >= 3:
        shallow_shift = depths[0][1]["shift"] if depths[0][1]["shift"] else 0
        mid_shift = depths[2][1]["shift"] if len(depths) > 2 else 0
        deep_shift = depths[-1][1]["shift"] if depths[-1][1]["shift"] else 0

        print(f"""
  FAR-SUPPRESS ZONE SHIFT (relaxation - compression):
    Shallowest: {shallow_shift:+.2f}
    Mid-crust:  {mid_shift:+.2f}
    Deepest:    {deep_shift:+.2f}

  PORE FLUID HYPOTHESIS:
  If telluric currents modulate earthquakes through pore pressure,
  the signal should be STRONGEST where pore fluid exists:
    - Shallow crust (0-15 km): saturated, high porosity -> STRONG
    - Mid-crust (15-35 km): decreasing porosity -> MODERATE
    - Lower crust (35-70 km): dry (below brittle-ductile) -> WEAK
    - Deep (>70 km): only in subduction zones (dehydration) -> VARIABLE
    - Very deep (>300 km): no pore fluid -> NONE

  EM DIFFUSION HYPOTHESIS:
  If the coupling is through EM diffusion to depth:
    - All depths should show the effect (EM penetrates everywhere)
    - But with increasing TIME DELAY at greater depths
    - The compression/relaxation phases we test (day-scale) are
      much longer than EM diffusion times (minutes), so depth
      dependence should come from PORE FLUID, not diffusion lag
""")

    # Save
    with open(OUT / "jellyball_depth_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {OUT / 'jellyball_depth_analysis.json'}")


if __name__ == "__main__":
    main()
