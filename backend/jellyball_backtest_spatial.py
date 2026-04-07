"""
JellyBall Spatial Backtest — phase-resolved zone ratios from 26 years of data.

Computes the actual earthquake rate in each Jelly Ball zone as a function of
geomagnetic state (Kp, Bz, J-phase), then derives corrected zone ratios
for each phase of the storm cycle.

Data: 183K M4.5+ earthquakes (2000-2026) + OMNI2 hourly Kp/Bz/Dst/Vsw
"""
import math
import csv
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data"

# ─--Zone definitions (Paper XXV) ---
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

# Zone solid angle fractions (for normalization)
def zone_solid_angle(a1_deg, a2_deg):
    return (math.cos(math.radians(a1_deg)) - math.cos(math.radians(a2_deg))) / 2

ZONE_AREAS = {z[0]: zone_solid_angle(z[1], z[2]) for z in ZONES}
TOTAL_AREA = sum(ZONE_AREAS.values())

# ─--Subsolar point computation ---
def subsolar_point(dt):
    doy = dt.timetuple().tm_yday
    declination = 23.44 * math.sin(math.radians((360 / 365) * (doy - 81)))
    hour_frac = dt.hour + dt.minute / 60 + dt.second / 3600
    lon = (12 - hour_frac) * 15
    if lon > 180: lon -= 360
    if lon < -180: lon += 360
    return declination, lon

# ─--Angular distance ---
def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))

# ─--Assign zone ---
def assign_zone(ang_dist):
    for name, a1, a2, _ in ZONES:
        if ang_dist < a2:
            return name
    return "antipodal"

# ─--J stiffness from Kp (simplified model matching server.py) ---
def compute_j(kp, bz=0, vsw=400):
    # Simplified: J = sum of coupling terms normalized to [0, 1]
    # Solar: Kp-driven
    solar = min(1.0, kp / 9.0) * 0.4
    # Lunar: always present, ~0.1
    lunar = 0.1
    # Storm: from Kp proxy
    storm = min(1.0, max(0, (kp - 2)) / 7.0) * 0.5
    j = solar + lunar + storm
    return min(1.0, j)

J_C = 2 / math.pi  # 0.6366

# ─--Phase classification ---
def classify_phase(j, kp, prev_kp=None):
    gap_pct = (J_C - j) / J_C * 100
    if j > J_C:
        return "above_critical"
    elif gap_pct < 10:
        if prev_kp is not None and prev_kp > kp:
            return "relaxation"
        else:
            return "critical_transition"
    elif kp >= 5:
        return "compression"
    elif kp >= 3:
        return "unsettled"
    else:
        return "quiet"

# ─--Load OMNI2 data ---
print("Loading OMNI2 hourly data...")
omni = {}  # datetime_str -> {kp, bz, dst, vsw}
omni_file = OUTPUT_DIR / "omni2_hourly.csv"
with open(omni_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            dt_str = row['datetime'][:13]  # "2000-01-01 00"
            kp = float(row['kp']) if row['kp'] else None
            bz = float(row['bz']) if row['bz'] else None
            dst = float(row['dst']) if row['dst'] else None
            vsw = float(row['v_sw']) if row['v_sw'] else None
            if kp is not None:
                omni[dt_str] = {'kp': kp, 'bz': bz or 0, 'dst': dst or 0, 'vsw': vsw or 400}
        except (ValueError, KeyError):
            pass

print(f"  Loaded {len(omni)} hourly records")

# ─--Load earthquake data ---
print("Loading earthquake catalog...")
eq_file = OUTPUT_DIR / "earthquakes_m4.5_cache.csv"
earthquakes = []
with open(eq_file, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            dt = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            mag = float(row['mag'])
            if mag >= 4.5:
                earthquakes.append({'dt': dt, 'lat': lat, 'lon': lon, 'mag': mag})
        except (ValueError, KeyError):
            pass

print(f"  Loaded {len(earthquakes)} M4.5+ earthquakes")

# ─--Backtest: assign each earthquake to zone + phase ---
print("\nComputing zone assignments and phase states...")

# Phase bins
phase_zone_counts = defaultdict(lambda: defaultdict(int))
phase_totals = defaultdict(int)
kp_zone_counts = defaultdict(lambda: defaultdict(int))  # Kp bin -> zone -> count
kp_totals = defaultdict(int)

# Also track 6-month rolling window for baseline rate normalization
n_processed = 0
n_matched = 0
prev_kp = None

for eq in earthquakes:
    dt = eq['dt']
    # Find matching OMNI2 hour
    dt_str = dt.strftime('%Y-%m-%d %H')
    sw = omni.get(dt_str)
    if not sw:
        # Try nearest hour
        dt_str2 = (dt - timedelta(hours=1)).strftime('%Y-%m-%d %H')
        sw = omni.get(dt_str2)
    if not sw:
        continue

    n_matched += 1

    # Compute subsolar point
    ss_lat, ss_lon = subsolar_point(dt)

    # Angular distance from subsolar
    ang_dist = angular_distance(ss_lat, ss_lon, eq['lat'], eq['lon'])
    zone = assign_zone(ang_dist)

    # Compute J and phase
    j = compute_j(sw['kp'], sw['bz'], sw['vsw'])
    phase = classify_phase(j, sw['kp'], prev_kp)
    prev_kp = sw['kp']

    # Count
    phase_zone_counts[phase][zone] += 1
    phase_totals[phase] += 1

    # Kp bin
    kp_bin = f"Kp{int(sw['kp'])}"
    kp_zone_counts[kp_bin][zone] += 1
    kp_totals[kp_bin] += 1

    n_processed += 1
    if n_processed % 50000 == 0:
        print(f"  Processed {n_processed}...")

print(f"  Matched {n_matched} earthquakes to OMNI2 data")

# ─--Compute observed ratios ---
print("\n" + "=" * 80)
print("PHASE-RESOLVED ZONE RATIOS (observed / uniform expectation)")
print("=" * 80)

results = {}

for phase in sorted(phase_totals.keys()):
    total = phase_totals[phase]
    if total < 100:  # skip phases with too few events
        continue
    print(f"\n-- {phase.upper()} (n={total}) --")
    phase_results = {}
    for name, a1, a2, paper25_ratio in ZONES:
        count = phase_zone_counts[phase].get(name, 0)
        expected_frac = ZONE_AREAS[name] / TOTAL_AREA
        observed_frac = count / total if total > 0 else 0
        ratio = observed_frac / expected_frac if expected_frac > 0 else 0
        p25 = paper25_ratio
        diff = ratio - p25
        phase_results[name] = round(ratio, 3)
        print(f"  {name:<18} n={count:>6}  ratio={ratio:>6.3f}x  Paper25={p25:.2f}x  diff={diff:>+.3f}")
    results[phase] = phase_results

# ─--Kp-resolved ---
print("\n" + "=" * 80)
print("Kp-RESOLVED ZONE RATIOS")
print("=" * 80)

kp_results = {}
for kp_bin in sorted(kp_totals.keys()):
    total = kp_totals[kp_bin]
    if total < 50:
        continue
    print(f"\n--{kp_bin} (n={total}) ──")
    kp_res = {}
    for name, a1, a2, paper25_ratio in ZONES:
        count = kp_zone_counts[kp_bin].get(name, 0)
        expected_frac = ZONE_AREAS[name] / TOTAL_AREA
        observed_frac = count / total if total > 0 else 0
        ratio = observed_frac / expected_frac if expected_frac > 0 else 0
        kp_res[name] = round(ratio, 3)
        print(f"  {name:<18} n={count:>6}  ratio={ratio:>6.3f}x")
    kp_results[kp_bin] = kp_res

# ─--Overall (all conditions) ---
print("\n" + "=" * 80)
print("OVERALL ZONE RATIOS (all conditions, n={})".format(n_matched))
print("=" * 80)

overall = {}
for name, a1, a2, paper25_ratio in ZONES:
    count = sum(phase_zone_counts[p].get(name, 0) for p in phase_zone_counts)
    expected_frac = ZONE_AREAS[name] / TOTAL_AREA
    observed_frac = count / n_matched if n_matched > 0 else 0
    ratio = observed_frac / expected_frac if expected_frac > 0 else 0
    overall[name] = round(ratio, 3)
    sig = "*" if abs(ratio - 1.0) > 0.05 else " "
    print(f"  {name:<18} n={count:>6}  ratio={ratio:>6.3f}x  Paper25={paper25_ratio:.2f}x {sig}")

# ─--Save results ---
output = {
    "n_earthquakes": n_matched,
    "n_omni_hours": len(omni),
    "phase_resolved": results,
    "kp_resolved": kp_results,
    "overall": overall,
    "paper25_static": {z[0]: z[3] for z in ZONES},
}

out_file = OUTPUT_DIR / "jellyball_spatial_backtest.json"
with open(out_file, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_file}")

# ─--Summary ---
print("\n" + "=" * 80)
print("SUMMARY: Paper XXV accuracy by phase")
print("=" * 80)
for phase in sorted(results.keys()):
    res = results[phase]
    mse = sum((res[z] - ZONES[i][3])**2 for i, z in enumerate(res)) / len(res)
    direction_hits = sum(1 for z in res if (res[z] >= 1 and dict([(zz[0], zz[3]) for zz in ZONES])[z] >= 1) or (res[z] < 1 and dict([(zz[0], zz[3]) for zz in ZONES])[z] < 1))
    print(f"  {phase:<22} MSE={mse:.4f}  direction={direction_hits}/10")
