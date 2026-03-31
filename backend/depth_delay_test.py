#!/usr/bin/env python3
"""
Test: does flare-earthquake delay scale with depth?
Does the wavefront zone show enrichment for rapid (<60 min) events?
"""
import sys, os, csv, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

DATA = Path(__file__).parent / "data"

# Load flares M1+
flares = []
with open(DATA / "solar_flares.csv", encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f):
        try:
            peak = datetime.fromisoformat(row["peakTime"].replace("Z", "+00:00"))
            cls = row.get("classType", "")
            if not cls:
                continue
            letter, num = cls[0], float(cls[1:]) if len(cls) > 1 else 1.0
            if letter == "M":
                flux = num * 1e-5
            elif letter == "X":
                flux = num * 1e-4
            else:
                continue
            flares.append({"peak": peak, "class": cls, "flux": flux})
        except Exception:
            pass
flares.sort(key=lambda x: x["peak"])
n_x = sum(1 for f in flares if f["class"].startswith("X"))
print(f"Flares: {len(flares)} (M: {len(flares)-n_x}, X: {n_x})")

# Load earthquakes M4.5+
eqs_all = []
with open(DATA / "earthquakes_m4.5.csv", encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f):
        try:
            t = datetime.fromisoformat(row["time_parsed"].replace("Z", "+00:00"))
            eqs_all.append({
                "time": t,
                "mag": float(row.get("mag", row.get("magnitude", "0"))),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "depth": float(row.get("depth", -1)),
            })
        except Exception:
            pass
eq_m5 = [e for e in eqs_all if e["mag"] >= 5.0]
print(f"Earthquakes: {len(eqs_all)} M4.5+, {len(eq_m5)} M5+")


def subsolar(dt):
    doy = dt.timetuple().tm_yday
    dec = 23.44 * math.sin(math.radians((360 / 365) * (doy - 81)))
    hf = dt.hour + dt.minute / 60 + dt.second / 3600
    lon = (12 - hf) * 15
    return dec, lon


def adist(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = [math.radians(x) for x in [lat1, lon1, lat2, lon2]]
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


# Build all pairs: M1+ flare -> M5+ eq within 6h
print("\nBuilding flare-earthquake pairs (<6h, M5+)...")
pairs = []
for fl in flares:
    ss_lat, ss_lon = subsolar(fl["peak"])
    for eq in eq_m5:
        delta_s = (eq["time"] - fl["peak"]).total_seconds()
        if delta_s < 0 or delta_s > 6 * 3600:
            continue
        dist = adist(ss_lat, ss_lon, eq["lat"], eq["lon"])
        pairs.append({
            "delta_min": delta_s / 60,
            "depth": eq["depth"],
            "ang_dist": dist,
            "eq_mag": eq["mag"],
            "flare_flux": fl["flux"],
            "flare_class": fl["class"],
            "eq_time": eq["time"],
            "fl_peak": fl["peak"],
        })

print(f"Total pairs: {len(pairs)}")
wf_pairs = [p for p in pairs if 60 <= p["ang_dist"] < 100]
print(f"Wavefront zone (60-100 deg): {len(wf_pairs)}")

# ===== DEPTH vs DELAY =====
wf_valid = [(p["depth"], p["delta_min"]) for p in wf_pairs if p["depth"] > 0]
if len(wf_valid) > 10:
    depths = np.array([x[0] for x in wf_valid])
    delays = np.array([x[1] for x in wf_valid])

    r, p_val = sp_stats.pearsonr(depths, delays)
    rho, p_rho = sp_stats.spearmanr(depths, delays)

    print(f"\n{'='*70}")
    print(f"DEPTH vs DELAY (wavefront zone, n={len(depths)})")
    print(f"{'='*70}")
    print(f"Pearson  r = {r:.4f},  p = {p_val:.2e}")
    print(f"Spearman rho = {rho:.4f},  p = {p_rho:.2e}")

    depth_bins = [
        (0, 20, "shallow 0-20km"),
        (20, 70, "upper 20-70km"),
        (70, 150, "intermediate 70-150km"),
        (150, 700, "deep 150-700km"),
    ]
    print(f"\nMedian delay by depth bin (wavefront):")
    for dlo, dhi, label in depth_bins:
        mask = (depths >= dlo) & (depths < dhi)
        n = mask.sum()
        if n > 0:
            med = np.median(delays[mask])
            q25, q75 = np.percentile(delays[mask], [25, 75])
            print(f"  {label:>25}: n={n:4d}  median={med:6.0f}min  IQR=[{q25:.0f}-{q75:.0f}]")

# ===== FLARE INTENSITY vs DELAY =====
if len(wf_valid) > 10:
    fluxes = np.array([p["flare_flux"] for p in wf_pairs if p["depth"] > 0])
    log_flux = np.log10(fluxes)
    r2, p2 = sp_stats.pearsonr(log_flux, delays)

    print(f"\n{'='*70}")
    print(f"FLARE INTENSITY vs DELAY (wavefront)")
    print(f"{'='*70}")
    print(f"Pearson r(log10(flux), delay) = {r2:.4f}, p = {p2:.2e}")

    print(f"\nMedian delay by flare class (wavefront):")
    for flo, fhi, label in [
        (1e-5, 5e-5, "M1-M4"),
        (5e-5, 1e-4, "M5-M9"),
        (1e-4, 1, "X-class"),
    ]:
        mask = (fluxes >= flo) & (fluxes < fhi)
        n = mask.sum()
        if n > 0:
            med = np.median(delays[mask])
            print(f"  {label:>10}: n={n:4d}  median={med:6.0f}min")

# ===== RAPID EVENTS BY ZONE =====
print(f"\n{'='*70}")
print(f"RAPID EVENTS (<60 min after M1+ flare, M5+ eq)")
print(f"{'='*70}")

rapid = [p for p in pairs if p["delta_min"] < 60]
print(f"Total rapid pairs: {len(rapid)}")

zone_bins = [
    (0, 30, "eye"),
    (30, 60, "inner"),
    (60, 100, "wavefront"),
    (100, 140, "outer"),
    (140, 180, "far+anti"),
]

print(f"\nBy zone (with solid-angle normalization):")
for zlo, zhi, zname in zone_bins:
    z_rapid = [p for p in rapid if zlo <= p["ang_dist"] < zhi]
    solid_frac = (math.cos(math.radians(zlo)) - math.cos(math.radians(zhi))) / 2
    expected = len(rapid) * solid_frac
    ratio = len(z_rapid) / max(0.1, expected)
    sig = " **" if ratio > 1.5 else " *" if ratio > 1.2 else ""
    print(f"  {zname:>10}: {len(z_rapid):4d}  expected={expected:5.0f}  ratio={ratio:.2f}x{sig}")

# Same for <30 min
rapid30 = [p for p in pairs if p["delta_min"] < 30]
print(f"\nUltra-rapid (<30 min): {len(rapid30)} pairs")
for zlo, zhi, zname in zone_bins:
    z_r = [p for p in rapid30 if zlo <= p["ang_dist"] < zhi]
    solid_frac = (math.cos(math.radians(zlo)) - math.cos(math.radians(zhi))) / 2
    expected = len(rapid30) * solid_frac
    ratio = len(z_r) / max(0.1, expected)
    sig = " **" if ratio > 1.5 else " *" if ratio > 1.2 else ""
    print(f"  {zname:>10}: {len(z_r):4d}  expected={expected:5.0f}  ratio={ratio:.2f}x{sig}")

# ===== BACKGROUND COMPARISON =====
print(f"\n{'='*70}")
print(f"BACKGROUND RATE TEST")
print(f"{'='*70}")

total_hours = (eqs_all[-1]["time"] - eqs_all[0]["time"]).total_seconds() / 3600
m5_per_hour = len(eq_m5) / total_hours
wf_solid = (math.cos(math.radians(60)) - math.cos(math.radians(100))) / 2

for window_min, label in [(60, "1 hour"), (30, "30 min"), (10, "10 min")]:
    window_h = window_min / 60
    expected = m5_per_hour * wf_solid * len(flares) * window_h
    observed = len([p for p in pairs if p["delta_min"] < window_min and 60 <= p["ang_dist"] < 100])
    enrichment = observed / max(0.1, expected)
    print(f"  {label:>8} wavefront: expected={expected:.1f}  observed={observed}  enrichment={enrichment:.2f}x")

# ===== NOTABLE RAPID WAVEFRONT EVENTS =====
print(f"\n{'='*70}")
print(f"ALL RAPID WAVEFRONT EVENTS (M5.5+, <60min, 60-100deg)")
print(f"{'='*70}")

notable = sorted(
    [p for p in rapid if 60 <= p["ang_dist"] < 100 and p["eq_mag"] >= 5.5],
    key=lambda x: x["delta_min"],
)
print(f"Count: {len(notable)}")
for p in notable:
    print(
        f"  M{p['eq_mag']:.1f} +{p['delta_min']:5.0f}min | {p['flare_class']:>5} | "
        f"{p['ang_dist']:.0f}deg | depth {p['depth']:.0f}km | "
        f"{p['eq_time'].strftime('%Y-%m-%d %H:%M')}"
    )

# ===== DEPTH vs DELAY SCATTER FOR ALL ZONES =====
print(f"\n{'='*70}")
print(f"DEPTH vs DELAY CORRELATION BY ZONE")
print(f"{'='*70}")
for zlo, zhi, zname in zone_bins:
    z_valid = [(p["depth"], p["delta_min"]) for p in pairs if zlo <= p["ang_dist"] < zhi and p["depth"] > 0]
    if len(z_valid) > 20:
        d = np.array([x[0] for x in z_valid])
        t = np.array([x[1] for x in z_valid])
        r, p = sp_stats.spearmanr(d, t)
        print(f"  {zname:>10}: n={len(z_valid):5d}  Spearman rho={r:+.4f}  p={p:.2e}")
