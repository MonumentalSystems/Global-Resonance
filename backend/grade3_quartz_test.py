#!/usr/bin/env python3
"""
Grade-3 Quartz Chirality Test
================================
Test whether the solar-seismic correlation (Paper XXV) depends on
the chirality of the host rock, as predicted by the grade-3 analysis.

Approach: We don't have a global quartz chirality map (it doesn't exist!).
But we can use PROXIES:

1. LATITUDE PROXY: The grade-3 field {J, B}₃ has a strong latitude dependence
   (zero at magnetic equator, maximum at poles). If grade-3 modulates the
   coupling, the solar-seismic correlation should be STRONGER at high latitudes
   and WEAKER near the equator — independent of chirality.

2. INCLINATION PROXY: The grade-3 coupling for vertical telluric currents
   scales as sin(inclination). We can compute this for each earthquake's
   location and test whether high-inclination events show stronger solar
   correlation.

3. CONTINENTAL vs OCEANIC: Continental crust is quartz-rich (granitic).
   Oceanic crust is quartz-poor (basaltic). If quartz chirality matters,
   the correlation should be stronger for continental earthquakes.

4. GRANITE vs BASALT: Specifically, events in granitic terrain (quartz-bearing)
   should show different correlation than events in basaltic/ophiolitic terrain
   (quartz-absent). We can approximate this from depth + tectonic setting.

Method: Split the earthquake catalog by the proxy variable, compute the
solar-seismic correlation (Kp vs seismicity) for each subset, and test
whether the correlation differs between subsets.
"""

import numpy as np
import pandas as pd
import sys, os
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "earthquake-analysis" / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load earthquake and Kp data."""
    eq = pd.read_csv(DATA_DIR / "earthquakes_m4.5.csv", parse_dates=["time_parsed"])
    kp = pd.read_csv(DATA_DIR / "kp_daily.csv")
    kp["date"] = pd.to_datetime(kp[["year", "month", "day"]])
    print(f"Earthquakes: {len(eq)} M4.5+ events")
    print(f"Kp data: {len(kp)} daily records")
    return eq, kp


def magnetic_inclination(lat):
    """Approximate magnetic inclination from dipole model.
    tan(I) = 2 tan(λ) where λ is magnetic latitude.
    Simplified: use geographic latitude (≈ magnetic for most regions)."""
    return np.degrees(np.arctan(2 * np.tan(np.radians(lat))))


def grade3_coupling(lat):
    """Grade-3 coupling strength for vertical telluric currents.
    Proportional to sin(inclination) = component of B along vertical."""
    inc = magnetic_inclination(lat)
    return np.abs(np.sin(np.radians(inc)))


def compute_correlation(eq_subset, kp, label=""):
    """Compute daily earthquake rate vs Kp correlation for a subset."""
    # Daily earthquake count
    eq_daily = eq_subset.set_index("time_parsed").resample("D").size().rename("eq_count")
    eq_daily = eq_daily.reset_index()
    eq_daily.columns = ["date", "eq_count"]
    eq_daily["date"] = eq_daily["date"].dt.normalize()

    # Merge with Kp
    merged = pd.merge(eq_daily, kp[["date", "kp_mean"]], on="date", how="inner")
    merged = merged.dropna()

    if len(merged) < 30:
        return None, None, len(merged)

    from scipy import stats
    r, p = stats.pearsonr(merged["kp_mean"], merged["eq_count"])
    return r, p, len(merged)


def main():
    eq, kp = load_data()

    print("\n" + "=" * 70)
    print("GRADE-3 QUARTZ CHIRALITY TEST")
    print("Does the solar-seismic correlation depend on magnetic inclination?")
    print("=" * 70)

    # Add grade-3 coupling to each earthquake
    eq["inclination"] = magnetic_inclination(eq["latitude"])
    eq["g3_coupling"] = grade3_coupling(eq["latitude"])

    # TEST 1: Split by latitude bands (grade-3 coupling strength)
    print("\n\n=== TEST 1: Latitude bands (grade-3 coupling proxy) ===\n")
    bands = [
        ("Equatorial (0-15°)", (0, 15)),
        ("Subtropical (15-30°)", (15, 30)),
        ("Midlatitude (30-50°)", (30, 50)),
        ("High latitude (50-70°)", (50, 70)),
        ("Polar (70-90°)", (70, 90)),
    ]

    print(f"{'Band':35s} {'N events':>10s} {'g3_coupling':>12s} {'r(Kp,EQ)':>10s} {'p-value':>10s}")
    print("-" * 80)

    for name, (lat_lo, lat_hi) in bands:
        mask = (eq["latitude"].abs() >= lat_lo) & (eq["latitude"].abs() < lat_hi)
        subset = eq[mask]
        avg_g3 = subset["g3_coupling"].mean()
        r, p, n = compute_correlation(subset, kp, name)
        if r is not None:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{name:35s} {len(subset):10d} {avg_g3:12.3f} {r:+10.4f} {p:10.4f} {sig}")
        else:
            print(f"{name:35s} {len(subset):10d} {avg_g3:12.3f} {'N/A':>10s} {'N/A':>10s}")

    # TEST 2: Split by depth (shallow = crustal/quartz-bearing vs deep = mantle)
    print("\n\n=== TEST 2: Depth (crustal quartz proxy) ===\n")
    depth_bands = [
        ("Shallow (0-35 km, crustal)", (0, 35)),
        ("Intermediate (35-100 km)", (35, 100)),
        ("Deep (100-300 km, mantle)", (100, 300)),
        ("Very deep (300+ km)", (300, 700)),
    ]

    print(f"{'Band':35s} {'N events':>10s} {'Avg depth':>10s} {'r(Kp,EQ)':>10s} {'p-value':>10s}")
    print("-" * 80)

    for name, (d_lo, d_hi) in depth_bands:
        mask = (eq["depth"] >= d_lo) & (eq["depth"] < d_hi)
        subset = eq[mask]
        avg_d = subset["depth"].mean()
        r, p, n = compute_correlation(subset, kp, name)
        if r is not None:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{name:35s} {len(subset):10d} {avg_d:10.1f} {r:+10.4f} {p:10.4f} {sig}")
        else:
            print(f"{name:35s} {len(subset):10d} {avg_d:10.1f} {'N/A':>10s} {'N/A':>10s}")

    # TEST 3: Continental vs oceanic (quartz content proxy)
    print("\n\n=== TEST 3: Continental vs oceanic (quartz content) ===\n")
    print("Approximate: events within 200 km of plate boundaries at depth < 35 km")
    print("are likely in oceanic or transitional crust. Events far from boundaries")
    print("at shallow depth are continental (granitic, quartz-rich).\n")

    # Simple proxy: events at depth < 35 km AND |latitude| > 20° are more
    # likely continental. Events at depth < 35 km near trenches (specific regions)
    # are oceanic.
    # Oceanic regions: mid-ocean ridges, island arcs
    oceanic_mask = (
        (eq["depth"] < 35) &
        (
            ((eq["longitude"] > -50) & (eq["longitude"] < -10) & (eq["latitude"].abs() < 30)) |  # MAR
            ((eq["longitude"] > 120) & (eq["longitude"] < 180) & (eq["latitude"] > -10) & (eq["latitude"] < 30)) |  # Western Pacific arcs
            ((eq["longitude"] > -180) & (eq["longitude"] < -150) & (eq["latitude"] > 50))  # Aleutians
        )
    )
    continental_mask = (eq["depth"] < 35) & ~oceanic_mask & (eq["latitude"].abs() > 20)

    for name, mask in [("Continental (quartz-rich)", continental_mask),
                       ("Oceanic (quartz-poor)", oceanic_mask)]:
        subset = eq[mask]
        r, p, n = compute_correlation(subset, kp, name)
        if r is not None:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{name:35s} N={len(subset):6d} r={r:+.4f} p={p:.4f} {sig}")
        else:
            print(f"{name:35s} N={len(subset):6d} insufficient data")

    # TEST 4: Direct g3 coupling × Kp correlation
    print("\n\n=== TEST 4: Grade-3 coupling as continuous variable ===\n")

    # Bin earthquakes by g3_coupling and compute correlation per bin
    eq["g3_bin"] = pd.cut(eq["g3_coupling"], bins=5)
    print(f"{'g3_coupling bin':35s} {'N events':>10s} {'r(Kp,EQ)':>10s} {'p-value':>10s}")
    print("-" * 70)

    for g3_bin in sorted(eq["g3_bin"].unique()):
        mask = eq["g3_bin"] == g3_bin
        subset = eq[mask]
        r, p, n = compute_correlation(subset, kp)
        if r is not None:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{str(g3_bin):35s} {len(subset):10d} {r:+10.4f} {p:10.4f} {sig}")

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  INTERPRETATION                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  If the grade-3 channel modulates the solar-seismic coupling:║
    ║                                                              ║
    ║  TEST 1: High-latitude events should show STRONGER r(Kp,EQ) ║
    ║          than equatorial events (grade-3 coupling increases  ║
    ║          with latitude).                                     ║
    ║                                                              ║
    ║  TEST 2: Shallow (crustal) events should show STRONGER       ║
    ║          correlation than deep events (quartz is in the      ║
    ║          crust, not the mantle).                              ║
    ║                                                              ║
    ║  TEST 3: Continental events should show STRONGER correlation ║
    ║          than oceanic (continental crust has more quartz).    ║
    ║                                                              ║
    ║  TEST 4: r(Kp,EQ) should INCREASE with g3_coupling bin.     ║
    ║                                                              ║
    ║  All four tests probe the same prediction from different     ║
    ║  angles. Consistent results across all four would strongly   ║
    ║  support the grade-3 telluric mechanism.                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
