#!/usr/bin/env python3
"""
Foreshock Grade-Arrival Test + Eclipse-Seismicity
===================================================
1. FORESHOCK TEST: Before major earthquakes, do we see a grade-arrival
   sequence? Grade-0 (EM/ionospheric) should arrive BEFORE grade-2
   (mechanical rupture). Test with ionospheric TEC anomalies + local
   micro-seismicity timing.

2. ANTIPODAL TEST: At the antipodal point of a major earthquake,
   is there micro-seismicity enhancement that PRECEDES the main shock
   (grade-0 wave arriving at c) and/or FOLLOWS it (grade-2 at
   seismic wave speed)?

3. ECLIPSE TEST: The April 8, 2024 solar eclipse crossed the New
   Madrid Seismic Zone. An eclipse rapidly changes atmospheric
   conductivity along a narrow strip. Did seismicity change?
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"


def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def antipodal_point(lat, lon):
    """Return the antipodal point."""
    return -lat, (lon + 180) % 360 - 180


# ═══════════════════════════════════════════════════════════════════════
# 1. ANTIPODAL FORESHOCK TEST
# ═══════════════════════════════════════════════════════════════════════

def antipodal_test(eq):
    """
    For each M7.5+ earthquake:
    1. Find the antipodal point
    2. Check micro-seismicity (M4+) within 20° of the antipode
    3. Look at HOURLY rate in the 48h before and after the main shock
    4. Is there a pre-shock enhancement at the antipode (grade-0 arrival)?
    """
    print("\n=== Antipodal Foreshock/Aftershock Test ===")

    large = eq[eq["magnitude"] >= 7.5].copy()
    print(f"M7.5+ earthquakes: {len(large)}")

    eq_times = eq["time_parsed"].values.astype("datetime64[h]")
    eq_lats = eq["latitude"].values
    eq_lons = eq["longitude"].values

    hour_window = np.arange(-48, 49)
    antipodal_rates = np.zeros((len(large), len(hour_window)))
    local_rates = np.zeros((len(large), len(hour_window)))

    for i, (_, ev) in enumerate(large.iterrows()):
        t0 = np.datetime64(ev["time_parsed"], "h")
        elat, elon = ev["latitude"], ev["longitude"]
        alat, alon = antipodal_point(elat, elon)

        for j, h in enumerate(hour_window):
            t_start = t0 + np.timedelta64(h, "h")
            t_end = t0 + np.timedelta64(h + 1, "h")
            mask = (eq_times >= t_start) & (eq_times < t_end)

            if np.any(mask):
                # Antipodal: within 20° of antipode
                dists_anti = angular_distance(alat, alon, eq_lats[mask], eq_lons[mask])
                antipodal_rates[i, j] = np.sum(dists_anti < 20)

                # Local: within 20° of epicenter (for comparison)
                dists_local = angular_distance(elat, elon, eq_lats[mask], eq_lons[mask])
                local_rates[i, j] = np.sum(dists_local < 20)

    mean_anti = np.mean(antipodal_rates, axis=0)
    mean_local = np.mean(local_rates, axis=0)

    bg_anti = np.mean(mean_anti[:24])  # -48 to -24h background
    bg_local = np.mean(mean_local[:24])

    print(f"\nAntipodal background rate: {bg_anti:.3f}/hr")
    print(f"Local background rate: {bg_local:.3f}/hr")

    print(f"\nAntipodal micro-seismicity around M7.5+ main shocks:")
    print(f"{'Hour':>6s} {'Antipodal':>10s} {'Ratio':>8s} {'Local':>10s} {'Ratio':>8s}")
    for h_idx in [20, 22, 24, 42, 44, 46, 47, 48, 49, 50, 52, 54, 60, 66, 72]:
        if h_idx < len(hour_window):
            h = hour_window[h_idx]
            a_rate = mean_anti[h_idx]
            a_ratio = a_rate / max(bg_anti, 0.001)
            l_rate = mean_local[h_idx]
            l_ratio = l_rate / max(bg_local, 0.001)
            marker = ""
            if h == 0: marker = " <-- MAIN SHOCK"
            if h == -2: marker = " <-- 2h before"
            if h == -1: marker = " <-- 1h before"
            if h == 1: marker = " <-- 1h after"
            print(f"  {h:+4d}h  {a_rate:>9.3f}  {a_ratio:>7.2f}x  {l_rate:>9.3f}  {l_ratio:>7.2f}x{marker}")

    # Key test: is there an antipodal PRECURSOR?
    # Grade-0 at c: arrives in < 1 second (essentially instant)
    # Grade-2 seismic: P-wave at 8 km/s across Earth diameter 12742 km = ~26 min
    # Surface wave: ~3.5 km/s = ~60 min for 12742 km
    pre_1h = mean_anti[47]  # -1 hour
    pre_6h = np.mean(mean_anti[42:48])  # -6h to 0
    post_1h = mean_anti[49]  # +1 hour
    post_6h = np.mean(mean_anti[48:54])  # 0 to +6h

    print(f"\n  Antipodal rate 1h BEFORE main shock: {pre_1h:.3f} ({pre_1h/max(bg_anti,0.001):.2f}x bg)")
    print(f"  Antipodal rate 6h before: {pre_6h:.3f} ({pre_6h/max(bg_anti,0.001):.2f}x bg)")
    print(f"  Antipodal rate 1h AFTER: {post_1h:.3f} ({post_1h/max(bg_anti,0.001):.2f}x bg)")
    print(f"  Antipodal rate 6h after: {post_6h:.3f} ({post_6h/max(bg_anti,0.001):.2f}x bg)")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    kernel = np.ones(3) / 3
    ax.plot(hour_window, np.convolve(mean_anti, kernel, "same"), "o-",
            color="steelblue", lw=2, markersize=3, label="Antipodal (within 20deg)")
    ax.axhline(bg_anti, color="red", linestyle="--", alpha=0.5, label=f"Background: {bg_anti:.3f}")
    ax.axvline(0, color="orange", lw=2, alpha=0.7, label="Main shock")
    ax.set_ylabel("Mean M4.5+ quakes/hour")
    ax.set_title(f"Antipodal Micro-Seismicity Around {len(large)} M7.5+ Earthquakes\n"
                 f"Is there a precursor signal at the antipode?")
    ax.legend()

    ax = axes[1]
    ax.plot(hour_window, np.convolve(mean_local, kernel, "same"), "o-",
            color="#e41a1c", lw=2, markersize=3, label="Local (within 20deg)")
    ax.axhline(bg_local, color="red", linestyle="--", alpha=0.5)
    ax.axvline(0, color="orange", lw=2, alpha=0.7, label="Main shock")
    ax.set_ylabel("Mean M4.5+ quakes/hour")
    ax.set_xlabel("Hours relative to M7.5+ main shock")
    ax.set_title("Local Micro-Seismicity (foreshocks + aftershocks)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "antipodal_foreshock.png", dpi=150)
    print(f"\nSaved: antipodal_foreshock.png")

    return mean_anti, mean_local, bg_anti


# ═══════════════════════════════════════════════════════════════════════
# 2. ECLIPSE + NEW MADRID TEST
# ═══════════════════════════════════════════════════════════════════════

def eclipse_test(eq):
    """
    Test: does seismicity change along eclipse paths?
    Focus on the April 8, 2024 total solar eclipse which crossed
    the New Madrid Seismic Zone (NMSZ).

    Eclipse path: Texas -> Arkansas -> Missouri -> Illinois -> Indiana ->
    Ohio -> New York -> Vermont -> Maine

    NMSZ: approximately 36N, 89.5W (Memphis/New Madrid area)
    """
    print("\n=== Eclipse + New Madrid Seismic Zone ===")

    # Eclipse dates and paths
    eclipses = [
        {"date": "2024-04-08", "name": "April 2024 Total Solar Eclipse",
         "path_lat": [25.5, 30, 33, 36, 39, 42, 44, 46],
         "path_lon": [-100, -95, -92, -89.5, -84, -78, -73, -69],
         "nmsz": True},
        {"date": "2017-08-21", "name": "August 2017 Great American Eclipse",
         "path_lat": [33, 35, 37, 39, 41, 44],
         "path_lon": [-118, -112, -104, -96, -87, -80],
         "nmsz": False},  # Crossed further west
    ]

    for ecl in eclipses:
        print(f"\n  {ecl['name']} ({ecl['date']})")

        ecl_date = pd.Timestamp(ecl["date"])

        # NMSZ region: 35-37.5N, 88-91W
        nmsz_eq = eq[(eq["latitude"] > 35) & (eq["latitude"] < 37.5) &
                      (eq["longitude"] > -91) & (eq["longitude"] < -88)]

        # Activity around eclipse date
        for window_name, days_before, days_after in [
            ("2 weeks before", -14, 0),
            ("Eclipse week", -3, 3),
            ("2 weeks after", 0, 14),
            ("Month before", -30, 0),
            ("Month after", 0, 30),
        ]:
            subset = nmsz_eq[(nmsz_eq["time_parsed"] >= ecl_date + pd.Timedelta(days=days_before)) &
                              (nmsz_eq["time_parsed"] <= ecl_date + pd.Timedelta(days=days_after))]
            print(f"    {window_name}: {len(subset)} events in NMSZ")

        # Compare to same period in prior years
        print(f"    Historical comparison (same 2 weeks in prior years):")
        for yr_offset in [-3, -2, -1, 1]:
            comp_date = ecl_date + pd.DateOffset(years=yr_offset)
            comp_eq = nmsz_eq[(nmsz_eq["time_parsed"] >= comp_date - pd.Timedelta(days=14)) &
                               (nmsz_eq["time_parsed"] <= comp_date + pd.Timedelta(days=14))]
            print(f"      {comp_date.year}: {len(comp_eq)} events")

        # Along the eclipse path: check seismicity within 100 km of path
        print(f"\n    Seismicity along eclipse path (within 1 deg):")
        path_eq_before = 0
        path_eq_after = 0
        for plat, plon in zip(ecl["path_lat"], ecl["path_lon"]):
            near = eq[(eq["latitude"] > plat - 1) & (eq["latitude"] < plat + 1) &
                       (eq["longitude"] > plon - 1) & (eq["longitude"] < plon + 1)]
            before = near[(near["time_parsed"] >= ecl_date - pd.Timedelta(days=7)) &
                           (near["time_parsed"] < ecl_date)]
            after = near[(near["time_parsed"] > ecl_date) &
                          (near["time_parsed"] <= ecl_date + pd.Timedelta(days=7))]
            path_eq_before += len(before)
            path_eq_after += len(after)

        print(f"      7 days before eclipse: {path_eq_before} events along path")
        print(f"      7 days after eclipse:  {path_eq_after} events along path")
        if path_eq_before > 0:
            ratio = path_eq_after / path_eq_before
            print(f"      After/before ratio: {ratio:.2f}x")

    # Broader test: all eclipses since 2000 (there are many partial/annular too)
    print("\n  Note: A comprehensive eclipse-seismicity test would need:")
    print("    - Full eclipse catalog (NASA Five Millennium Canon)")
    print("    - Test ALL eclipses, not just 2024")
    print("    - Control for seasonal seismicity variation")
    print("    - The mechanism: eclipse shadow rapidly changes atmospheric")
    print("      conductivity -> telluric current perturbation -> J change")
    print("      This is a moving grade-0 perturbation at ~1000 mph")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FORESHOCK GRADE-ARRIVAL + ECLIPSE-SEISMICITY")
    print("=" * 60)

    eq = pd.read_csv(DATA_DIR / "earthquakes_m4.5.csv", parse_dates=["time_parsed"])
    print(f"Earthquakes: {len(eq)}")

    antipodal_test(eq)
    eclipse_test(eq)

    print("\n" + "=" * 60)
    print("FRAMEWORK INTERPRETATION")
    print("=" * 60)
    print("""
ANTIPODAL: If the commutator [F, nabla F] propagates through Earth:
  Grade-0 (EM): arrives at c in ~0.04 seconds (diameter/c)
  Grade-2 (seismic P-wave): arrives in ~20 minutes (diameter/8km/s)
  Grade-4 (surface wave): arrives in ~60 minutes

A pre-shock EM signal at the antipode (within hours) would indicate
the grade-0 field perturbation reaches the far side BEFORE the
mechanical rupture occurs. This is the foreshock prediction:
the commutator fires electromagnetically first, then mechanically.

ECLIPSE: An eclipse is a narrow (~100 km wide) shadow moving at
~1000 mph across the surface. The shadow:
  - Reduces photoionization -> atmospheric conductivity drops
  - Changes the local fair-weather field
  - Creates a moving gradient in J along the shadow path
  - This is a natural experiment: a controlled, predictable,
    narrow perturbation crossing known fault zones

If seismicity changes along eclipse paths, the atmospheric
electric coupling to the crust is confirmed.
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
