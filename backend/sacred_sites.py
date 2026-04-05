#!/usr/bin/env python3
"""
Sacred Sites, Ley Lines, and Telluric Conductivity Boundaries

Tests whether sacred sites cluster at geological features that
concentrate telluric currents: faults, springs, conductivity boundaries.
"""
import numpy as np
from scipy.special import legendre
from scipy import stats

P2 = legendre(2)

SITES = [
    ("Delphi Oracle", 38.5, 22.5, True, True, True),
    ("Glastonbury Tor", 51.1, -2.7, False, True, True),
    ("Lourdes", 43.1, -0.05, True, True, False),
    ("Chartres Cathedral", 48.4, 1.5, False, True, True),
    ("Santiago de Compostela", 42.9, -8.5, False, True, True),
    ("Stonehenge", 51.2, -1.8, False, False, True),
    ("Avebury", 51.4, -1.9, False, True, True),
    ("Bath (Aquae Sulis)", 51.4, -2.4, True, True, False),
    ("Jerusalem Temple Mount", 31.8, 35.2, True, True, False),
    ("Mecca Zamzam", 21.4, 39.8, True, True, False),
    ("Varanasi", 25.3, 83.0, True, True, False),
    ("Bodh Gaya", 24.7, 85.0, False, False, True),
    ("Ise Shrine", 34.5, 136.7, False, False, True),
    ("Uluru", -25.3, 131.0, False, True, False),
    ("Machu Picchu", -13.2, -72.5, True, True, False),
    ("Teotihuacan", 19.7, -98.8, False, True, True),
    ("Angkor Wat", 13.4, 103.9, False, False, False),
    ("Mount Kailash", 31.1, 81.3, True, True, False),
    ("Sedona vortex", 34.9, -111.8, True, True, True),
    ("Newgrange", 53.7, -6.5, False, False, True),
    ("Olympia", 37.6, 21.6, False, False, True),
    ("Epidaurus", 37.6, 23.1, False, True, False),
    ("Fatima", 39.6, -8.8, False, True, False),
    ("Lhasa Potala", 29.7, 91.1, True, True, False),
    ("Easter Island", -27.1, -109.4, False, False, False),
    ("Göbekli Tepe", 37.2, 38.9, False, True, True),
    ("Karnak Temple", 25.7, 32.7, True, False, False),
    ("Dodona Oracle", 39.5, 20.8, False, True, False),
    ("Delos", 37.4, 25.3, False, True, False),
    ("Mount Olympus", 40.1, 22.4, True, True, False),
]
# Columns: name, lat, lon, has_fault, has_spring, has_geo_boundary


def main():
    print("=" * 70)
    print("  SACRED SITES: Geological + Telluric Distribution")
    print("=" * 70)

    lats = np.array([s[1] for s in SITES])
    faults = sum(1 for s in SITES if s[3])
    springs = sum(1 for s in SITES if s[4])
    boundaries = sum(1 for s in SITES if s[5])
    N = len(SITES)

    # P_2 values
    p2_vals = np.array([float(P2(np.cos(np.radians(90 - abs(l))))) for l in lats])
    near_p2 = sum(1 for l in lats if abs(abs(l) - 35) < 12)

    print(f"\n  {N} sacred sites analyzed:")
    print(f"    On/near fault:          {faults}/{N} ({faults/N*100:.0f}%)")
    print(f"    Spring or holy well:    {springs}/{N} ({springs/N*100:.0f}%)")
    print(f"    Geological boundary:    {boundaries}/{N} ({boundaries/N*100:.0f}%)")
    print(f"    ANY geo feature:        {sum(1 for s in SITES if s[3] or s[4] or s[5])}/{N}")
    print(f"    Near P_2 node (23-47):  {near_p2}/{N} ({near_p2/N*100:.0f}%)")

    # Compare to random: what fraction of Earth's land at 23-47 deg?
    # ~30% of land area is at 23-47 deg latitude
    p_random = 0.30
    binom_p = 1 - stats.binom.cdf(near_p2 - 1, N, p_random)
    print(f"\n    Expected at P_2 band (30% of land): {N*p_random:.0f}")
    print(f"    Observed: {near_p2}")
    print(f"    Binomial p = {binom_p:.4f}")

    # Latitude histogram
    print(f"\n  LATITUDE DISTRIBUTION:")
    bins = [(-40, -20), (-20, 0), (0, 20), (20, 35), (35, 45), (45, 55), (55, 70)]
    for lo, hi in bins:
        n = sum(1 for l in lats if lo <= l < hi)
        bar = "#" * (n * 3)
        p2_zone = " <-- P_2 NODE" if 25 <= (lo + hi) / 2 <= 42 else ""
        print(f"    {lo:+3d} to {hi:+3d}: {n:3d} {bar}{p2_zone}")

    # Mean P_2 value
    mean_p2 = np.mean(p2_vals)
    rand_mean = np.mean([float(P2(np.cos(np.radians(90 - abs(l)))))
                         for l in np.random.uniform(-60, 60, 10000)])
    t, p = stats.ttest_1samp(p2_vals, rand_mean)
    print(f"\n  Mean P_2 at sacred sites: {mean_p2:+.3f}")
    print(f"  Random land P_2:          {rand_mean:+.3f}")
    print(f"  t = {t:.2f}, p = {p:.4f}")

    # The geological feature statistics
    print(f"""
  INTERPRETATION:

  {springs}/{N} ({springs/N*100:.0f}%) of sacred sites have springs/wells.
  {faults}/{N} ({faults/N*100:.0f}%) sit on or near faults.
  {boundaries}/{N} ({boundaries/N*100:.0f}%) are at geological boundaries.

  A 'holy well' is a TELLURIC DISCHARGE POINT:
    Groundwater emerges where geological structure channels it.
    The same structure channels telluric current.
    At the spring: current converges, Ez is enhanced, air is ionized.

  A 'ley line' is a FAULT/AQUIFER TRACE:
    Straight geological features (faults, contacts, ridges)
    that are preferential telluric current paths.
    Sacred sites along them mark spring emergence points.

  The monasteries built on cisterns weren't just practical
  water supply — they were siting on telluric nodes where
  the l=2 coupling chain reaches the surface through springs.
""")


if __name__ == "__main__":
    main()
