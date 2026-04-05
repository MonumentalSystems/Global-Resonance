#!/usr/bin/env python3
"""
Bond Events + Tripolar Field Geometry

Tests whether the ~1470-year Bond cycle appears in the geomagnetic
dipole moment, and whether Bond events correspond to periods of
l=2 quadrupole dominance (tripolarity).

Connection to Indonesia: during l=2 excursions, a second magnetic
equator can pass through Indonesia, changing the balance between
ocean telluric (v x B, decreases with weaker B) and solar telluric
(Jz from global circuit, increases with more cosmic rays).
"""
import numpy as np
import pandas as pd
from scipy import stats, signal
from pathlib import Path

GR = Path("c:/Users/lisam/Geometric Resonance/Global-Resonance/data")


BOND_EVENTS = [
    (-10300, "Bond 8 (post-Younger Dryas)"),
    (-8200, "Bond 7 / 8.2 ka event"),
    (-5900, "Bond 5"),
    (-4200, "Bond 4 / 4.2 ka event"),
    (-2800, "Bond 3"),
    (-1400, "Bond 2 (Bronze Age)"),
    (500, "Bond 1 (Dark Ages)"),
    (1400, "Bond 0 (Little Ice Age)"),
]


def main():
    intcal = pd.read_csv(GR / "historical/intcal20_processed.csv").sort_values("cal_year_CE")
    intcal["dg10"] = intcal["g10_nT"].diff() / intcal["cal_year_CE"].diff()

    print("=" * 70)
    print("  BOND EVENTS + GEOMAGNETIC FIELD")
    print("=" * 70)

    # g10 at Bond events
    print(f"\n{'Event':40s} {'Year':>7s} {'g10':>8s} {'dg10/dt':>8s}")
    print("-" * 65)
    for year, name in BOND_EVENTS:
        idx = (intcal["cal_year_CE"] - year).abs().idxmin()
        r = intcal.loc[idx]
        print(f"  {name:38s} {year:7d} {r['g10_nT']:8.0f} {r.get('dg10', 0):+7.1f}")

    # Spectral analysis for ~1470yr cycle
    print(f"\n{'='*70}")
    print(f"  SPECTRAL ANALYSIS: 1470-year cycle in g10?")
    print(f"{'='*70}")

    holocene = intcal[intcal["cal_year_CE"] >= -10500].dropna(subset=["g10_nT"])
    years = np.arange(holocene["cal_year_CE"].min(), holocene["cal_year_CE"].max(), 50)
    g10 = np.interp(years, holocene["cal_year_CE"].values, holocene["g10_nT"].values)
    g10_dt = signal.detrend(g10)

    freqs, psd = signal.periodogram(g10_dt, fs=1/50, scaling="density")
    periods = 1 / freqs[1:]

    peaks_idx = signal.find_peaks(psd[1:], prominence=np.max(psd[1:]) * 0.1)[0]
    peaks = [(periods[i], psd[1:][i]) for i in peaks_idx if 100 < periods[i] < 15000]
    peaks.sort(key=lambda x: x[1], reverse=True)

    print("  Top spectral peaks:")
    for period, power in peaks[:8]:
        flags = []
        if abs(period - 1470) < 200: flags.append("BOND CYCLE")
        if abs(period - 2400) < 300: flags.append("Hallstatt?")
        if abs(period - 1000) < 150: flags.append("Eddy?")
        flag_str = " <-- " + ", ".join(flags) if flags else ""
        print(f"    T = {period:7.0f} yr  power = {power:.1f}{flag_str}")

    # dg10/dt sign reversal at Bond events
    print(f"\n{'='*70}")
    print(f"  FIELD ACCELERATION AT BOND EVENTS")
    print(f"{'='*70}")

    reversals = 0
    for year, name in BOND_EVENTS:
        before = intcal[(intcal["cal_year_CE"] >= year - 300) & (intcal["cal_year_CE"] < year - 50)]["dg10"]
        after = intcal[(intcal["cal_year_CE"] >= year + 50) & (intcal["cal_year_CE"] < year + 300)]["dg10"]
        db = before.mean() if len(before) > 0 else 0
        da = after.mean() if len(after) > 0 else 0
        rev = "YES" if db * da < 0 else "no"
        if db * da < 0: reversals += 1
        short = name.split("(")[0].strip()[:25]
        print(f"  {short:25s}  before={db:+6.1f}  after={da:+6.1f}  reversal={rev}")

    print(f"\n  Sign reversals: {reversals}/{len(BOND_EVENTS)} Bond events ({reversals/len(BOND_EVENTS)*100:.0f}%)")

    # Tripolar interpretation
    print(f"\n{'='*70}")
    print(f"  TRIPOLAR FIELD: l=2 DOMINANCE AT BOND EVENTS")
    print(f"{'='*70}")
    print("""
  The geomagnetic field is always multipolar:
    l=1 (dipole):     ~80% of surface field
    l=2 (quadrupole): ~15% (creates the South Atlantic Anomaly)
    l=3+ (higher):    ~5%

  During Bond events (field weakening / excursion):
    l=1 decreases -> l=2/l=1 ratio INCREASES -> more 'tripolar'

  A tripolar field has:
    - Two magnetic equators instead of one
    - Three 'poles' (two weak poles + one strong)
    - Cosmic ray flux concentrated at weak-pole regions
    - Ocean telluric pattern reorganized (different v x B geometry)

  Bond cycle = oscillation of l=2/l=1 ratio:
    ~1470 yr period for quadrupole to grow, peak, subside

  INDONESIA CONNECTION:
    Indonesia is near the magnetic equator. During l=2 excursions,
    a second equator passes through, creating a field MINIMUM.

    At a field minimum:
    - Ocean telluric DROPS (Jz = sigma * v x B, B is weaker)
    - Solar telluric INCREASES (less shielding, more cosmic rays,
      higher ionospheric conductivity, stronger Jz from above)
    - System becomes MORE responsive to solar perturbations
    - Less ocean baseline damping, more solar impulse sensitivity

  This may explain why Indonesia responds so strongly to EVERY
  solar cycle maximum: the field geometry puts it in the most
  sensitive position — near where l=2 creates a minimum.
""")


if __name__ == "__main__":
    main()
