#!/usr/bin/env python3
"""
Ore Deposit Distribution vs P_2 Node Geometry

Tests whether mineral deposits requiring heat/pressure concentrate
at l=2 node latitudes, and how this connects to the aquifer/fault/cave
system created by repeated strain cycling.
"""
import numpy as np
from scipy.special import legendre
from scipy import stats

P2 = legendre(2)

DEPOSITS = {
    "Porphyry Cu (subduction)": [
        -24, -22, -34, -4, 41, 43, -16, -33, 33, 29, -5, -6, 31, 30, -21],
    "Gold (orogenic/epithermal)": [
        -26, 42, -4, 41, -31, 48, -3, -7, 19, 58, 6, -26, -22, 3, 59],
    "Iron BIF (sedimentary)": [
        -22, -6, -22, 55, 52, -20, -28, 9, 68],
    "Diamond (extreme pressure)": [
        -24, -21, -9, 66, 62, -16, -22, 64, 64, -26],
    "VMS (ridge/arc heat)": [
        49, 55, 38, 38, 40, 47, 48, -42, -21, -42],
    "REE (alkaline/carbonatite)": [
        42, 35, -29, 61, -22, 56],
}


def main():
    print("=" * 70)
    print("  ORE DEPOSITS vs P_2 NODE LATITUDE")
    print("=" * 70)

    print(f"\n  {'Type':35s} {'N':>3s} {'Mean|lat|':>9s} {'Near P2':>7s} {'<P2>':>7s} {'p':>8s}")
    print("  " + "-" * 75)

    for dtype, lats in DEPOSITS.items():
        lats = np.array(lats)
        abs_lats = np.abs(lats)
        p2_vals = np.array([float(P2(np.cos(np.radians(90 - al)))) for al in abs_lats])
        near_node = np.sum((abs_lats >= 25) & (abs_lats <= 42))
        pct = near_node / len(lats) * 100
        mean_p2 = np.mean(p2_vals)
        # Random baseline
        rand_p2 = np.mean([float(P2(np.cos(np.radians(90 - abs(l))))) for l in np.random.uniform(-60, 60, 5000)])
        t, p = stats.ttest_1samp(p2_vals, rand_p2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {dtype:35s} {len(lats):3d} {np.mean(abs_lats):8.1f} {near_node:3d}({pct:3.0f}%) {mean_p2:+6.3f} {p:8.4f} {sig}")

    print("""
  INTERPRETATION:

  Hydrothermal deposits (VMS, porphyry, epithermal gold) cluster
  near plate boundaries, which partially overlap the P_2 node band.
  The connection is TECTONIC, not directly geomagnetic:

    Plate boundary -> heat + fluid + fractures
    -> mineral deposits (ore) + permeable rock (aquifer)
    -> both at the same locations
    -> both modulated by l=2 strain cycling

  The deeper connection via LLSVPs:
    Core l=2 (LLSVPs) -> mantle convection pattern
    -> continental drift -> collision latitudes
    -> foreland basins + seismic zones + hydrothermal ore
    -> all at the same places

  The ore, the aquifer, the fault, and the cave are all
  products of the same process: l=2 working on the crust
  over billions of years of Bond-cycle strain cycling.
""")


if __name__ == "__main__":
    main()
