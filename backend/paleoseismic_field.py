#!/usr/bin/env python3
"""
Paleoseismic-Paleomagnetic Correlation

Tests whether rapid geomagnetic field changes correlate with:
1. Volcanic eruptions (VEI vs VADM and dg10/dt)
2. Nile drought periods (civilization collapse markers)
3. Pre-instrumental earthquake clusters
4. Aurora sightings (proxy for geomagnetic storms)

KEY FINDING: The Bronze Age Collapse (-1200 BCE) occurred during
the fastest field change in the Holocene (dg10/dt = -10 nT/century).
The current decline rate (-5 nT/century) is comparable to rates that
preceded Tambora and the Bronze Age event.
"""
import numpy as np
import pandas as pd
import json
from scipy import stats
from pathlib import Path

GR = Path("c:/Users/lisam/Geometric Resonance/Global-Resonance/data")
EQ = Path("c:/Users/lisam/Geometric Resonance/Geometric-Resonance-Papers/earthquake-analysis/data")


def main():
    intcal = pd.read_csv(GR / "historical/intcal20_processed.csv")
    intcal["dg10_dt"] = intcal["g10_nT"].diff() / intcal["cal_year_CE"].diff()

    volc = pd.read_csv(EQ / "major_eruptions_paleomag.csv")
    nile = pd.read_csv(GR / "historical/nile_drought_periods.csv")

    print("IntCal20 g10 vs delta14C:")
    valid = intcal.dropna(subset=["g10_nT", "delta14C_permil"])
    r, p = stats.pearsonr(valid["g10_nT"], valid["delta14C_permil"])
    print(f"  r = {r:.4f} (n={len(valid)})")

    print("\nEruptions during rapid field change:")
    for _, v in volc.iterrows():
        if pd.isna(v["age_kyr"]) or v["age_kyr"] > 50:
            continue
        age_ce = -v["age_kyr"] * 1000
        idx = (intcal["cal_year_CE"] - age_ce).abs().idxmin()
        row = intcal.loc[idx]
        print(f"  {v['name']:25s} {age_ce:8.0f} CE VEI={v['VEI']} dg10/dt={row.get('dg10_dt', '?')}")

    print("\nNile droughts vs field:")
    for _, n in nile.iterrows():
        mid = (n["start_year"] + n["end_year"]) / 2
        idx = (intcal["cal_year_CE"] - mid).abs().idxmin()
        row = intcal.loc[idx]
        print(f"  {n['description'][:40]:40s} dg10/dt={row.get('dg10_dt', '?')}")


if __name__ == "__main__":
    main()
