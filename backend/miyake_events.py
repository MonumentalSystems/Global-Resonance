#!/usr/bin/env python3
"""
Miyake Events in the Subharmonic Cascade

Miyake events (extreme cosmic ray / solar proton events) are spaced
at de Vries (~210yr) and Bond (~1470yr) intervals — they sit ON the
subharmonic ladder as maximum-amplitude spikes.

774 -> 993 CE: 219yr = 1.0x de Vries (4% error)
-5475 -> -5259 BCE: 216yr = 1.0x de Vries (0% error)
-660 BCE -> 774 CE: 1434yr = 1.0x Bond (2% error)
"""
import numpy as np
import pandas as pd
from pathlib import Path

GR = Path("c:/Users/lisam/Geometric Resonance/Global-Resonance/data")

MIYAKE = [
    (774, 4.20, "774/775 CE mega SPE"),
    (993, 1.40, "993/994 CE SPE"),
    (-660, 1.00, "660 BCE candidate"),
    (-5475, 1.48, "5480 BCE spike"),
    (-5259, 0.16, "5259 BCE largest (tree-ring)"),
    (-7176, 0.32, "7176 BCE candidate"),
]


def main():
    lines = []
    with open(GR / "historical/intcal20.14c") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.strip().split(",")
            if len(parts) >= 5:
                try:
                    lines.append({"cal_CE": 1950 - float(parts[0]), "delta14C": float(parts[3])})
                except: pass
    ic = pd.DataFrame(lines).sort_values("cal_CE")
    ic["rel_dipole"] = 1.0 / (1 + ic["delta14C"] / 500)

    print("Miyake events + field state:")
    for year, rate, name in MIYAKE:
        idx = (ic["cal_CE"] - year).abs().idxmin()
        r = ic.loc[idx]
        print(f"  {year:8d} CE  M={r['rel_dipole']:.3f}  rate={rate:.2f}  {name}")

    print("\nSpacing analysis:")
    events = sorted(MIYAKE, key=lambda x: x[0])
    for i in range(1, len(events)):
        gap = events[i][0] - events[i-1][0]
        for name, T in [("Bond", 1470), ("de Vries", 210)]:
            ratio = gap / T
            if abs(ratio - round(ratio)) < 0.15:
                print(f"  {events[i-1][0]} -> {events[i][0]}: {gap}yr = {ratio:.1f}x {name}")


if __name__ == "__main__":
    main()
