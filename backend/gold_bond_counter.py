#!/usr/bin/env python3
"""
Gold as Bond Cycle Counter

Repeated melt/remobilization at subduction zones concentrates Au.
Each Bond cycle: fracture -> fluid flow -> partial melt -> Au upward.
The deposit tonnage is proportional to cumulative l=2 strain cycles.

Au/cycle rate by type:
  Orogenic: 0.006 t/cycle (slow, old deposits, collision zones)
  Porphyry: 0.42 t/cycle (rapid, young, subduction heat)
  Epithermal: 0.79 t/cycle (fastest, shallowest)
"""
import numpy as np
from scipy import stats

DEPOSITS = [
    ("Witwatersrand", -26, 2900, 50000, "paleoplacer"),
    ("Grasberg", -4, 3, 2500, "porphyry"),
    ("Muruntau", 42, 290, 5300, "orogenic"),
    ("Carlin Trend", 41, 40, 4500, "sediment-hosted"),
    ("Kalgoorlie", -31, 2700, 2200, "orogenic"),
    ("Timmins", 48, 2700, 2000, "orogenic"),
    ("Sukhoi Log", 58, 450, 2000, "orogenic"),
    ("Lihir", -3, 1, 1500, "epithermal"),
    ("Yanacocha", -7, 12, 1200, "epithermal"),
    ("Obuasi", 6, 2100, 1100, "orogenic"),
    ("Kibali", 3, 2500, 800, "orogenic"),
    ("Olympic Dam", -31, 1600, 1200, "IOCG"),
    ("Pebble", 60, 90, 1800, "porphyry"),
    ("Cadia", -34, 440, 800, "porphyry"),
]

BOND = 0.00147  # Myr


def main():
    print("Gold deposits as Bond cycle counters:")
    print(f"{'Deposit':20s} {'Age Ma':>7s} {'Au t':>7s} {'Cycles':>10s} {'t/cycle':>8s}")
    for name, lat, age, au, dtype in DEPOSITS:
        cycles = age / BOND
        rate = au / cycles
        print(f"  {name:18s} {age:7.0f} {au:7.0f} {cycles:10.0f} {rate:8.4f}")

    # By type
    print("\nBy type:")
    for dtype in ["orogenic", "porphyry", "epithermal"]:
        subset = [(a, t) for _, _, a, t, d in DEPOSITS if d == dtype]
        if subset:
            rates = [t / (a / BOND) for a, t in subset]
            print(f"  {dtype:12s}: {np.mean(rates):.4f} t/cycle (n={len(subset)})")


if __name__ == "__main__":
    main()
