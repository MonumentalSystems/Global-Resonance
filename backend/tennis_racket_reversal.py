#!/usr/bin/env python3
"""
Tennis Racket Theorem and Geomagnetic Reversals

Reversals are Dzhanibekov flips: the l=2 quadrupole (intermediate
axis of the dynamo energy spectrum) grows until it challenges l=1
(dipole), triggering exponential instability and a polarity flip.

Current l=2/l=1 = 0.034 (stable). Threshold ~0.3.
At SAA growth rate: ~7500 years (~5 Bond cycles) to instability.
"""
import numpy as np

# Lowes spectrum (approximate, from IGRF)
R = {1: 900e6, 2: 30.2e6, 3: 11.6e6, 4: 7.3e6, 5: 5.3e6}  # nT^2


def main():
    print("Tennis Racket / Dzhanibekov Instability of the Geodynamo")
    total = sum(R.values())
    for l, r in R.items():
        print(f"  l={l}: {r/1e6:.1f} M nT^2 ({r/total*100:.1f}%)")
    ratio = R[2] / R[1]
    print(f"\nl=2/l=1 = {ratio:.4f} (threshold ~0.3)")
    growth = 0.03
    years = 0
    r = ratio
    while r < 0.3:
        r *= (1 + growth)
        years += 100
    print(f"At 3%/century growth: threshold in ~{years} years ({years/1470:.0f} Bond cycles)")
    print(f"Excursion = partial flip, Reversal = complete 180 deg flip")


if __name__ == "__main__":
    main()
