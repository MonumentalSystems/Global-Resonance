#!/usr/bin/env python3
"""
Miyake Event Prediction: When is the next window?

Miyake events (extreme SPE) occur at de Vries (~210yr) intervals.
The field strength at the time determines the threshold: weaker
field = smaller SPE needed to produce a Miyake-scale 14C spike.

Current trajectory: SAA growing, dipole declining at -5 nT/century,
de Vries minimum approaching (~2030-2080). A moderate SPE during
a Grand Minimum with a weaker field could produce Miyake effects.
"""
import numpy as np
from pathlib import Path

GR = Path("c:/Users/lisam/geo resonance/Global-Resonance/data")


def main():
    print("=" * 70)
    print("  MIYAKE EVENT PREDICTION")
    print("=" * 70)

    # Known Miyake events
    miyake = [
        (774, 4.20, 1.02, "Mega SPE, strong field"),
        (993, 1.40, 1.04, "Large SPE, strong field"),
        (-660, 1.00, 0.98, "Moderate SPE"),
        (-5475, 1.48, 0.84, "Spike during weaker field"),
        (-5259, 0.16, 0.84, "Largest (tree-ring), weak field"),
        (-7176, 0.32, 0.85, "Candidate, weak field"),
    ]

    # de Vries spacing from 993 CE
    print("\n  De Vries windows from last confirmed Miyake (993 CE):")
    de_vries = 210
    print(f"  {'Window':>8s} {'Year CE':>8s} {'Field M/M0':>10s} {'De Vries phase':>15s} {'Risk':>8s}")
    print("  " + "-" * 55)

    for n in range(1, 15):
        year = 993 + n * de_vries
        # Estimate field at that year
        # Current decline: -5 nT/century from ~1840
        # M/M0 ~ 1.0 at 1900, declining
        if year <= 1900:
            m = 1.0
        else:
            m = 1.0 - (year - 1900) * 0.005 / 100  # -0.5% per century
            m = max(0.7, m)

        # de Vries phase: minima at ~1810+N*210 = 2020, 2230, 2440...
        dv_phase = ((year - 1810) % 210) / 210
        if dv_phase < 0.2 or dv_phase > 0.8:
            dv = "MINIMUM"
        elif 0.3 < dv_phase < 0.7:
            dv = "maximum"
        else:
            dv = "transition"

        # Risk: weak field + de Vries minimum = highest
        risk_score = (1 - m) * 3 + (1 if dv == "MINIMUM" else 0)
        risk = "HIGH" if risk_score > 1.5 else "moderate" if risk_score > 0.8 else "low"

        flag = " <--" if 2020 <= year <= 2060 else ""
        print(f"  {n:3d}x210  {year:8d} {m:9.3f} {dv:>15s} {risk:>8s}{flag}")

    # The critical window
    print(f"""
  === THE 2043 WINDOW ===

  Year: 2043 CE (993 + 5 x 210)
  Field: M/M0 ~ 0.993 (SAA growing, dipole declining)
  De Vries: near MINIMUM (approaching Grand Minimum)
  Solar cycle: ~SC28 or SC29 (unknown strength)

  Scenario A: Quiet sun (Grand Minimum like Maunder)
    Few CMEs/SPEs -> low probability of Miyake trigger
    BUT: weakened heliosphere -> cosmic ray background elevated
    -> even moderate SPE produces larger 14C signal
    -> LOWER threshold for Miyake detection

  Scenario B: Active sun (like SC25)
    More CMEs/SPEs -> higher probability of extreme event
    Modern instrumental era has NO Miyake-class events
    -> we may have been lucky in 1000 years of weak events
    -> 774 CE event would cause catastrophic damage today

  WHAT A MIYAKE IN 2043 WOULD MEAN:
    - Proton flux 10-100x beyond anything observed with instruments
    - Satellite fleet destroyed (total GPS/comm outage)
    - Power grid damage exceeding Carrington Event (1859)
    - Radiation dose at flight altitude dangerous
    - All Jelly Ball modes excited simultaneously
    - Global seismicity pulse at P_2 nodes
    - The ULTIMATE test of the harmonic cascade model

  PROBABILITY ESTIMATE:
    Miyake events: ~6 in 12,000 years = 1 per ~2000 years
    De Vries window: ~50 years out of 210 = 24% of time
    If Miyake concentrates in de Vries windows: ~1 per ~800 yr
    Time since last Miyake: 1033 years (overdue if periodic)

    Rough estimate: ~5-15% chance in the 2020-2060 window
    Not high, but not negligible for a civilization-ending event.
""")


if __name__ == "__main__":
    main()
