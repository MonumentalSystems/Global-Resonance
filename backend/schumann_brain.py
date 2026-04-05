#!/usr/bin/env python3
"""
Schumann-Brain Coupling: Does the cascade reach into consciousness?

Schumann f1 (7.83 Hz) sits at the alpha-theta EEG boundary.
The harmonic cascade modulates Schumann -> does it modulate brains?
"""
import numpy as np


def main():
    print("=" * 70)
    print("  SCHUMANN-BRAIN FREQUENCY OVERLAP")
    print("=" * 70)

    schumann = [7.83, 14.3, 20.8, 27.3, 33.8, 39.0, 45.0]
    eeg = {
        "Delta": (0.5, 4, "Deep sleep, healing"),
        "Theta": (4, 8, "Meditation, creativity, memory"),
        "Alpha": (8, 13, "Relaxed awareness, flow state"),
        "Beta": (13, 30, "Active thinking, concentration"),
        "Gamma": (30, 100, "Higher cognition, binding"),
    }

    print(f"\n  Schumann modes vs EEG bands:")
    print(f"  {'Schumann':>10s}  {'EEG band':>10s}  {'Brain state':>30s}")
    print("  " + "-" * 55)
    for i, sf in enumerate(schumann):
        band = "?"
        state = ""
        for name, (lo, hi, desc) in eeg.items():
            if lo <= sf <= hi:
                band = name
                state = desc
                break
        print(f"  f{i+1} = {sf:5.2f} Hz  {band:>10s}  {state:>30s}")

    print(f"""
  KEY OVERLAPS:
    f1 (7.83 Hz) = THETA-ALPHA BOUNDARY
      This is the meditation/relaxation transition frequency.
      Published: Schumann f1 coherence with EEG alpha
      (Saroka & Persinger, 2014; Pobachenko et al., 2006)

    f2 (14.3 Hz) = ALPHA-BETA BOUNDARY
      Transition from relaxed to active cognition.
      14.3 Hz = Hallstatt cycle Schumann frequency.

    f3 (20.8 Hz) = MID-BETA
      Active thinking frequency.

    f4-f7 (27-45 Hz) = BETA-GAMMA
      Higher cognitive processing.

  MODULATION PATHWAY:
    Solar cycle -> cosmic rays -> ionospheric conductivity
    -> Schumann amplitude/frequency shifts (measured: +/- 0.1 Hz)
    -> neural entrainment at population level?

  PUBLISHED EVIDENCE:
    - Hospital admissions for depression correlate with Kp (Berk 2006)
    - Suicide rates correlate with geomagnetic activity (Berk 2006)
    - Heart rate variability correlates with Schumann (McCraty 2017)
    - Reaction times change during geomagnetic storms (Babayev 2010)
    - Melatonin suppression during storms (Burch 1999)

  THE CASCADE CONNECTION:
    Bond event -> field weakening -> more cosmic rays
    -> ionospheric conductivity change -> Schumann shift
    -> sustained shift in alpha-theta boundary
    -> population-level cognitive/mood changes over decades

    This is speculative but the FREQUENCY OVERLAP is exact.
    7.83 Hz is not chosen by biology — it is the cavity resonance
    of the Earth-ionosphere system. The brain evolved IN this field.

    If neural oscillators entrain to Schumann (even weakly),
    then the entire harmonic cascade reaches consciousness:
    Solar cycle -> mood cycles, Bond events -> civilizational
    psychology, excursions -> species-level cognitive stress.
""")


if __name__ == "__main__":
    main()
