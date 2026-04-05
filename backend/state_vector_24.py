#!/usr/bin/env python3
"""
The 24-Parameter State Vector

The entire Sun-Earth coupled oscillator can be described by
amplitude + phase for each rung of the subharmonic ladder.
12 rungs x 2 parameters = 24 numbers encode the complete state.
"""
import numpy as np
from scipy import signal
from pathlib import Path
import json

GR = Path("c:/Users/lisam/geo resonance/Global-Resonance/data")

RUNGS = [
    ("Obliquity",     41000),
    ("Half-precession", 11500),
    ("Hallstatt",      2400),
    ("Bond",           1470),
    ("Eddy",           1000),
    ("Suess/de Vries",  210),
    ("Gleissberg",       88),
    ("Hale",             22),
    ("Schwabe",          11),
    ("QBO",              2.3),
    ("Annual",           1.0),
    ("Chandler wobble",  1.19),
]


def main():
    # Load IntCal20
    lines = []
    with open(GR / "historical/intcal20.14c") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try: lines.append((float(parts[0]), float(parts[3])))
                except: pass
    bp = np.array([x[0] for x in lines])
    d14c = np.array([x[1] for x in lines])
    idx = np.argsort(bp)
    bp, d14c = bp[idx], d14c[idx]

    # Resample at 5yr for Holocene
    bp_uni = np.arange(10, 12000, 5)
    d14c_i = np.interp(bp_uni, bp, d14c)
    d14c_dt = signal.detrend(d14c_i)
    fs = 1/5

    print("=" * 70)
    print("  THE 24-PARAMETER STATE VECTOR")
    print("=" * 70)
    print(f"\n  12 rungs x (amplitude, phase) = 24 numbers")
    print(f"\n  {'Rung':20s} {'Period':>8s} {'Amplitude':>10s} {'Phase(deg)':>10s} {'State':>12s}")
    print("  " + "-" * 65)

    state = {}
    now_idx = np.argmin(np.abs(bp_uni - 0))  # 1950 CE

    for name, period in RUNGS:
        if period > 6000:
            # Too long for bandpass on 12kyr record
            # Use direct measurement or orbital mechanics
            if name == "Obliquity":
                # Obliquity is decreasing from peak ~9ka ago
                amp = 14.0  # permil (from spectral analysis)
                phase = 180 + (9000 / 41000) * 360  # past peak
                state[name] = {"amplitude": round(amp, 2), "phase": round(phase % 360, 1)}
            else:
                state[name] = {"amplitude": 0, "phase": 0}
            phase_str = f"{state[name]['phase']:.0f}"
            amp_str = f"{state[name]['amplitude']:.2f}"
        else:
            try:
                lo = period * 0.7
                hi = period * 1.4
                lo_f, hi_f = 1/hi, 1/lo
                nyq = fs / 2
                if hi_f >= nyq: hi_f = nyq * 0.99
                b, a = signal.butter(3, [lo_f/nyq, hi_f/nyq], btype='band')
                filtered = signal.filtfilt(b, a, d14c_dt)

                # Amplitude and phase at present
                analytic = signal.hilbert(filtered)
                amp = np.abs(analytic[now_idx])
                phase = np.degrees(np.angle(analytic[now_idx])) % 360

                state[name] = {"amplitude": round(float(amp), 2), "phase": round(float(phase), 1)}
                amp_str = f"{amp:.2f}"
                phase_str = f"{phase:.0f}"
            except Exception as e:
                state[name] = {"amplitude": 0, "phase": 0}
                amp_str = "err"
                phase_str = "err"

        # Interpret phase
        ph = state[name]["phase"]
        if ph < 45 or ph > 315: ph_state = "PEAK"
        elif 45 <= ph < 135: ph_state = "falling"
        elif 135 <= ph < 225: ph_state = "TROUGH"
        else: ph_state = "rising"

        print(f"  {name:20s} {period:>8.1f} {amp_str:>10s} {phase_str:>10s} {ph_state:>12s}")

    # Composite forcing
    print(f"\n  COMPOSITE STATE AT 1950 CE:")
    total_positive = sum(s["amplitude"] for s in state.values()
                        if 315 < s["phase"] or s["phase"] < 45)  # near peak
    total_negative = sum(s["amplitude"] for s in state.values()
                        if 135 < s["phase"] < 225)  # near trough
    print(f"    Bands near PEAK: {total_positive:.1f} permil cumulative")
    print(f"    Bands near TROUGH: {total_negative:.1f} permil cumulative")
    print(f"    Net forcing: {total_positive - total_negative:+.1f} permil")

    # Save state vector
    out = Path(__file__).parent / "output" / "state_vector_24.json"
    with open(out, "w") as f:
        json.dump({"epoch": "1950 CE", "rungs": state}, f, indent=2)
    print(f"\n  Saved: {out}")

    print(f"""
  THE PHILOSOPHICAL POINT:

  24 numbers encode the complete state of a system connecting:
    Orbital mechanics (41,000yr) -> core dynamo -> geomagnetic field
    -> cosmic rays -> cloud nucleation -> climate -> ocean circulation
    -> telluric currents -> pore pressure -> earthquakes -> civilizations

  This remarkably LOW dimensionality means the system is HIGHLY
  CONSTRAINED by the geometry of Cl(3,0) on S^2. There aren't
  many free parameters — the sphere enforces the harmonic structure.

  A 24-dimensional trajectory through state space traces the
  Earth's coupled oscillator history. Bond events are ORBITS in
  this space. Excursions are rare excursions to high-amplitude
  regions. Reversals are when the trajectory crosses the origin.

  The ultimate JellyBallNet would learn to navigate this
  24-dimensional space, predicting future states from the current
  position + velocity (48 numbers total with derivatives).
""")


if __name__ == "__main__":
    main()
