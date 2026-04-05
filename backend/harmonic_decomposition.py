#!/usr/bin/env python3
"""
Harmonic Decomposition: Multi-scale contribution to total field variability

Decomposes IntCal20 (55kyr) into harmonic bands corresponding to each
rung of the subharmonic ladder, then measures how much variance each
band contributes and how they interact (constructive/destructive).

Bands:
  Orbital:    > 10,000 yr (obliquity, precession)
  Hallstatt:  1800-3000 yr
  Bond:       1200-1800 yr
  Eddy:       800-1200 yr
  de Vries:   150-280 yr
  Gleissberg: 60-120 yr
  Hale:       18-26 yr
  Schwabe:    9-14 yr

Key question: do these bands constructively interfere at known
disruption events (Bond events, Miyake events, collapses)?
"""
import numpy as np
from scipy import signal
from pathlib import Path

GR = Path("c:/Users/lisam/geo resonance/Global-Resonance/data")


def load_intcal():
    lines = []
    with open(GR / "historical/intcal20.14c") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try:
                    lines.append((float(parts[0]), float(parts[3])))  # cal_BP, delta14C
                except: pass
    bp = np.array([x[0] for x in lines])
    d14c = np.array([x[1] for x in lines])
    # Sort by age
    idx = np.argsort(bp)
    return bp[idx], d14c[idx]


def bandpass(data, fs, lo, hi):
    """Butterworth bandpass filter."""
    nyq = fs / 2
    if hi >= nyq: hi = nyq * 0.99
    if lo <= 0: lo = 1 / len(data) * fs
    b, a = signal.butter(3, [lo/nyq, hi/nyq], btype='band')
    return signal.filtfilt(b, a, data)


def main():
    bp, d14c = load_intcal()
    print("=" * 80)
    print("  HARMONIC DECOMPOSITION: 55,000 years of cosmic ray history")
    print("=" * 80)
    print(f"\n  IntCal20: {len(bp)} pts, {bp.min():.0f} to {bp.max():.0f} yr BP")

    # Resample to uniform 5yr spacing (good for resolving de Vries)
    bp_uni = np.arange(bp.min() + 50, min(bp.max(), 12000) - 50, 5)  # Holocene only for high-res
    d14c_holo = np.interp(bp_uni, bp, d14c)
    d14c_dt = signal.detrend(d14c_holo)
    fs = 1 / 5  # samples per year
    total_var = np.var(d14c_dt)

    print(f"  Holocene resampled: {len(bp_uni)} pts at 5yr, total variance = {total_var:.2f}")

    # Define harmonic bands
    bands = [
        ("Orbital",    5000, 15000, "Obliquity / precession remnant"),
        ("Hallstatt",  1800, 3000,  "Obliquity/17, ~2400yr"),
        ("Bond",       1200, 1800,  "Obliquity/28, ~1470yr"),
        ("Eddy",       700,  1200,  "Obliquity/41, ~1000yr"),
        ("de Vries",   150,  280,   "Bond/7, ~210yr"),
        ("Gleissberg", 60,   120,   "Bond/17, ~88yr"),
        ("Hale-like",  18,   30,    "Bond/67, ~22yr"),
        ("Schwabe-like", 9,  15,    "Bond/134, ~11yr"),
    ]

    print(f"\n  {'Band':15s} {'Period':>12s} {'Variance':>10s} {'% Total':>8s} {'Amplitude':>10s}")
    print("  " + "-" * 60)

    band_signals = {}
    for name, lo_yr, hi_yr, desc in bands:
        lo_freq = 1 / hi_yr  # low period = high freq
        hi_freq = 1 / lo_yr
        try:
            filtered = bandpass(d14c_dt, fs, lo_freq, hi_freq)
            var = np.var(filtered)
            pct = var / total_var * 100
            amp = np.sqrt(2 * var)  # RMS amplitude -> peak
            band_signals[name] = filtered
            print(f"  {name:15s} {lo_yr:5d}-{hi_yr:5d}yr {var:9.2f} {pct:7.1f}% {amp:9.2f} permil")
        except Exception as e:
            print(f"  {name:15s} {lo_yr:5d}-{hi_yr:5d}yr  FAILED: {e}")

    # === INTERACTION: When do bands constructively interfere? ===
    print(f"\n{'='*70}")
    print(f"  CONSTRUCTIVE INTERFERENCE AT KNOWN EVENTS")
    print(f"{'='*70}")

    events = [
        (774, "Miyake 774 CE"),
        (957, "Miyake 993 CE"),  # 1950 - 993
        (3150, "Bronze Age Collapse (-1200 CE)"),
        (6150, "4.2 ka event (Akkadian)"),
        (10150, "8.2 ka event (Bond 7)"),
        (1550, "Dark Ages (400 CE)"),
        (550, "Little Ice Age (1400 CE)"),
        (135, "Tambora (1815 CE)"),
    ]

    available_bands = [n for n in ["Bond", "Hallstatt", "Eddy", "de Vries", "Gleissberg"] if n in band_signals]

    if available_bands:
        print(f"\n  {'Event':35s}", end="")
        for b in available_bands:
            print(f" {b:>10s}", end="")
        print(f" {'Sum':>8s} {'Constr?':>8s}")
        print("  " + "-" * (40 + 11 * len(available_bands)))

        for event_bp, event_name in events:
            idx = np.argmin(np.abs(bp_uni - event_bp))
            if idx < 10 or idx > len(bp_uni) - 10:
                continue
            vals = []
            for b in available_bands:
                v = band_signals[b][idx]
                vals.append(v)
            total = sum(vals)
            # Constructive = all same sign AND large
            same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
            large = abs(total) > np.std(d14c_dt) * 0.5
            constr = "YES" if same_sign and large else "partial" if large else "no"

            print(f"  {event_name:35s}", end="")
            for v in vals:
                color = "+" if v > 0 else "-"
                print(f" {color}{abs(v):8.2f}", end="")
            print(f" {total:+7.2f} {constr:>8s}")

    # === COUPLING BETWEEN BANDS ===
    print(f"\n{'='*70}")
    print(f"  CROSS-BAND CORRELATION (do bands lock phase?)")
    print(f"{'='*70}")

    from scipy.stats import pearsonr
    band_names = list(band_signals.keys())
    print(f"\n  {'':10s}", end="")
    for n in band_names[:6]:
        print(f" {n[:8]:>9s}", end="")
    print()

    for i, n1 in enumerate(band_names[:6]):
        print(f"  {n1[:10]:10s}", end="")
        for j, n2 in enumerate(band_names[:6]):
            if i == j:
                print(f"     1.00 ", end="")
            elif j > i:
                r, p = pearsonr(band_signals[n1], band_signals[n2])
                sig = "*" if p < 0.05 else " "
                print(f" {r:+7.3f}{sig}", end="")
            else:
                print(f"          ", end="")
        print()

    # === ENVELOPE: When is total forcing strongest? ===
    print(f"\n{'='*70}")
    print(f"  TOTAL HARMONIC FORCING ENVELOPE")
    print(f"{'='*70}")

    # Sum all bands
    total_forcing = np.zeros_like(d14c_dt)
    for name, sig in band_signals.items():
        total_forcing += sig

    # Find peaks of the envelope
    envelope = np.abs(signal.hilbert(total_forcing))
    peak_idx = signal.find_peaks(envelope, distance=50, prominence=np.std(envelope))[0]

    # Top 15 strongest forcing moments
    top = sorted(peak_idx, key=lambda i: envelope[i], reverse=True)[:15]
    print(f"\n  Top 15 strongest multi-band forcing moments:")
    print(f"  {'Year CE':>8s} {'yr BP':>8s} {'Forcing':>8s} {'Known event?':>35s}")
    print("  " + "-" * 65)

    known_events = {774: "Miyake 774", 993: "Miyake 993", -1200: "Bronze Age Collapse",
                    -4200: "Akkadian", -8200: "8.2ka Bond 7", 400: "Bond 1 / Dark Ages",
                    1400: "Bond 0 / LIA", 1815: "Tambora", -2800: "Bond 3",
                    -5900: "Bond 5", 535: "LALIA"}

    for i in top:
        yr_bp = bp_uni[i]
        yr_ce = 1950 - yr_bp
        forcing = envelope[i]
        match = ""
        for yr, name in known_events.items():
            if abs(yr_ce - yr) < 100:
                match = name
                break
        print(f"  {yr_ce:8.0f} {yr_bp:8.0f} {forcing:8.2f} {match:>35s}")


if __name__ == "__main__":
    main()
