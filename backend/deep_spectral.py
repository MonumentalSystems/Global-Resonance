#!/usr/bin/env python3
"""
Multi-Scale Spectral Cascade: 5.3 Myr to 1 year resolution

Tests the subharmonic ladder across ALL available records:
LR04 (5.3Myr) -> NGRIP (60kyr) -> IntCal20 (55kyr) -> Common Era (2kyr)
"""
import numpy as np
from scipy import signal
from pathlib import Path

GR = Path("c:/Users/lisam/geo resonance/Global-Resonance/data")

KNOWN = {41000: "Obliq", 23000: "Prec", 100000: "Ecc", 413000: "400kEcc",
         1470: "Bond", 2400: "Hallst", 1000: "Eddy", 210: "deVries",
         88: "Gleiss", 22: "Hale", 11: "Schwabe"}


def spectral_peaks(data, fs, label, min_T, max_T, n_peaks=8):
    dt = signal.detrend(data)
    freqs, psd = signal.welch(dt, fs=fs, nperseg=min(512, len(dt) // 3))
    periods = 1 / freqs[1:]
    idx = signal.find_peaks(psd[1:], prominence=np.max(psd[1:]) * 0.03)[0]
    peaks = sorted([(periods[i], psd[1:][i]) for i in idx if min_T < periods[i] < max_T],
                   key=lambda x: x[1], reverse=True)

    print(f"\n  {label}:")
    for period, power in peaks[:n_peaks]:
        match = ""
        for T, name in KNOWN.items():
            if abs(period / T - 1) < 0.12:
                match = f" = {name} ({T}yr)"
                break
        if not match:
            for T, name in KNOWN.items():
                ratio = period / T
                if 1.8 < ratio < 30 and abs(ratio - round(ratio)) < 0.1:
                    match = f" = {round(ratio)}x{name}"
                    break
                ratio = T / period
                if 1.8 < ratio < 30 and abs(ratio - round(ratio)) < 0.1:
                    match = f" = {name}/{round(ratio)}"
                    break
        print(f"    T = {period:8.0f} yr  pwr = {power:.2e}{match}")


def main():
    print("=" * 70)
    print("  MULTI-SCALE SPECTRAL CASCADE")
    print("=" * 70)

    # LR04 (5.3 Myr)
    lr04 = []
    with open(GR / "deep_time/lr04_d18o_5.3myr.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try: lr04.append((float(parts[0]), float(parts[1])))
                except: pass
    lr04_d18o = np.array([x[1] for x in lr04])
    print(f"\nLR04: {len(lr04)} pts, 5.3 Myr")
    spectral_peaks(lr04_d18o, 1.0, "LR04 orbital band (1kyr)", 5000, 500000)
    spectral_peaks(lr04_d18o, 1.0, "LR04 sub-orbital band (1kyr)", 500, 5000)

    # NGRIP d18O (60 kyr)
    ngrip = []
    with open(GR / "deep_time/ngrip_gicc05_60ka.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    a = float(parts[0])  # age b2k
                    d = float(parts[2])  # d18O is column 3
                    if 0 < a < 70000 and -50 < d < -20:
                        ngrip.append((a, d))
                except: pass
    na = np.array([x[0] for x in ngrip])
    nd = np.array([x[1] for x in ngrip])
    print(f"\nNGRIP d18O: {len(ngrip)} pts, {na.min():.0f} to {na.max():.0f} yr BP")
    yrs = np.arange(na.min(), na.max(), 50)
    nd_i = np.interp(yrs, na, nd)
    spectral_peaks(nd_i, 1 / 50, "NGRIP 60kyr millennial band (50yr)", 200, 30000)

    # IntCal20
    ic = []
    with open(GR / "historical/intcal20.14c") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try: ic.append((float(parts[0]), float(parts[3])))
                except: pass
    ic_bp = np.array([x[0] for x in ic])
    ic_d14c = np.array([x[1] for x in ic])

    # Holocene
    mask = ic_bp < 12000
    if mask.sum() > 100:
        yrs_h = np.arange(ic_bp[mask].min(), ic_bp[mask].max(), 10)
        d14c_h = np.interp(yrs_h, ic_bp[mask], ic_d14c[mask])
        print(f"\nIntCal20 Holocene: {mask.sum()} pts")
        spectral_peaks(d14c_h, 1 / 10, "IntCal20 Holocene (10yr)", 50, 6000)

    # Common Era
    mask = ic_bp < 2000
    if mask.sum() > 100:
        print(f"\nIntCal20 Common Era: {mask.sum()} pts")
        spectral_peaks(ic_d14c[mask], 1.0, "IntCal20 CE (1yr)", 8, 500)

    # DO events in NGRIP
    print(f"\n{'='*70}")
    print(f"  DANSGAARD-OESCHGER SPACING")
    print(f"{'='*70}")
    rate = np.diff(nd_i) / 50
    jumps = np.where(rate > np.percentile(rate, 99.5))[0]
    events = [yrs[jumps[0]]]
    for j in jumps[1:]:
        if yrs[j] - events[-1] > 1000:
            events.append(yrs[j])
    print(f"  {len(events)} rapid warmings found")
    if len(events) > 2:
        spacings = np.diff(events)
        print(f"  Mean spacing: {np.mean(spacings):.0f} yr")
        print(f"  = {np.mean(spacings)/1470:.1f}x Bond cycle")
        for s in spacings:
            print(f"    {s:.0f} yr = {s/1470:.1f}x Bond")


if __name__ == "__main__":
    main()
