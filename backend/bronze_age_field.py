#!/usr/bin/env python3
"""
Reconstruct geomagnetic field intensity through the Bronze Age
using the pfm9k.2 Holocene model.

Tests the KT framework prediction: the Levantine anomaly
is a regional J crossing J_c.
"""
import sys, os, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "paleomag"
COEFF_FILE = DATA / "pfm9k2" / "pfm9k2_coeffs" / "pfm9k2-mean.txt"


def load_pfm9k2():
    """Load pfm9k.2 mean Gauss coefficients."""
    with open(COEFF_FILE) as f:
        lines = f.readlines()

    # First line: time axis (years CE, negative = BCE)
    times = np.array([float(x) for x in lines[0].split()])

    # Remaining lines: Gauss coefficients g/h for each time
    # Order: g10, g11, h11, g20, g21, h21, g22, h22, ...
    coeffs = []
    for line in lines[1:]:
        vals = [float(x) for x in line.split()]
        coeffs.append(vals)

    coeffs = np.array(coeffs)  # shape: (n_coeffs, n_times)
    return times, coeffs


def gauss_to_field(coeffs_at_t, lat_deg, lon_deg, r_ratio=1.0):
    """
    Evaluate magnetic field from Gauss coefficients at a surface point.

    Simple dipole + quadrupole for quick evaluation.
    Full SH expansion for accuracy.

    Returns: (Br, Btheta, Bphi, F) in nT
    where F = sqrt(Br^2 + Btheta^2 + Bphi^2) is total intensity.
    """
    theta = math.radians(90 - lat_deg)  # colatitude
    phi = math.radians(lon_deg)
    a = 6371.2  # Earth reference radius km

    # Gauss coefficient indexing:
    # Row 0: g10, Row 1: g11, Row 2: h11
    # Row 3: g20, Row 4: g21, Row 5: h21, Row 6: g22, Row 7: h22
    # General: for degree l, order m:
    #   g_lm at index l^2 + 2*m - 2 (for m > 0: g then h)
    #   Actually simpler: sequential g10,g11,h11,g20,g21,h21,g22,h22,...

    # Build index map
    idx = 0
    coeff_map = {}
    max_l = int((-1 + math.sqrt(1 + len(coeffs_at_t))) / 2)  # rough
    for l in range(1, 15):  # up to degree 14
        for m in range(0, l + 1):
            if idx >= len(coeffs_at_t):
                break
            coeff_map[('g', l, m)] = coeffs_at_t[idx]
            idx += 1
            if m > 0 and idx < len(coeffs_at_t):
                coeff_map[('h', l, m)] = coeffs_at_t[idx]
                idx += 1
        if idx >= len(coeffs_at_t):
            break

    # Evaluate using Associated Legendre Polynomials
    # Simplified: use just dipole (l=1) and quadrupole (l=2) for speed
    # Full evaluation would need scipy.special.lpmv

    ct = math.cos(theta)
    st = math.sin(theta)

    # Dipole (l=1): P10 = cos(theta), P11 = sin(theta)
    g10 = coeff_map.get(('g', 1, 0), 0)
    g11 = coeff_map.get(('g', 1, 1), 0)
    h11 = coeff_map.get(('h', 1, 1), 0)

    # Br = -dV/dr = sum (l+1)(a/r)^(l+2) * [g*cos(m*phi)+h*sin(m*phi)] * P_lm
    # Btheta = (1/r)dV/dtheta
    # For l=1:
    Br = 2 * (g10 * ct + (g11 * math.cos(phi) + h11 * math.sin(phi)) * st)
    Bt = -(- g10 * st + (g11 * math.cos(phi) + h11 * math.sin(phi)) * ct)
    Bp = (g11 * math.sin(phi) - h11 * math.cos(phi))  # divided by sin(theta)
    if abs(st) > 0.01:
        Bp = Bp / st
    else:
        Bp = 0

    # Quadrupole corrections (l=2) - add if coefficients exist
    g20 = coeff_map.get(('g', 2, 0), 0)
    g21 = coeff_map.get(('g', 2, 1), 0)
    h21 = coeff_map.get(('h', 2, 1), 0)
    g22 = coeff_map.get(('g', 2, 2), 0)
    h22 = coeff_map.get(('h', 2, 2), 0)

    # P20 = (3cos^2(theta)-1)/2
    P20 = (3 * ct * ct - 1) / 2
    P21 = 3 * st * ct  # unnormalized
    P22 = 3 * st * st

    Br += 3 * (g20 * P20 + (g21 * math.cos(phi) + h21 * math.sin(phi)) * P21
               + (g22 * math.cos(2*phi) + h22 * math.sin(2*phi)) * P22)

    F = math.sqrt(Br**2 + Bt**2 + Bp**2)
    return Br, Bt, Bp, F


def main():
    times, coeffs = load_pfm9k2()
    print(f"pfm9k.2: {len(times)} time steps, {coeffs.shape[0]} coefficients")
    print(f"Time range: {times[0]:.0f} to {times[-1]:.0f} CE")
    print()

    # Locations to evaluate
    locations = [
        ("Levant (Jerusalem)", 31.8, 35.2),
        ("Levant (Tel Megiddo)", 32.6, 35.2),
        ("Greece (Mycenae)", 37.7, 22.8),
        ("Egypt (Thebes)", 25.7, 32.6),
        ("Anatolia (Hattusa)", 40.0, 34.6),
        ("Mesopotamia (Babylon)", 32.5, 44.4),
        ("China (Anyang)", 36.1, 114.4),
        ("Europe (Rome)", 41.9, 12.5),
        ("S Atlantic (SAA center)", -25.0, -50.0),
        ("Modern reference (London)", 51.5, -0.1),
    ]

    # Focus on Bronze Age collapse period
    # 1500-500 BCE = -1500 to -500 in the model
    mask = (times >= -2000) & (times <= 0)
    t_focus = times[mask]
    c_focus = coeffs[:, mask]

    print("FIELD INTENSITY THROUGH THE BRONZE AGE")
    print("=" * 80)

    for name, lat, lon in locations:
        intensities = []
        for i in range(len(t_focus)):
            _, _, _, F = gauss_to_field(c_focus[:, i], lat, lon)
            intensities.append(F)

        intensities = np.array(intensities)
        peak_idx = np.argmax(intensities)
        peak_time = t_focus[peak_idx]
        peak_val = intensities[peak_idx]
        mean_val = intensities.mean()
        modern_val = intensities[-1] if len(intensities) > 0 else 0

        # Find the intensity at key dates
        bc1200_idx = np.argmin(np.abs(t_focus - (-1200)))
        bc1000_idx = np.argmin(np.abs(t_focus - (-1000)))
        bc800_idx = np.argmin(np.abs(t_focus - (-800)))

        print(f"\n{name} ({lat}N, {lon}E):")
        print(f"  Peak: {peak_val/1000:.1f} uT at {peak_time:.0f} CE ({-peak_time:.0f} BCE)")
        print(f"  Mean: {mean_val/1000:.1f} uT")
        print(f"  At 1200 BCE: {intensities[bc1200_idx]/1000:.1f} uT")
        print(f"  At 1000 BCE: {intensities[bc1000_idx]/1000:.1f} uT")
        print(f"  At  800 BCE: {intensities[bc800_idx]/1000:.1f} uT")
        if peak_val > 0:
            print(f"  Peak/mean ratio: {peak_val/mean_val:.2f}x")

    # Time series for Levant
    print("\n" + "=" * 80)
    print("LEVANT (JERUSALEM) INTENSITY TIME SERIES")
    print("=" * 80)
    print(f"{'Year':>6} {'F(uT)':>8} {'Bar'}")
    print("-" * 50)
    for i in range(len(t_focus)):
        _, _, _, F = gauss_to_field(c_focus[:, i], 31.8, 35.2)
        year = t_focus[i]
        if year % 100 == 0:  # every 100 years
            bar = "#" * int(F / 1000)
            marker = ""
            if -1250 < year < -1150:
                marker = " <-- Bronze Age Collapse"
            elif -1050 < year < -950:
                marker = " <-- Levantine Anomaly peak?"
            elif -850 < year < -750:
                marker = " <-- Anomaly decline"
            print(f"{year:>6.0f} {F/1000:>8.1f} {bar}{marker}")

    # China comparison
    print("\n" + "=" * 80)
    print("CHINA (ANYANG/SHANG) vs LEVANT COMPARISON")
    print("=" * 80)
    print(f"{'Year':>6} {'Levant':>8} {'China':>8} {'Ratio':>8}")
    print("-" * 40)
    for i in range(len(t_focus)):
        year = t_focus[i]
        if year % 200 == 0:
            _, _, _, F_lev = gauss_to_field(c_focus[:, i], 31.8, 35.2)
            _, _, _, F_chi = gauss_to_field(c_focus[:, i], 36.1, 114.4)
            print(f"{year:>6.0f} {F_lev/1000:>8.1f} {F_chi/1000:>8.1f} {F_lev/max(F_chi,1):>8.2f}")


if __name__ == "__main__":
    main()
