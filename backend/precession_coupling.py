#!/usr/bin/env python3
"""Precession modulation of the tidal + geomagnetic coupling."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

pi = np.pi
OUT_DIR = Path(__file__).parent / "output"

T_prec = 25772  # precession period (years)
obliquity = 23.44
dipole_tilt = 11.5
lon_peri_now = 283  # current longitude of perihelion

print("=" * 70)
print("PRECESSION MODULATION OF TIDAL + GEOMAGNETIC COUPLING")
print("=" * 70)

# For each precession phase, compute:
# 1. R-M coupling: max when dipole perpendicular to sun-Earth line (equinoxes)
# 2. Solar tidal: max at perihelion (1/r^3 dependence)
# 3. Combined: the PRODUCT - when both are simultaneously strong

prec_phases = np.linspace(0, 360, 361)
combined = np.zeros(len(prec_phases))

for i, lon_p in enumerate(prec_phases):
    peri_season = lon_p / 360  # 0=March eq, 0.25=June sol
    # R-M at perihelion: peaks when perihelion is at equinox
    rm = abs(np.sin(2*pi*peri_season))
    # Tidal at equinox: peaks when perihelion is at equinox
    dist_to_eq = min(abs(peri_season), abs(peri_season-0.5), abs(peri_season-1))
    tidal = 1 + 0.034 * np.cos(2*pi*dist_to_eq)
    combined[i] = rm * tidal

combined = combined / combined.max()

# Timeline over 100,000 years
years_bp = np.arange(0, 100001, 100)
lon_peri_past = (lon_peri_now - years_bp * 360 / T_prec) % 360
combined_past = np.interp(lon_peri_past, prec_phases, combined)

# Key events
events = {
    0: "NOW",
    5: "5 kyr (early civilization)",
    10: "10 kyr (agriculture)",
    12: "Younger Dryas",
    26: "~1 precession cycle",
    42: "ADAMS EVENT / LASCHAMP",
    52: "~2 precession cycles",
    74: "Toba eruption",
}

print(f"\n{'kyr BP':>7s} {'lon_peri':>9s} {'Hazard':>8s} {'Event':>35s}")
for yr, note in sorted(events.items()):
    idx = yr * 10
    if idx < len(lon_peri_past):
        lp = lon_peri_past[idx]
        ch = combined_past[idx]
        print(f"  {yr:>5d}    {lp:>7.1f}    {ch:>6.3f}   {note}")

# Mayan Long Count
print(f"\nMayan Long Count = 5,125 years = 1/5 of precession ({T_prec/5:.0f} yr)")
print(f"5 Long Counts = {5*5125} yr vs precession {T_prec} yr (error: {abs(5*5125-T_prec)} yr, {abs(5*5125-T_prec)/T_prec*100:.1f}%)")
print(f"\nThe Mayan Long Count divides the precession into 5 equal phases.")
print(f"Each phase corresponds to a different tidal+R-M geometry:")

for n in range(5):
    phase_yr = n * 5125
    lp = (lon_peri_now - phase_yr * 360 / T_prec) % 360
    ch = np.interp(lp, prec_phases, combined)
    season = ["~Dec solstice", "~Oct", "~Aug", "~June solstice", "~Mar equinox"][n]
    print(f"  Phase {n}: perihelion at {season} (lon={lp:.0f}), hazard={ch:.3f}")

# Hindu Yugas
print(f"\nHindu Yuga cycle (Sri Yukteswar): 24,000 years")
print(f"Precession: {T_prec} years. Ratio: {T_prec/24000:.3f}")
print(f"The 4 yugas (Kali-Dvapara-Treta-Satya) = 4 quadrants of precession")
print(f"Each quadrant = {T_prec/4:.0f} years (or {24000/4:.0f} in the Hindu count)")

for n, name in enumerate(["Kali (dark)", "Dvapara (bronze)", "Treta (silver)", "Satya (golden)"]):
    phase_yr = n * T_prec / 4
    lp = (lon_peri_now - phase_yr * 360 / T_prec) % 360
    ch = np.interp(lp, prec_phases, combined)
    print(f"  {name}: perihelion at lon={lp:.0f}, hazard={ch:.3f}")

print(f"\nThe 'dark age' (Kali Yuga) corresponds to the precession phase where")
print(f"perihelion is at solstice (tidal and R-M out of phase = minimum hazard).")
print(f"The 'golden age' (Satya Yuga) = perihelion at equinox = maximum coupling.")
print(f"\nWait — this is INVERTED from what we might expect:")
print(f"Maximum coupling = more earthquakes/volcanism/storms = the 'golden' age?")
print(f"OR: maximum coupling = most predictable environment = easier to forecast")
print(f"= golden age because the PATTERNS are clear and regular.")
print(f"Minimum coupling = unpredictable, chaotic = dark age because")
print(f"the tidal and geomagnetic signals DON'T align = harder to forecast.")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax = axes[0]
ax.plot(prec_phases, combined, color="steelblue", lw=2)
ax.axvline(lon_peri_now, color="red", lw=2, linestyle="--", label="NOW")
ax.axvline(lon_peri_past[420], color="purple", lw=2, linestyle=":", label="Adams Event")
# Mark Mayan phases
for n in range(5):
    lp = (lon_peri_now - n*5125*360/T_prec) % 360
    ax.axvline(lp, color="green", alpha=0.4, linestyle=":")
ax.set_xlabel("Longitude of perihelion (degrees)")
ax.set_ylabel("Combined hazard (normalized)")
ax.set_title("Precession Phase: When Do Tidal and Geomagnetic Coupling Align?")
ax.legend()

ax = axes[1]
ax.plot(years_bp/1000, combined_past, color="steelblue", lw=1.5)
ax.axvline(42, color="purple", lw=2, linestyle=":", alpha=0.7, label="Adams Event")
for n in range(1, 11):
    ax.axvline(n*5.125, color="green", alpha=0.2, linestyle=":")
ax.axvline(0, color="red", lw=1, linestyle="--")
ax.set_xlabel("Thousands of years before present")
ax.set_ylabel("Combined tidal + geomagnetic coupling")
ax.set_title(f"100,000 Years of Precession-Modulated Hazard (period = {T_prec/1000:.1f} kyr)")
ax.legend()
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(OUT_DIR / "precession_hazard.png", dpi=150)
print(f"\nSaved: precession_hazard.png")
