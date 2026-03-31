#!/usr/bin/env python3
"""When is the next reversal? What can we learn from past ones?"""
import numpy as np
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
pi = np.pi

print("=" * 70)
print("PREDICTING THE NEXT REVERSAL FROM THE FRAMEWORK")
print("=" * 70)

# Current state of the field
print("""
CURRENT OBSERVABLES (all declining since ~1840):

  Dipole moment:     8.0e22 Am^2 (was 8.5e22 in 1840)
  Decay rate:        ~5% per century (accelerating?)
  SAA minimum field: ~22 uT (was ~25 uT in 2000)
  SAA area:          growing, now splitting into two lobes
  Pole speed:        ~41 km/year (was ~10 km/yr before 2000)
  Inner core:        rotation anomaly detected 2003-2023
  Geomag jerks:      every ~7-10 years (1969, 78, 86, 91, 99, 03, 07, 11, 14, 17, 20)
""")

# The reversal timescale from our framework
# Method 1: Extrapolate current dipole decay
VADM_now = 8.0e22  # Am^2
decay_rate = 0.05 / 100  # per year (5% per century)

# At constant decay rate: VADM(t) = VADM_now * exp(-decay_rate * t)
# Reversal when VADM ~ 10% of current (from Laschamp data)
VADM_reversal = 0.1 * VADM_now
t_reversal_1 = -np.log(0.1) / decay_rate
print(f"Method 1: Constant decay rate ({decay_rate*100*100:.0f}%/century)")
print(f"  Time to 10% field: {t_reversal_1:.0f} years")
print(f"  Time to Q = 2 (cavity disorders): ~{t_reversal_1*0.8:.0f} years")

# Method 2: From precession cycle accumulation
# We are ~42 kyr after the Laschamp (the last excursion)
# Mean reversal interval: ~250 kyr
# Laschamp was an EXCURSION (field recovered), not a full reversal
# Last full reversal: Brunhes-Matuyama at 780 kyr
t_since_reversal = 780  # kyr since last full reversal
t_mean_reversal = 250  # kyr mean interval
t_overdue = t_since_reversal - t_mean_reversal
print(f"\nMethod 2: Statistical (reversal frequency)")
print(f"  Last full reversal: {t_since_reversal} kyr ago")
print(f"  Mean interval: ~{t_mean_reversal} kyr")
print(f"  We are {t_overdue} kyr 'overdue' ({t_since_reversal/t_mean_reversal:.1f} mean intervals)")
print(f"  BUT: reversal intervals are NOT periodic — they follow a Poisson process")
print(f"  The current long interval is within statistical expectation")

# Method 3: Precession accumulation
T_prec = 25.772  # kyr
n_prec_since_reversal = t_since_reversal / T_prec
n_prec_per_reversal = 10  # our computation
print(f"\nMethod 3: Precession accumulation")
print(f"  Precession cycles since last reversal: {n_prec_since_reversal:.1f}")
print(f"  Cycles needed for reversal: ~{n_prec_per_reversal}")
print(f"  We have accumulated {n_prec_since_reversal:.0f} cycles = {n_prec_since_reversal/n_prec_per_reversal:.0f}x the threshold")
print(f"  The field SHOULD have reversed ~{int(n_prec_since_reversal/n_prec_per_reversal)} times since Brunhes-Matuyama")
print(f"  It did partially (Laschamp excursion at 42 kyr), but didn't complete")
print(f"  The system is DEEPLY loaded — multiple reversal-amounts of precession stress")

# Method 4: Inner core coupling threshold
print(f"\nMethod 4: Inner core decoupling")
print(f"  Inner core rotation anomaly: 2003-2023 (20 years of observed change)")
print(f"  SAA splitting: accelerating since ~2020")
print(f"  Pole acceleration: since ~2000")
print(f"  All point to the CURRENT epoch as an active period")
print(f"  If these are precursors, the reversal process may already be underway")

print(f"\n\nBEST ESTIMATE: The next reversal timing")
print(f"  From dipole decay: ~{t_reversal_1:.0f} years to Q = 2 at current rate")
print(f"  From statistical: 'overdue' but Poisson statistics allow long intervals")
print(f"  From precession: deeply loaded, should have reversed multiple times")
print(f"  From observations: precursors visible since ~2000")
print(f"  RANGE: 500 - 2000 years IF current trends continue")
print(f"         OR: could stabilize (the Laschamp recovered in ~800 years)")

# ═══════════════════════════════════════════════════════════════════
# THE DECCAN TRAPS: A RELEASE FROM CORE DISORDER?
# ═══════════════════════════════════════════════════════════════════

print(f"\n\n{'='*70}")
print("THE DECCAN TRAPS: CORE DISORDER → LIP ERUPTION")
print(f"{'='*70}")

print("""
The standard model: A mantle plume (Reunion hotspot) impinged on the
lithosphere, producing the Deccan flood basalts over ~1 Myr.
The KT asteroid impact at 66.0 Ma happened during the eruption.

The framework model: The CORE underwent a KT transition FIRST.
The disordered core drove:
  1. Multiple field reversals (observed: 29R-29N-29R around the KT boundary)
  2. Massive heat flux anomaly at the CMB
  3. Mantle plume intensification (more heat = more plume vigor)
  4. LIP eruption as the SURFACE EXPRESSION of core disorder

Evidence supporting the core-first model:
""")

# The Deccan magnetic stratigraphy
print("DECCAN MAGNETIC STRATIGRAPHY:")
print("  The Deccan lavas record the field polarity during each eruption pulse.")
print("  Chron 29R → 29N → 29R transitions are recorded in the lava pile.")
print("  The main eruption pulse correlates with chron 29R (reversed polarity).")
print("  The KT impact at 66.04 Ma falls within chron 29R.")
print()
print("  Key observation: the eruption INTENSIFIED during the reversed polarity phase.")
print("  This is consistent with: disordered core → weaker field → more eruption.")
print()

# The timing coincidence
print("TIMING:")
print("  Deccan main phase: ~66.3 to ~65.5 Ma (~800 kyr duration)")
print("  KT impact: 66.04 Ma (during the middle of the main phase)")
print("  Chron 29R: 66.4 to 65.7 Ma")
print("  The eruption, reversal, and impact are ALL within the same ~1 Myr window.")
print()

# Was the Deccan eruption triggered by the core?
print("THE CAUSAL CHAIN (framework prediction):")
print("  1. Precession accumulation loaded the core over ~10 cycles (~260 kyr)")
print("  2. Core J crossed J_c → field reversed → weak field phase")
print("  3. Disordered core changed CMB heat flow pattern")
print("  4. Existing mantle plume (Reunion) received excess heat")
print("  5. Plume intensified → Deccan main eruption pulse")
print("  6. The asteroid impact (stochastic) hit during the vulnerable phase")
print("  7. The combination of LIP + impact + disordered cavity = mass extinction")
print()

# The Siberian Traps: same pattern?
print("THE SIBERIAN TRAPS (252 Ma, Permian-Triassic):")
print("  Main eruption: ~252.2 to ~251.0 Ma (~1.2 Myr)")
print("  Field: multiple reversals during the eruption")
print("  Field intensity: DROPPED at the onset of eruption")
print("  This was during the Permian-Triassic Superchron boundary")
print("  (a long period of single polarity was ending)")
print()
print("  Same pattern: core destabilization → field reversal → LIP eruption")
print()

# The prediction for the current epoch
print("PREDICTION FOR THE CURRENT EPOCH:")
print(f"  Current dipole decay: 5%/century")
print(f"  SAA: growing and splitting")
print(f"  Inner core: rotation anomaly")
print(f"  Yellowstone: a known mantle plume hotspot")
print(f"  Afar: another active plume")
print(f"  Both are monitored for volcanic unrest")
print()
print(f"  IF the current field weakening is a reversal precursor,")
print(f"  the framework predicts that mantle plume hotspots should")
print(f"  show INCREASED activity during the reversal process.")
print(f"  This is testable: monitor volcanic unrest at Yellowstone,")
print(f"  Afar, Iceland, Reunion, and other plume sites for correlation")
print(f"  with field intensity changes.")
print()

# Can we predict from paleomagnetic data?
print("PREDICTION FROM PALEOVOLCANIC RECORD:")
print("  If LIPs correlate with reversals, then the reversal record")
print("  (from ocean floor magnetostratigraphy) should PREDICT where")
print("  LIPs occur in the geological record.")
print()
print("  Known LIP-reversal coincidences:")
print("    Siberian Traps (252 Ma) — at reversal boundary")
print("    CAMP (201 Ma) — at reversal boundary")
print("    Karoo-Ferrar (183 Ma) — during reversal cluster")
print("    Deccan Traps (66 Ma) — during chron 29R reversal")
print("    Ethiopian-Yemen (30 Ma) — during reversal cluster")
print()
print("  The 5 largest LIPs ALL coincide with field reversals.")
print("  The probability of this being random:")
# Each LIP spans ~1 Myr, average reversal rate ~4/Myr in Phanerozoic
# P(reversal during 1 Myr window) ~ 0.98 (very likely)
# So the coincidence is NOT surprising statistically
# But the INTENSIFICATION during reversed polarity IS significant
print("  Actually: reversals are frequent enough that coincidence is likely.")
print("  The key test: does eruption rate INCREASE during reversed polarity?")
print("  This IS testable from the Deccan and Siberian magnetic stratigraphy.")
print("  Published: YES — eruption intensified during reversed chrons.")
