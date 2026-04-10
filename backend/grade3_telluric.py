#!/usr/bin/env python3
"""
Grade-3 Telluric Current Analysis
====================================
The {J_telluric, B_earth}₃ anti-commutator at fault zones.

The Jelly Ball model (Paper XXV) showed that solar-telluric coupling
modulates pore pressure at ~130 Pa — enough to trigger faults near
J_c. That analysis used the even grades (0, 2).

The grade-3 channel adds the PSEUDOSCALAR:
  {J_telluric, B_earth}₃ = grade-3 projection of (JB + BJ)

where J is the telluric current density (grade 1, a vector) and
B is the geomagnetic field (grade 2, a bivector). The anti-commutator
of a vector and a bivector produces grades 1 and 3:
  {v, B} = vB + Bv → grade 1 + grade 3

The grade-3 component is the pseudoscalar: the HELICITY of the
current-field system. It is nonzero whenever J has a component
along the magnetic field direction — i.e., whenever the telluric
current is not perpendicular to B.

Physical consequences:
1. The grade-3 content modulates the CHIRALITY of the electrokinetic
   response in pore fluid. Left-handed and right-handed mineral
   structures respond differently to the pseudoscalar forcing.
2. Quartz (SiO₂) is CHIRAL — it comes in left-handed (L) and
   right-handed (D) forms. The grade-3 field preferentially
   activates one handedness over the other through the CISS effect.
3. This creates a direction-dependent pore pressure response:
   the SAME telluric current produces DIFFERENT pore pressure
   changes depending on whether the quartz is L or D — because
   the pseudoscalar coupling to chiral pore structure is sign-dependent.

Prediction: faults in L-quartz vs D-quartz host rock should show
different seismic responses to the same solar-telluric forcing.
The grade-3 channel introduces chirality into earthquake triggering.
"""

import numpy as np
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PI = np.pi


def telluric_grade3():
    """Compute grade-3 content of telluric currents at major fault zones."""

    print("=" * 70)
    print("GRADE-3 TELLURIC ANALYSIS: {J, B}₃ AT FAULT ZONES")
    print("=" * 70)

    # Physical parameters
    B_surface = 50e-6       # T (typical surface field)
    sigma_crust = 0.01      # S/m (crustal conductivity)
    sigma_seawater = 4.0    # S/m (ocean conductivity)

    # Telluric current from different sources
    print("\n--- Telluric current sources ---")

    sources = {
        "Quiet Sq (diurnal)":     {"E": 0.01, "desc": "Daily Sq variation, ~10 mV/km"},
        "Storm (Kp=5)":           {"E": 1.3,  "desc": "Geomagnetic storm, ~1.3 V/km"},
        "Severe storm (Kp=9)":    {"E": 10.0, "desc": "Carrington-class, ~10 V/km"},
        "Ocean current (Gulf)":   {"E": 0.05, "desc": "v×B for v=1 m/s in 50 μT"},
        "Lightning return stroke": {"E": 1000, "desc": "~30 kA over ~1 km², brief"},
    }

    print(f"\n{'Source':30s} {'E (V/km)':>10s} {'J_crust':>12s} {'J_ocean':>12s} {'P_pore (Pa)':>12s}")
    print("-" * 80)

    for name, src in sources.items():
        E = src["E"]  # V/km = mV/m
        E_si = E * 1e-3  # V/m

        J_crust = sigma_crust * E_si  # A/m²
        J_ocean = sigma_seawater * E_si  # A/m²

        # Pore pressure from electrokinetic effect
        # ΔP = (ε ζ / (η σ_f)) × E  (Helmholtz-Smoluchowski)
        # For typical rock: ε=80ε₀, ζ=-50mV, η=1e-3 Pa·s, σ_f=0.01 S/m
        epsilon = 80 * 8.854e-12
        zeta = -50e-3  # V
        eta = 1e-3  # Pa·s
        sigma_f = 0.01  # S/m
        dP = abs(epsilon * zeta / (eta * sigma_f)) * E_si

        print(f"{name:30s} {E:10.3f} {J_crust:12.2e} A/m² {J_ocean:12.2e} A/m² {dP:12.1f}")

    # Grade-3 computation
    print("\n\n--- Grade-3 pseudoscalar {J, B}₃ ---")
    print("""
    In Cl(3,0), the anti-commutator of a vector J and a bivector B:
      {J, B} = JB + BJ
    decomposes as:
      {J, B} = 2⟨JB⟩₁ + 2⟨JB⟩₃

    The grade-3 component ⟨JB⟩₃ = J · B* where B* is the
    Hodge dual of B (a vector in 3D). In component form:
      ⟨JB⟩₃ = J_x B_yz + J_y B_zx + J_z B_xy

    This is the DOT PRODUCT of J with the vector dual of B —
    the same as J · B_vector = J_x Bx + J_y By + J_z Bz.

    So the grade-3 content is simply J · B (in the usual sense).
    """)

    # Fault zone analysis
    print("--- Major fault zones: {J, B}₃ content ---\n")

    faults = [
        {"name": "San Andreas (Parkfield)",
         "lat": 36, "lon": -120,
         "B_inc": 61, "B_dec": 14, "B_mag": 48000,  # nT
         "J_dir": "NW-SE (along fault)", "J_angle_to_B": 47,
         "rock": "L+D quartz (mixed granodiorite)"},

        {"name": "Cascadia Subduction",
         "lat": 46, "lon": -124,
         "B_inc": 68, "B_dec": 17, "B_mag": 54000,
         "J_dir": "E-W (ocean telluric)", "J_angle_to_B": 72,
         "rock": "Serpentinite (chiral chain silicate)"},

        {"name": "Japan Trench",
         "lat": 38, "lon": 143,
         "B_inc": 50, "B_dec": -7, "B_mag": 46000,
         "J_dir": "NW-SE (Pacific plate motion)", "J_angle_to_B": 57,
         "rock": "Basalt + pelagic sediment"},

        {"name": "Dead Sea Transform",
         "lat": 31, "lon": 35,
         "B_inc": 43, "B_dec": 3, "B_mag": 44000,
         "J_dir": "N-S (along rift)", "J_angle_to_B": 47,
         "rock": "Limestone (chiral calcite crystals)"},

        {"name": "Mid-Atlantic Ridge",
         "lat": 30, "lon": -42,
         "B_inc": 40, "B_dec": -15, "B_mag": 38000,
         "J_dir": "E-W (plate divergence)", "J_angle_to_B": 50,
         "rock": "Olivine + serpentinite"},

        {"name": "East African Rift",
         "lat": -3, "lon": 36,
         "B_inc": -25, "B_dec": -1, "B_mag": 32000,
         "J_dir": "N-S (along rift)", "J_angle_to_B": 65,
         "rock": "Alkaline volcanics"},
    ]

    # Storm-level telluric
    E_storm = 1.3e-3  # V/m
    J_storm = sigma_crust * E_storm  # A/m²

    print(f"{'Fault Zone':30s} {'|B| nT':>8s} {'Inc°':>5s} {'sin(θ_JB)':>10s} "
          f"{'|{J,B}₃| nT·A/m²':>18s} {'Rock chirality':>20s}")
    print("-" * 100)

    for f in faults:
        theta = np.radians(f["J_angle_to_B"])
        B_T = f["B_mag"] * 1e-9  # Tesla
        # {J, B}₃ = |J||B|cos(angle) for the pseudoscalar (J·B_dual)
        # Actually: {J, B}₃ = |J||B|sin(angle) when J and B_dual are non-aligned
        # For a vector and bivector: the g3 part is |J||B|×(component along B's normal)
        # Simplify: g3 ∝ J × B × cos(angle between J and B-direction)
        cos_angle = np.cos(theta)
        g3 = J_storm * B_T * abs(cos_angle)

        print(f"{f['name']:30s} {f['B_mag']:8.0f} {f['B_inc']:+5.0f} "
              f"{np.sin(theta):10.3f} {g3:18.2e} {f['rock']:>20s}")

    # The chirality connection
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  THE CHIRALITY CONNECTION                                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Quartz (SiO₂) is the most abundant mineral in the crust.   ║
    ║  It comes in two chiral forms: L-quartz and D-quartz.        ║
    ║  They have OPPOSITE piezoelectric responses.                 ║
    ║                                                              ║
    ║  The grade-3 field {J, B}₃ is a PSEUDOSCALAR.               ║
    ║  A pseudoscalar couples differently to L and D structures:   ║
    ║    L-quartz: {J, B}₃ × d_piezo > 0  (enhances pressure)    ║
    ║    D-quartz: {J, B}₃ × d_piezo < 0  (reduces pressure)     ║
    ║                                                              ║
    ║  This means the SAME telluric storm produces OPPOSITE        ║
    ║  pore pressure changes in L-quartz vs D-quartz host rock.   ║
    ║                                                              ║
    ║  The even-grade (Paper XXV) analysis averages over chirality ║
    ║  and gets the net effect (~130 Pa). The grade-3 analysis     ║
    ║  shows this is the SUM of a positive and negative response   ║
    ║  that partially cancel.                                      ║
    ║                                                              ║
    ║  In a fault zone with predominantly L-quartz gouge:          ║
    ║    ΔP = ΔP_even + ΔP_odd  (enhanced, up to 2× the average) ║
    ║  In a fault zone with predominantly D-quartz gouge:          ║
    ║    ΔP = ΔP_even - ΔP_odd  (reduced, potentially zero)       ║
    ║                                                              ║
    ║  PREDICTION: The solar-seismic correlation from Paper XXV    ║
    ║  should be STRONGER in fault zones hosted in L-quartz-rich   ║
    ║  rock and WEAKER in D-quartz-rich rock. The chirality of     ║
    ║  the host rock modulates the coupling efficiency.            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Ocean telluric grade-3
    print("\n--- Ocean telluric grade-3 ---\n")
    print("Ocean eddies (Paper XXXIX Section 4) generate telluric currents")
    print("J = σ(v × B) in conducting seawater.\n")

    v_gulf = 1.0    # m/s (Gulf Stream)
    B = 50e-6        # T
    J_ocean = sigma_seawater * v_gulf * B  # A/m²
    g3_ocean = J_ocean * B  # The {J, B}₃ product

    print(f"Gulf Stream: v={v_gulf} m/s, σ={sigma_seawater} S/m")
    print(f"  J_telluric = σ(v×B) = {J_ocean:.2e} A/m²")
    print(f"  {{J, B}}₃ = {g3_ocean:.2e} T·A/m² (pseudoscalar density)")
    print(f"  Over a 100 km × 1 km eddy: total grade-3 = {g3_ocean * 1e5 * 1e3:.2e} T·A·m")

    # Lightning
    print("\n\n--- Lightning grade-3 ---\n")
    I_lightning = 30000  # A
    channel_length = 5000  # m
    A_channel = PI * (1)**2  # m² (1 m radius channel)
    J_lightning = I_lightning / A_channel  # A/m²
    g3_lightning = J_lightning * B

    print(f"Return stroke: I={I_lightning/1000:.0f} kA, channel radius ~1 m")
    print(f"  J = {J_lightning:.0f} A/m²")
    print(f"  {{J, B}}₃ = {g3_lightning:.2e} T·A/m² (ENORMOUS but transient)")
    print(f"  Duration: ~1 ms")
    print(f"  The sferic pulse carries this grade-3 content globally.")
    print(f"  Prediction: sferic circular polarization ∝ J·B·cos(θ)")
    print(f"  where θ = angle between channel and local B direction.")

    # Latitude dependence
    print("\n\n--- Latitude dependence of grade-3 coupling ---\n")
    print("The {J, B}₃ pseudoscalar depends on how much of B is")
    print("ALONG the current direction. For vertical lightning in B:\n")
    print(f"{'Latitude':>10s} {'B_inc':>8s} {'cos(inc)':>10s} {'Coupling':>12s}")
    print("-" * 45)
    for lat in [-60, -30, 0, 30, 60, 90]:
        # Approximate inclination from dipole: tan(I) = 2*tan(lat)
        inc = np.degrees(np.arctan(2 * np.tan(np.radians(lat))))
        # Vertical lightning in tilted B: component along channel ∝ sin(inc)
        coupling = abs(np.sin(np.radians(inc)))
        print(f"{lat:+10.0f}° {inc:+8.1f}° {np.cos(np.radians(inc)):10.3f} {coupling:12.3f}")

    print("""
    Maximum grade-3 coupling for vertical lightning at the POLES
    (where B is vertical = parallel to the current).
    Zero at the magnetic equator (B horizontal ⊥ vertical current).

    But lightning is mostly in the TROPICS (0-30° latitude).
    The grade-3 coupling for tropical lightning is moderate
    (coupling ≈ 0.3-0.7), not zero and not maximum.

    The STRONGEST grade-3 lightning would be at high latitudes
    where B is nearly vertical — consistent with the observation
    that high-latitude lightning (sprites, blue jets) shows unusual
    electromagnetic properties that tropical lightning does not.
    """)


if __name__ == "__main__":
    telluric_grade3()
