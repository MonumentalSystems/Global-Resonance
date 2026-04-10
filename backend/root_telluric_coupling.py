#!/usr/bin/env python3
"""
Root Current → Telluric Coupling: The Biological Antenna
==========================================================
Root tips generate ~1 μA each (Bose, 1920s; measured by Jaffe 1979,
Weisenseel et al. 1979, many since). The current flows INTO the tip
and OUT along the root surface — an electric dipole.

The question: what is the COLLECTIVE effect of billions of root tips
across a landscape? And how does this interact with:
  - Telluric currents in the soil (σ_soil × E_storm)
  - Magnetic anomalies (Bayan Obo, Kursk, etc.)
  - The grade-3 field {J_root, B}₃
  - The CISS effect (plants are CHIRAL — L-amino acids, D-sugars)

The answer turns out to be startling: root current density in
vegetated soil can EXCEED telluric storm current density.
The biosphere is not a passive passenger — it is an active
electromagnetic participant.
"""

import numpy as np
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PI = np.pi
MU0 = 4 * PI * 1e-7


# ═══════════════════════════════════════════════════════════════════════
# ROOT CURRENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

# Individual root tip current (measured values from literature)
I_ROOT_TIP_A = 1e-6    # 1 μA per root tip (Jaffe 1979, Weisenseel 1979)
I_ROOT_TIP_MAX = 10e-6  # up to 10 μA for vigorous tips (Behrens et al. 1982)

# Root tip densities by ecosystem
ECOSYSTEMS = {
    "Tropical rainforest": {
        "root_tips_per_m2": 5000,    # dense fine root network
        "root_depth_m": 2.0,         # active root zone
        "soil_sigma_Sm": 0.05,       # moist tropical soil
        "area_fraction": 0.10,       # fraction of land surface
        "biomass_t_ha": 300,
        "notes": "Fine root turnover ~1 yr, mycorrhizal network adds connectivity",
    },
    "Temperate deciduous forest": {
        "root_tips_per_m2": 3000,
        "root_depth_m": 1.5,
        "soil_sigma_Sm": 0.03,
        "area_fraction": 0.06,
        "biomass_t_ha": 150,
        "notes": "Seasonal: root currents peak in growing season",
    },
    "Boreal forest": {
        "root_tips_per_m2": 2000,
        "root_depth_m": 0.8,
        "soil_sigma_Sm": 0.02,
        "area_fraction": 0.12,
        "biomass_t_ha": 80,
        "notes": "Shallow root zone, permafrost limits depth",
    },
    "Grassland/steppe": {
        "root_tips_per_m2": 8000,    # grasses have VERY dense root tips
        "root_depth_m": 1.0,
        "soil_sigma_Sm": 0.02,
        "area_fraction": 0.10,
        "biomass_t_ha": 20,          # above-ground, but roots are 3-5× more
        "notes": "Root:shoot ratio ~4:1. Enormous underground current network.",
    },
    "Cropland": {
        "root_tips_per_m2": 4000,
        "root_depth_m": 0.5,
        "soil_sigma_Sm": 0.04,
        "area_fraction": 0.10,
        "biomass_t_ha": 10,
        "notes": "Annual plowing disrupts root network. Seasonal.",
    },
    "Desert scrub": {
        "root_tips_per_m2": 200,
        "root_depth_m": 3.0,         # deep taproots
        "soil_sigma_Sm": 0.005,      # dry soil = low conductivity
        "area_fraction": 0.15,
        "biomass_t_ha": 5,
        "notes": "Sparse but DEEP roots. Low soil σ amplifies E-field effect.",
    },
    "Wetland/mangrove": {
        "root_tips_per_m2": 6000,
        "root_depth_m": 0.5,
        "soil_sigma_Sm": 0.2,        # saline, very conductive
        "area_fraction": 0.02,
        "biomass_t_ha": 100,
        "notes": "High σ soil shorts out root dipoles. But pneumatophores create "
                 "vertical current paths from air to saline water.",
    },
}

# Comparison: telluric current densities
TELLURIC_J = {
    "Quiet Sq (diurnal)":     0.01e-3 * 0.01,    # E=10 mV/km, σ=0.01 → 1e-7 A/m²
    "Storm (Kp=5)":           1.3e-3 * 0.01,      # E=1.3 V/km → 1.3e-5 A/m²
    "Severe storm (Kp=9)":    10e-3 * 0.01,        # E=10 V/km → 1e-4 A/m²
    "Ocean (Gulf Stream)":    4.0 * 1.0 * 50e-6,   # σ×v×B → 2e-4 A/m²
}


def compute_root_current_density(ecosystem):
    """
    Compute the volumetric current density from root tips.

    Each root tip is a current dipole: I flows in at the tip
    and out along the proximal root surface over ~1 cm.

    The VOLUME-AVERAGED current density:
      J_root = (N_tips/m² × I_tip) / depth
              = collective current per unit cross-section

    This is directly comparable to J_telluric = σ_soil × E_storm.
    """
    N = ecosystem["root_tips_per_m2"]
    d = ecosystem["root_depth_m"]

    # Each tip drives I through a local volume of ~1 cm³
    # But the return current spreads over the root surface (~10 cm²)
    # The NET effect averaged over the root zone volume:
    J_root = N * I_ROOT_TIP_A / d  # A/m² (averaged over root zone depth)

    # The root current is not unidirectional — it's a dipole field
    # But neighboring dipoles partially align due to gravitropism
    # (roots grow DOWN, so the current IN is at the bottom, OUT on the sides)
    # Coherence factor: ~30% alignment gives net vertical current
    coherence = 0.3
    J_net_vertical = J_root * coherence

    return {
        "J_root_total": J_root,
        "J_net_vertical": J_net_vertical,
        "total_current_per_m2": N * I_ROOT_TIP_A,
        "equivalent_E_field": J_net_vertical / ecosystem["soil_sigma_Sm"],
    }


def root_telluric_comparison():
    """Compare root currents to telluric currents."""
    print("=" * 80)
    print("  ROOT CURRENT vs TELLURIC CURRENT: Who dominates the soil?")
    print("=" * 80)

    # Telluric reference
    J_quiet = TELLURIC_J["Quiet Sq (diurnal)"]
    J_storm = TELLURIC_J["Storm (Kp=5)"]
    J_severe = TELLURIC_J["Severe storm (Kp=9)"]

    print(f"\n  Telluric current densities in upper crust (σ=0.01 S/m):")
    for name, J in TELLURIC_J.items():
        print(f"    {name:30s}  J = {J:.2e} A/m²")

    print(f"\n\n  Root current densities by ecosystem:")
    print(f"  {'Ecosystem':30s} {'tips/m²':>8s} {'J_root':>12s} {'J_net':>12s} {'J/J_storm':>10s} {'E_equiv':>12s}")
    print("  " + "-" * 90)

    global_total_A = 0
    for name, eco in ECOSYSTEMS.items():
        rc = compute_root_current_density(eco)
        ratio = rc["J_net_vertical"] / J_storm if J_storm > 0 else 0

        print(f"  {name:30s} {eco['root_tips_per_m2']:8d} "
              f"{rc['J_root_total']:12.2e} {rc['J_net_vertical']:12.2e} "
              f"{ratio:10.1f}× {rc['equivalent_E_field']*1e3:10.1f} mV/m")

        # Global contribution
        land_area_m2 = 1.5e14  # 150 million km²
        this_area = land_area_m2 * eco["area_fraction"]
        global_total_A += rc["total_current_per_m2"] * this_area

    print(f"\n  Global total root current: {global_total_A:.2e} A")
    print(f"  (Compare: total Sq ionospheric current ≈ 10⁵ A)")
    print(f"  (Compare: single lightning stroke ≈ 3×10⁴ A)")

    return global_total_A


def grade3_root_field():
    """
    The grade-3 coupling of root currents with Earth's magnetic field.

    {J_root, B_earth}₃ = pseudoscalar

    Plants are CHIRAL:
      - L-amino acids (proteins)
      - D-sugars (cellulose, starch)
      - Right-handed DNA helix
      - Left-handed collagen-like structures in cell walls

    The CISS effect applies: electron transport through the chiral
    root tissue is SPIN-POLARIZED. The spin polarization interacts
    with the local B field through {J, B}₃.

    This means:
    1. Root growth direction couples to the local grade-3 field
    2. Over magnetic anomalies, root patterns should be different
    3. The root network acts as a BIOLOGICAL DETECTOR of {J, B}₃
    """
    print("\n" + "=" * 80)
    print("  GRADE-3: ROOT CURRENTS × EARTH'S FIELD → BIOLOGICAL PSEUDOSCALAR")
    print("=" * 80)

    print("""
  The anti-commutator {J_root, B_earth}₃ produces a pseudoscalar:

    J_root (grade 1, vector: into tip, out along root)
    B_earth (grade 2, bivector: magnetic field)
    {J, B} = JB + BJ → grade 1 + GRADE 3

  The grade-3 component:
    {J, B}₃ = |J||B|cos(θ_JB) × e₁₂₃

  For a vertical root tip in Earth's field with inclination I:
    cos(θ) = sin(I)  (vertical J dotted with field along inclination)
    {J, B}₃ = J_tip × B × sin(I)
    """)

    print(f"  {'Latitude':>10s} {'B_inc':>8s} {'sin(I)':>8s} "
          f"{'{{J,B}}₃ single':>14s} {'{{J,B}}₃ /m²':>14s} {'Coupling':>10s}")
    print("  " + "-" * 70)

    B = 50e-6  # T
    J_tip = I_ROOT_TIP_A  # 1 μA from single tip
    N_tips = 5000  # per m² (forest)

    for lat in [-60, -30, 0, 30, 45, 60, 75]:
        inc = np.degrees(np.arctan(2 * np.tan(np.radians(lat))))
        sin_inc = np.sin(np.radians(inc))

        g3_single = J_tip * B * abs(sin_inc)
        g3_per_m2 = N_tips * J_tip * B * abs(sin_inc) * 0.3  # 30% coherence

        # Coupling strength relative to equator (normalize)
        coupling = abs(sin_inc)

        print(f"  {lat:+10.0f}° {inc:+8.1f}° {sin_inc:8.3f} "
              f"{g3_single:14.2e} {g3_per_m2:14.2e} {coupling:10.3f}")

    print("""
  Maximum {J_root, B}₃ coupling at HIGH LATITUDES (vertical B ∥ vertical root).
  Zero at the magnetic equator (B horizontal ⊥ vertical root).

  This predicts LATITUDE-DEPENDENT root electrotropism:
    - Polar/subpolar: strong electromagnetic component to root growth
    - Tropical: gravitropism and hydrotropism dominate
    - Mid-latitude: intermediate — testable transition zone
    """)


def root_anomaly_interaction():
    """
    How do root currents interact with magnetic anomalies?

    Over Bayan Obo: B_local = 55000 + 1500 nT = 56500 nT
    The grade-3 field is enhanced by 2.7%.
    But the GRADIENT is enhanced by 43,000× (from the anomaly profile).

    Electrotropism: roots follow E-field gradients.
    Over a conductive ore body, telluric currents are focused,
    creating E-field gradients that roots can follow.

    This creates a FEEDBACK LOOP:
    1. Ore body focuses telluric current → E-field gradient
    2. Roots follow E-field gradient → grow toward conductive body
    3. Root network extends the conductive zone (root tissue σ ~ 0.1 S/m)
    4. Extended conductive zone captures more telluric current
    5. → More roots grow toward the anomaly
    """
    print("\n" + "=" * 80)
    print("  ROOT-ANOMALY FEEDBACK: Plants as Biogeoprospecting Indicators")
    print("=" * 80)

    print("""
  The feedback loop:

    ┌─────────────────────────────────────────────────────┐
    │  Ore body (σ_ore >> σ_soil)                         │
    │     ↓                                               │
    │  Telluric current focused → E-field gradient        │
    │     ↓                                               │
    │  Root electrotropism → roots grow toward ore body   │
    │     ↓                                               │
    │  Root network (σ ~ 0.05-0.1 S/m) extends conductor  │
    │     ↓                                               │
    │  Larger effective conductive zone                    │
    │     ↓                                               │
    │  More telluric focusing → stronger E gradient       │
    │     ↓                                               │
    │  MORE roots grow toward anomaly (positive feedback) │
    └─────────────────────────────────────────────────────┘

  This IS the mechanism behind biogeochemical prospecting:
    - "Indicator plants" over ore deposits (Cannon 1960, Brooks 1983)
    - Enhanced vegetation over kimberlite pipes (diamond exploration)
    - Anomalous plant chemistry over Cu, Ni, Zn deposits
    - Previously attributed solely to CHEMICAL uptake
    - But electrotropism provides a PHYSICAL mechanism:
      roots grow toward the ore because of the E-field, not
      (only) because of the dissolved metals
    """)

    # Quantitative comparison
    print(f"  E-field gradient at the edge of a conductive ore body:")
    print(f"  (Kp=5 storm, E_surface = 1.3 mV/m)")
    print(f"\n  {'Deposit':25s} {'σ_ore':>8s} {'σ_soil':>8s} {'E_inside':>10s} "
          f"{'E_outside':>10s} {'Gradient':>12s} {'Root sense?':>12s}")
    print("  " + "-" * 90)

    deposits = [
        ("Bayan Obo", 0.1, 0.005, "Desert soil"),
        ("Kiruna", 1.0, 0.02, "Boreal forest soil"),
        ("Bushveld", 0.3, 0.03, "Savanna soil"),
        ("Kursk", 0.5, 0.03, "Chernozem"),
        ("Palabora", 0.08, 0.02, "Subtropical soil"),
        ("Typical sulfide vein", 0.5, 0.01, "Any"),
    ]

    E_surface = 1.3e-3  # V/m (Kp=5)
    root_threshold = 1e-4  # V/m — root electrotropism threshold (Ishikawa & Evans 1990)

    for name, sigma_ore, sigma_soil, soil_type in deposits:
        # Inside ore: current concentrates, E decreases (more conductive)
        # At boundary: E-field has a discontinuity
        # Gradient ≈ (E_outside - E_inside) / boundary_width
        E_inside = E_surface * sigma_soil / sigma_ore  # current conservation
        E_outside = E_surface
        boundary_width = 10  # meters (transition zone)
        gradient = (E_outside - E_inside) / boundary_width  # V/m²

        # Can roots sense this?
        sensible = E_outside > root_threshold

        print(f"  {name:25s} {sigma_ore:8.3f} {sigma_soil:8.3f} "
              f"{E_inside*1e3:9.2f} mV/m {E_outside*1e3:9.2f} mV/m "
              f"{gradient*1e6:10.1f} μV/m² {'YES' if sensible else 'no':>12s}")

    print(f"""
  Root electrotropism threshold: ~0.1 mV/m (Ishikawa & Evans 1990)
  Storm telluric E-field: 1.3 mV/m
  → Roots can sense storm-level telluric fields!

  During a geomagnetic storm, the root network is ACTIVELY RESPONDING
  to the telluric current. Roots don't just passively conduct —
  they GROW toward the current source.

  Over a magnetic anomaly, the enhanced {'{J,B}'}₃ pseudoscalar
  additionally biases the CHIRALITY of root growth:
    - L-amino acid roots in N hemisphere (B·Ω > 0): one growth pattern
    - Same roots in S hemisphere (B·Ω < 0): subtly different pattern
    - Over magnetic anomalies: locally modified growth chirality
    """)


def collective_electromagnetic_budget():
    """
    The full electromagnetic budget of a vegetated landscape.
    """
    print("\n" + "=" * 80)
    print("  COLLECTIVE ELECTROMAGNETIC BUDGET: Forest as Antenna Array")
    print("=" * 80)

    print("""
  A mature temperate forest (1 km² = 10⁶ m²):

  ROOT CURRENT ARRAY:
  """)

    # Forest parameters
    area_m2 = 1e6  # 1 km²
    N_tips_per_m2 = 3000
    depth_m = 1.5
    total_tips = N_tips_per_m2 * area_m2
    total_I = total_tips * I_ROOT_TIP_A

    J_root = N_tips_per_m2 * I_ROOT_TIP_A / depth_m
    J_coherent = J_root * 0.3  # 30% gravitropic alignment

    B = 50e-6  # T
    sigma_soil = 0.03  # S/m

    # Magnetic field from root current array
    # Sheet current K = J × depth → B = μ₀K/2
    K = J_coherent * depth_m  # A/m (surface current density)
    B_root = MU0 * K / 2  # Tesla

    # Compare to telluric
    E_storm = 1.3e-3
    J_telluric = sigma_soil * E_storm

    # Grade-3 densities
    g3_root = J_coherent * B * 0.7  # sin(inc) at mid-lat
    g3_telluric = J_telluric * B

    # Self-field: does the root array's own magnetic field matter?
    # B_root vs B_earth
    B_root_nT = B_root * 1e9

    print(f"    Total root tips:       {total_tips:.2e} ({N_tips_per_m2}/m²)")
    print(f"    Total root current:    {total_I:.0f} A ({total_I*1e6:.0f} μA × {total_tips:.0e} tips)")
    print(f"    Root current density:  J_root = {J_root:.2e} A/m² (total)")
    print(f"                           J_net  = {J_coherent:.2e} A/m² (30% coherent)")
    print(f"    Equivalent E-field:    {J_coherent/sigma_soil*1e3:.1f} mV/m")
    print(f"    Root self-field:       B_root = {B_root_nT:.4f} nT")
    print(f"")
    print(f"  TELLURIC CURRENT (Kp=5 storm):")
    print(f"    J_telluric:            {J_telluric:.2e} A/m²")
    print(f"    E_storm:               {E_storm*1e3:.1f} mV/m")
    print(f"")
    print(f"  COMPARISON:")
    print(f"    J_root / J_telluric =  {J_coherent/J_telluric:.1f}×")
    print(f"    (Root currents are {J_coherent/J_telluric:.0f}× LARGER than storm telluric)")
    print(f"")
    print(f"  GRADE-3 COUPLING:")
    print(f"    {{J_root, B}}₃:         {g3_root:.2e} T·A/m²")
    print(f"    {{J_telluric, B}}₃:     {g3_telluric:.2e} T·A/m²")
    print(f"    Ratio:                {g3_root/g3_telluric:.1f}×")
    print(f"    Root grade-3 coupling DOMINATES in vegetated soil.")

    print(f"""
  THE FOREST AS PHASED ARRAY:

  The root network is not random — it has structure:
    - Gravitropism aligns ~30% of root current vertically
    - Hydrotropism creates lateral coherence (toward water)
    - Electrotropism creates feedback (toward telluric sources)
    - Mycorrhizal network (fungal hyphae) electrically connects trees

  A forest is a massive, slowly-reconfiguring PHASED ANTENNA ARRAY
  embedded in the upper 2 meters of soil. The array:
    - Generates {J_root/J_coherent*1e3:.0f} mA/m² of current density
    - Produces grade-3 pseudoscalar coupling to Earth's field
    - Responds to geomagnetic storms (electrotropism threshold < storm E)
    - Self-organizes around conductive anomalies (ore bodies)
    - Has CHIRALITY (L-amino acids → CISS-active)

  The iron thread extends into biology:
    Fe in hemoglobin → O₂ transport (animals)
    Fe in ferredoxin → electron transport (photosynthesis)
    Fe in cytochrome → mitochondrial current (all eukaryotes)
    Fe in magnetite → magnetoreception (bacteria, birds, bees)
    Fe in soil magnetite → root electrotropism target

  The same element, the same coupling, from mineral to organism.
    """)

    # Diurnal variation
    print(f"  DIURNAL CYCLE:")
    print(f"  Root currents follow a ~24-hour cycle (sap flow, photosynthesis)")
    print(f"  Telluric Sq current follows a ~24-hour cycle (ionospheric dynamo)")
    print(f"  Both are driven by the Sun. They are phase-locked.")
    print(f"")
    print(f"  During daytime (high sap flow + Sq maximum):")
    print(f"    J_root increases (active transport, photosynthesis)")
    print(f"    J_telluric increases (Sq dynamo peaks at local noon)")
    print(f"    {{J_root + J_telluric, B}}₃ is MAXIMUM at midday")
    print(f"")
    print(f"  During nighttime (low sap flow + Sq minimum):")
    print(f"    J_root decreases (maintenance metabolism only)")
    print(f"    J_telluric decreases (Sq dynamo quiet)")
    print(f"    {{J_total, B}}₃ is MINIMUM at midnight")
    print(f"")
    print(f"  The grade-3 field of a forest breathes with a 24-hour period,")
    print(f"  phase-locked to the Sun through two independent channels")
    print(f"  (photosynthesis + ionospheric Sq current).")

    return {
        "J_root_coherent": J_coherent,
        "J_telluric_storm": J_telluric,
        "ratio": J_coherent / J_telluric,
        "g3_root": g3_root,
        "g3_telluric": g3_telluric,
    }


# ═══════════════════════════════════════════════════════════════════════
# PORE PRESSURE: Root Water Absorption vs Electrokinetic Telluric
# ═══════════════════════════════════════════════════════════════════════

def pore_pressure_budget():
    """
    Root water uptake dominates the pore pressure budget in the rhizosphere.

    The Jelly Ball model (Paper XXV) showed that telluric currents produce
    ~130 Pa of electrokinetic pore pressure change — enough to trigger
    faults near J_c. But how does this compare to ROOT SUCTION?

    A tree transpires by pulling water through the soil pore network.
    The suction (matric potential) at the root surface can reach
    -1.5 MPa (permanent wilting point) — that's 1,500,000 Pa of
    NEGATIVE pore pressure. Six orders of magnitude larger than
    the telluric electrokinetic effect.

    But the key is not the magnitude — it's the MODULATION.
    The telluric effect is small but FAST (minutes to hours).
    The root suction is large but SLOW (hours to days).
    The question is: does root water uptake modulate the soil's
    SENSITIVITY to telluric perturbations?

    Answer: YES, through three mechanisms:
    1. Desaturation: roots remove water → air enters pores →
       partially saturated soil is MORE sensitive to pressure changes
    2. Electrokinetic amplification: drier soil has lower σ →
       same telluric E produces less J but MORE voltage per pore
    3. Suction cycling: diurnal transpiration creates a 24-hour
       pore pressure oscillation that pre-stresses the soil matrix
    """
    print("\n" + "=" * 80)
    print("  PORE PRESSURE: Root Water Uptake × Telluric Electrokinetics")
    print("=" * 80)

    # ─── Water budget by ecosystem ────────────────────────────────────

    WATER_BUDGET = {
        "Tropical rainforest": {
            "ET_mm_day": 5.0,      # evapotranspiration
            "rain_mm_day": 6.5,    # mean precipitation
            "tree_liters_day": 400, # mature canopy tree
            "trees_per_ha": 400,
            "root_depth_m": 2.0,
            "porosity": 0.45,
            "field_capacity": 0.35, # volumetric water content at FC
            "wilting_point": 0.15,
        },
        "Temperate deciduous": {
            "ET_mm_day": 3.5,
            "rain_mm_day": 2.5,
            "tree_liters_day": 200,
            "trees_per_ha": 300,
            "root_depth_m": 1.5,
            "porosity": 0.42,
            "field_capacity": 0.30,
            "wilting_point": 0.12,
        },
        "Boreal forest": {
            "ET_mm_day": 2.0,
            "rain_mm_day": 1.8,
            "tree_liters_day": 80,
            "trees_per_ha": 600,
            "root_depth_m": 0.8,
            "porosity": 0.50,      # organic-rich
            "field_capacity": 0.40,
            "wilting_point": 0.15,
        },
        "Grassland": {
            "ET_mm_day": 3.0,
            "rain_mm_day": 1.5,
            "tree_liters_day": 0,
            "trees_per_ha": 0,
            "root_depth_m": 1.0,
            "porosity": 0.40,
            "field_capacity": 0.28,
            "wilting_point": 0.10,
        },
    }

    print(f"""
  The pore pressure story has three time scales:

    FAST (seconds-minutes):
      Telluric electrokinetic:    ΔP ~ 130 Pa (Kp=5 storm)
      Lightning transient:        ΔP ~ 10 Pa (single sferic)
      Seismic wave:               ΔP ~ 1-100 Pa (passing surface wave)

    MEDIUM (hours):
      Diurnal transpiration:      ΔP ~ 10,000-50,000 Pa
      Tidal pore pressure:        ΔP ~ 1,000 Pa (body tide)
      Barometric pumping:         ΔP ~ 1,000 Pa (weather fronts)

    SLOW (days-seasons):
      Wet-dry cycling:            ΔP ~ 100,000-1,500,000 Pa
      Seasonal water table:       ΔP ~ 10,000-100,000 Pa
      Freeze-thaw:                ΔP ~ extreme (ice lens growth)

  The telluric effect (130 Pa) looks negligible next to root suction
  (50,000 Pa diurnal). But they operate on different pore populations:
    - Root suction acts on LARGE pores first (capillary drainage)
    - Electrokinetic acts on ALL pores simultaneously (body force)
    - The root-drained soil has the small pores STILL FULL
    - These small pores are the ones that control FAULT STRENGTH
    """)

    # ─── Diurnal transpiration cycle ──────────────────────────────────

    print(f"  DIURNAL TRANSPIRATION PORE PRESSURE CYCLE:")
    print(f"  {'Ecosystem':25s} {'ET mm/day':>10s} {'ΔP_diurnal':>12s} {'ΔP_telluric':>12s} "
          f"{'Ratio':>8s} {'Pore drain':>12s}")
    print("  " + "-" * 85)

    for name, wb in WATER_BUDGET.items():
        # ET in mm/day = liters/m²/day
        # This water comes from the pore space in the root zone
        # ΔP from draining: P = -ρgΔh (suction) ≈ -γ/r (capillary)
        # For a typical soil: ΔΨ ≈ ΔET × ρg / (θ × depth)
        # where θ is water content change

        ET_m_s = wb["ET_mm_day"] / (1000 * 86400)  # m/s
        depth = wb["root_depth_m"]
        porosity = wb["porosity"]
        FC = wb["field_capacity"]
        WP = wb["wilting_point"]

        # Available water: FC - WP (can be extracted by roots)
        available_water = (FC - WP) * depth * 1000  # mm

        # Diurnal extraction: ET_mm_day out of available water
        daily_fraction = wb["ET_mm_day"] / max(available_water, 1)

        # Pore pressure change from daily transpiration
        # Matric potential curve (van Genuchten):
        #   At FC: Ψ ≈ -10 kPa (-100 cm water)
        #   At WP: Ψ ≈ -1500 kPa (-15,000 cm)
        # Daily swing: about 10-50 kPa depending on ET/storage ratio
        psi_FC = -10e3    # Pa (field capacity)
        psi_WP = -1500e3  # Pa (wilting point)

        # Linearize: daily ΔΨ ≈ daily_fraction × (Ψ_WP - Ψ_FC)
        delta_psi_daily = daily_fraction * abs(psi_WP - psi_FC)
        # But most of the day the soil is near FC, so diurnal swing is smaller
        # Typical diurnal ΔΨ ≈ 10-50 kPa (Irvine et al. 1998, Anderegg 2012)
        delta_psi_diurnal = min(delta_psi_daily, 50e3)  # cap at 50 kPa

        # Telluric for comparison
        delta_P_telluric = 130  # Pa (from jelly_ball.py)

        ratio = delta_psi_diurnal / delta_P_telluric

        # How much of pore space is drained daily?
        drain_pct = daily_fraction * 100

        print(f"  {name:25s} {wb['ET_mm_day']:10.1f} "
              f"{delta_psi_diurnal/1e3:10.1f} kPa "
              f"{delta_P_telluric/1e3:10.4f} kPa "
              f"{ratio:8.0f}× {drain_pct:10.1f}%")

    # ─── The sensitivity amplification mechanism ──────────────────────

    print(f"""
  ROOT SUCTION AMPLIFIES TELLURIC SENSITIVITY:

  The key insight is that root water extraction creates PARTIAL SATURATION.
  In a partially saturated soil:

    1. AIR-WATER INTERFACES form in pores
       Surface tension: γ = 0.072 N/m
       In a 10 μm pore: capillary pressure = 2γ/r = 14,400 Pa
       A small ADDITIONAL pressure change (like 130 Pa telluric)
       can push this interface past a pore throat → snap-through
       → sudden pore filling or drainage event → stress change

    2. CONDUCTIVITY DROPS dramatically
       Saturated soil: σ ≈ 0.01-0.05 S/m
       At field capacity: σ ≈ 0.005-0.02 S/m
       At 50% saturation: σ ≈ 0.001-0.005 S/m
       → Same E-field produces LESS current but MORE voltage per pore
       → Electrokinetic effect per pore is AMPLIFIED

    3. MENISCUS STRESS concentrates at grain contacts
       Partially saturated soil has meniscus bridges between grains
       These bridges carry additional compressive stress
       Small pressure changes can rupture meniscus bridges
       → Sudden effective stress change → potential micro-fracture

  The mechanism is THRESHOLD-SENSITIVE, not linear:
    Root suction brings the pore system NEAR a critical state
    (air-entry value, meniscus rupture, snap-through)
    and the telluric perturbation provides the final push.
    """)

    # ─── Quantitative: pore-scale pressure balance ────────────────────

    print(f"  PORE-SCALE PRESSURE BALANCE AT THE RHIZOSPHERE:")
    print(f"  (Root surface, partially saturated, mid-afternoon)")
    print()

    # A single pore at the root surface
    pore_radii = [100e-6, 50e-6, 10e-6, 5e-6, 1e-6]  # meters
    gamma = 0.072  # N/m surface tension

    print(f"  {'Pore radius':>12s} {'P_capillary':>12s} {'P_root_suct':>12s} "
          f"{'P_telluric':>12s} {'P_tidal':>10s} {'Δθ from EK':>12s}")
    print("  " + "-" * 80)

    P_root = 30e3  # Pa suction (mid-day, mid-soil)
    P_telluric = 130  # Pa
    P_tidal = 1000  # Pa

    for r in pore_radii:
        P_cap = 2 * gamma / r  # capillary entry pressure
        # Is this pore drained by root suction?
        drained = "DRAINED" if P_root > P_cap else "full"
        # If the pore is near the drainage threshold, telluric can tip it
        margin = abs(P_cap - P_root)
        tippable = margin < P_telluric

        print(f"  {r*1e6:10.0f} μm {P_cap/1e3:11.1f} kPa {P_root/1e3:11.1f} kPa "
              f"{P_telluric/1e3:11.4f} kPa {P_tidal/1e3:9.2f} kPa "
              f"{'← TIPPABLE' if tippable else drained:>12s}")

    print(f"""
  Pores near the AIR-ENTRY THRESHOLD are tippable by telluric pressure.

  At mid-afternoon with P_root = 30 kPa suction:
    - Pores > ~5 μm are already drained (P_cap < P_root)
    - Pores < ~5 μm are still full (P_cap > P_root)
    - Pores at EXACTLY ~5 μm are at the threshold
    - For these pores, the 130 Pa telluric push matters

  The root system acts as a TUNER:
    - Morning (wet): threshold at ~100 μm pores (large, insensitive)
    - Afternoon (dry): threshold at ~5 μm pores (small, sensitive)
    - Night (recovery): threshold shifts back up

  The DIURNAL CYCLE OF ROOT SUCTION sweeps the sensitivity window
  across the pore size distribution. At some point during each day,
  the critical pore size aligns with the telluric perturbation scale.

  This is the coupling: roots SET the operating point, telluric
  provides the perturbation, and the pore network responds nonlinearly.
    """)

    # ─── Electroosmotic flow in root-modified pores ───────────────────

    print(f"  ELECTROOSMOTIC FLOW IN ROOT-MODIFIED PORES:")
    print()

    print(f"""
  Root exudates modify the pore surface chemistry:
    - Organic acids (citric, malic, oxalic) change ζ-potential
    - Mucilage (polysaccharides) increases surface charge
    - Root-associated bacteria form biofilms with charged surfaces
    - Mycorrhizal hyphae (fungi) create new conductive pathways

  The Helmholtz-Smoluchowski electrokinetic coupling coefficient:
    C_ek = -εζ/(ησ_f)

  Modified by root exudates:
    ε (permittivity):  unchanged (~80ε₀)
    ζ (zeta potential): INCREASED by organic acids (-50 → -80 mV)
    η (viscosity):      INCREASED by mucilage (1 → 2-5 mPa·s)
    σ_f (fluid σ):      INCREASED by dissolved organics (0.01 → 0.05 S/m)

  Net effect on C_ek:
    Bare soil:     C_ek = 80×8.85e-12 × 0.050 / (0.001 × 0.01) = 3.5e-6 m²/(V·s)
    Rhizosphere:   C_ek = 80×8.85e-12 × 0.080 / (0.003 × 0.05) = 3.8e-7 m²/(V·s)
    """)

    bare_C = 80 * 8.854e-12 * 0.050 / (0.001 * 0.01)
    rhizo_C = 80 * 8.854e-12 * 0.080 / (0.003 * 0.05)

    print(f"    Bare soil C_ek:       {bare_C:.2e} m²/(V·s)")
    print(f"    Rhizosphere C_ek:     {rhizo_C:.2e} m²/(V·s)")
    print(f"    Ratio:                {rhizo_C/bare_C:.2f}×")
    print(f"    The rhizosphere REDUCES the direct EK coupling coefficient.")
    print(f"    BUT: the partial saturation AMPLIFIES the pressure sensitivity.")
    print(f"    Net effect: the two partially cancel, but the THRESHOLD mechanism")
    print(f"    (snap-through at critical pore size) remains active.")

    # ─── Streaming potential: root flow generates its own E-field ─────

    print(f"""

  STREAMING POTENTIAL: Root Water Uptake → Electric Field

  The reverse electrokinetic effect: when water flows through charged
  pores (driven by root suction), it carries ions along → streaming
  current → streaming potential (Sill 1983, Revil et al. 1999).

    E_streaming = -C_ek × ΔP / σ_f

  For root suction flow:""")

    delta_P_root = 30e3  # Pa (mid-day suction)
    sigma_f = 0.02       # S/m
    C_ek_soil = 3.5e-6   # m²/(V·s)

    # Streaming potential from root suction
    E_streaming = C_ek_soil * delta_P_root / sigma_f
    J_streaming = sigma_f * E_streaming  # A/m² (this drives current in the opposite direction)

    print(f"    Root suction:     ΔP = {delta_P_root/1e3:.0f} kPa")
    print(f"    Streaming E:      {E_streaming*1e3:.1f} mV/m")
    print(f"    Streaming J:      {J_streaming:.2e} A/m²")
    print(f"    Telluric J:       {0.03 * 1.3e-3:.2e} A/m²")
    print(f"    ROOT STREAMING J / TELLURIC J = {J_streaming / (0.03 * 1.3e-3):.1f}×")

    print(f"""
  Root water uptake generates a STREAMING POTENTIAL that drives
  its own electric current through the soil. This current is
  {J_streaming / (0.03 * 1.3e-3):.0f}× larger than storm telluric current.

  The streaming current has the SAME coupling to B as telluric:
    {{J_streaming, B}}₃ is a pseudoscalar, just like {{J_telluric, B}}₃

  The root system generates THREE types of soil current:
    1. Root tip current:    ~1 μA/tip × 3000 tips/m² = biological
    2. Streaming current:   ~{J_streaming:.1e} A/m² = electrokinetic from suction
    3. Exudate redox:       Fe²⁺/Fe³⁺ cycling in rhizosphere = geochemical

  All three couple to B. All three carry chirality (L-amino acid roots).
  Together they make the rhizosphere the most electromagnetically
  active zone in the upper crust.
    """)

    return {
        "J_streaming": J_streaming,
        "E_streaming": E_streaming,
        "delta_psi_diurnal_Pa": delta_psi_diurnal,
    }


# ═══════════════════════════════════════════════════════════════════════
# THE VERTICAL PUMP: Roots as Parallel Pore-Forcing Wires
# ═══════════════════════════════════════════════════════════════════════

def vertical_pump_model():
    """
    Each root is a vertical suction tube pulling water upward through
    the surrounding charged pore network. The water flows TOWARD the
    root surface from all directions, then UP through the root to the
    xylem and out through the leaves.

    This is NOT a diffuse, random process. It is:
      - VERTICAL: water moves upward against gravity (transpiration pull)
      - PARALLEL: millions of roots all pull in the same direction
      - COHERENT: all driven by the same solar forcing (evaporation)
      - THROUGH CHARGED PORES: generating streaming current

    Each root is a wire of upward fluid flow. The streaming current
    generated by this flow is also vertical. Millions of parallel
    vertical current sources, all phase-locked to the Sun.

    This is geometrically identical to a bundle of wires carrying
    current upward — the return current flows diffusely down through
    the bulk soil as rain percolates.
    """
    print("\n" + "=" * 80)
    print("  THE VERTICAL PUMP: Roots as Parallel Pore-Forcing Current Sources")
    print("=" * 80)

    # ─── Single root as a suction tube ────────────────────────────────

    # A fine root (~1 mm diameter) extracts water from surrounding soil
    # The extraction zone (rhizosphere) extends ~5-10 mm from root surface
    # Water flows radially inward toward the root, then vertically up

    root_radius = 0.5e-3     # m (0.5 mm fine root radius)
    rhizosphere_radius = 5e-3  # m (5 mm influence zone)
    root_length = 0.1         # m (10 cm typical fine root segment)

    # Transpiration per root segment
    # A tree with 400 L/day and 1 km of fine roots:
    total_fine_root_length_m = 1000  # 1 km of fine roots (conservative)
    Q_tree = 400e-3 / 86400   # m³/s (400 L/day)
    Q_per_m = Q_tree / total_fine_root_length_m  # m³/s per m of root

    # Flow velocity at root surface (radial inward)
    A_root_surface = 2 * PI * root_radius * root_length  # m²
    v_root_surface = Q_per_m * root_length / A_root_surface  # m/s

    # Flow velocity at rhizosphere boundary (slower, larger area)
    A_rhizo = 2 * PI * rhizosphere_radius * root_length
    v_rhizo = Q_per_m * root_length / A_rhizo

    print(f"""
  A SINGLE FINE ROOT as suction tube:

    Root radius:          {root_radius*1e3:.1f} mm
    Rhizosphere radius:   {rhizosphere_radius*1e3:.0f} mm
    Extraction rate:      {Q_per_m*1e9:.2f} nL/s per mm of root

    Flow velocity at root surface:       {v_root_surface*1e6:.1f} μm/s
    Flow velocity at rhizosphere edge:   {v_rhizo*1e6:.1f} μm/s

  The water flows INWARD radially, then UPWARD through the root.
  In the pore network, this creates a convergent flow field —
  like a drain in a bathtub, but pulling sideways and then up.
    """)

    # ─── Streaming current per root ───────────────────────────────────

    # Streaming current from radial flow toward root
    # J_streaming = C_ek × ∇P (where ∇P is the pressure gradient)

    epsilon = 80 * 8.854e-12  # F/m
    zeta = -50e-3             # V (zeta potential)
    eta = 1e-3                # Pa·s
    sigma_f = 0.02            # S/m

    C_ek = epsilon * abs(zeta) / (eta * sigma_f)

    # Pressure gradient near root: ΔP/Δr ≈ Ψ_root / rhizosphere_radius
    psi_root = 30e3  # Pa (midday suction)
    grad_P = psi_root / rhizosphere_radius  # Pa/m

    # Streaming E-field at the root surface
    E_streaming_local = C_ek * grad_P / sigma_f
    J_streaming_local = sigma_f * E_streaming_local

    # BUT: the KEY is the VERTICAL component
    # Water enters radially but must travel UPWARD through the root zone
    # The net vertical flow rate is Q_tree / root_zone_area

    root_zone_area = PI * 15**2  # m² (15 m radius root zone for a tree)
    root_zone_depth = 1.5  # m

    # Darcy flux (vertical, upward, volume-averaged)
    q_vertical = Q_tree / root_zone_area  # m/s (Darcy velocity)

    # Streaming current from vertical flow
    # In the soil matrix, vertical flow through charged pores generates
    # a vertical streaming current
    # J_z = C_ek × ΔP_vertical / (σ_f × depth)

    # The vertical pressure gradient comes from the suction profile:
    # ΔP/Δz ≈ ρg + Ψ_root/depth (gravity + suction)
    rho_w = 1000  # kg/m³
    g = 9.81      # m/s²
    grad_P_vertical = rho_w * g + psi_root / root_zone_depth  # Pa/m

    E_streaming_vertical = C_ek * grad_P_vertical / sigma_f
    J_streaming_vertical = sigma_f * E_streaming_vertical

    print(f"  STREAMING CURRENT FROM VERTICAL WATER FLOW:")
    print(f"  (Water pulled upward through charged pore network)")
    print()
    print(f"    Darcy flux (upward): {q_vertical*1e6:.2f} μm/s  ({q_vertical*86400*1e3:.1f} mm/day)")
    print(f"    Vertical ∇P:        {grad_P_vertical:.0f} Pa/m (gravity + suction)")
    print(f"    Streaming E (vert): {E_streaming_vertical*1e3:.1f} mV/m")
    print(f"    Streaming J (vert): {J_streaming_vertical:.2e} A/m²")
    print(f"    Telluric J (Kp=5):  {0.03 * 1.3e-3:.2e} A/m²")
    print(f"    Ratio:              {J_streaming_vertical / (0.03 * 1.3e-3):.0f}×")

    # ─── Scale to the forest ──────────────────────────────────────────

    print(f"\n\n  SCALING: Forest as parallel vertical pump array")
    print("  " + "-" * 60)

    trees_per_ha = 300
    trees_per_km2 = trees_per_ha * 100
    total_Q = trees_per_km2 * Q_tree  # m³/s per km²
    total_Q_mm_day = total_Q / 1e6 * 86400 * 1e3  # mm/day over 1 km²

    # Average vertical Darcy flux over 1 km²
    q_forest = total_Q / 1e6  # m/s (1e6 m² per km²)
    J_forest_streaming = sigma_f * C_ek * (rho_w * g + psi_root / root_zone_depth) / sigma_f

    # Number of "wires" (root segments acting as individual pumps)
    N_roots_per_m2 = 3000  # fine root tips per m²
    # Each fine root is a ~10 cm long vertical pump
    wire_spacing = 1 / np.sqrt(N_roots_per_m2)  # m between roots

    print(f"""
    Trees per km²:         {trees_per_km2:,.0f}
    Total transpiration:   {total_Q_mm_day:.1f} mm/day ({total_Q*1e3:.1f} L/s per km²)
    Average Darcy flux:    {q_forest*1e6:.2f} μm/s (upward through soil)

    Root "wires":          {N_roots_per_m2} per m² (fine roots)
    Wire spacing:          {wire_spacing*1e3:.0f} mm apart
    Wire length:           ~{root_zone_depth*100:.0f} cm (root zone depth)
    Wire current:          each root segment drives ~{J_streaming_vertical * PI * rhizosphere_radius**2:.2e} A
                           of streaming current through its surrounding pore shell

    This is a PHASED ARRAY of {N_roots_per_m2} parallel vertical current sources
    per m², spaced {wire_spacing*1e3:.0f} mm apart, all driving current UPWARD,
    all synchronized to the Sun.
    """)

    # ─── The vertical current geometry and B field ────────────────────

    B_earth = 50e-6  # T
    inc_deg = 63  # mid-latitude inclination
    inc = np.radians(inc_deg)

    # Grade-3 coupling of vertical streaming current with B
    # J is vertical (upward), B has vertical component = B sin(I)
    # {J, B}₃ = J × B × cos(angle between J and B)
    # For vertical J and B at inclination I: angle = (90-I)
    # cos(90-I) = sin(I)
    g3_streaming = J_streaming_vertical * B_earth * np.sin(inc)

    # Compare to telluric grade-3
    J_telluric = 0.03 * 1.3e-3
    g3_telluric = J_telluric * B_earth * 0.5  # telluric is mostly horizontal

    print(f"  GRADE-3 COUPLING OF THE VERTICAL PUMP:")
    print("  " + "-" * 60)
    print(f"""
    J_streaming is VERTICAL (upward).
    B_earth at {inc_deg}° inclination has vertical component = B×sin(I).

    The vertical streaming current is OPTIMALLY ALIGNED with B
    at high latitudes — the same geometry as the root tip current.

    {{J_stream_vert, B}}₃ = J × B × sin(I)
                         = {J_streaming_vertical:.2e} × {B_earth:.0e} × sin({inc_deg}°)
                         = {g3_streaming:.2e} T·A/m²

    {{J_telluric, B}}₃    = {g3_telluric:.2e} T·A/m²  (telluric is mostly horizontal)

    Ratio: {g3_streaming/g3_telluric:.0f}×

    The vertical pump's grade-3 coupling dominates because:
      1. The current is LARGER ({J_streaming_vertical/J_telluric:.0f}× more current)
      2. The current is VERTICAL (aligned with B at high latitudes)
      3. Telluric current is mostly HORIZONTAL (poor alignment with B)

    The forest floor is not receiving a telluric signal through noise.
    The forest IS the signal. The biological vertical pump generates
    the dominant pseudoscalar field in the upper crust.
    """)

    # ─── The circuit: up through roots, down through rain ─────────────
    # The return leg is NOT diffuse — the forest makes its own rain.

    print(f"  THE COMPLETE CIRCUIT: The Forest Makes Its Own Rain")
    print("  " + "-" * 60)

    # Moisture recycling parameters
    amazon_area_km2 = 5.5e6
    amazon_ET_mm_day = 4.5       # transpiration
    amazon_rain_mm_day = 6.5     # total rainfall
    recycled_fraction = 0.50     # ~50% of Amazon rain is recycled transpiration
    stemflow_fraction = 0.15     # fraction of rain channeled down trunks
    drippoint_fraction = 0.35    # fraction concentrated at canopy drip points
    diffuse_fraction = 1 - stemflow_fraction - drippoint_fraction  # only ~50% diffuse

    print(f"""
    The Amazon "flying rivers" (Nobre 2014, Makarieva & Gorshkov 2007):
      Total Amazon rainfall:      {amazon_rain_mm_day:.1f} mm/day
      From transpiration:         {amazon_rain_mm_day * recycled_fraction:.1f} mm/day ({recycled_fraction*100:.0f}% recycled)
      From ocean evaporation:     {amazon_rain_mm_day * (1-recycled_fraction):.1f} mm/day

    The forest pumps water UP through roots, releases it from leaves,
    it forms clouds, and rains back down — often within 100-200 km.
    The forest IS the rain machine. Remove the forest → rainfall collapses.

    But the DOWNWARD leg also has structure:

    CANOPY REDISTRIBUTION:
      Stemflow (down the trunk):   {stemflow_fraction*100:.0f}% of rain → channeled into root zone
      Drip points (canopy edges):  {drippoint_fraction*100:.0f}% of rain → concentrated at canopy margin
      Diffuse throughfall:         {diffuse_fraction*100:.0f}% → relatively uniform

    Only {diffuse_fraction*100:.0f}% of rainfall reaches the ground diffusely.
    The rest is STRUCTURED by the tree architecture.
    """)

    # ─── Both legs are tree-structured ─────────────────────────────────

    # Stemflow: 15% of rain channeled down the trunk
    # A tree with 30 cm trunk diameter receives stemflow from ~50 m² of canopy
    # That flow converges to a ~1 m² zone around the trunk base
    canopy_area = PI * 10**2  # m² per tree (10 m crown radius)
    trunk_area = PI * 0.3**2  # m² trunk cross-section
    stemflow_concentration = canopy_area * stemflow_fraction / trunk_area

    # Stemflow velocity into the root zone
    stemflow_rate = amazon_rain_mm_day * stemflow_fraction / 1000 / 86400  # m/s over canopy
    stemflow_velocity = stemflow_rate * canopy_area / trunk_area  # m/s at trunk base

    # Streaming current from stemflow (downward, concentrated at trunk)
    psi_gravity = 1000 * 9.81 * 1.5  # Pa (1.5 m of water column)
    C_ek = 80 * 8.854e-12 * 0.05 / (1e-3 * 0.02)
    J_stemflow_down = 0.02 * C_ek * psi_gravity / 0.02 / root_zone_depth

    # Drip points: 35% concentrated at canopy edge
    drip_area_fraction = 0.10  # drip points cover ~10% of ground area
    drippoint_concentration = drippoint_fraction / drip_area_fraction

    print(f"  STEMFLOW AS STRUCTURED RETURN CURRENT:")
    print("  " + "-" * 60)
    print(f"    Canopy area per tree:     {canopy_area:.0f} m²")
    print(f"    Stemflow concentration:   {stemflow_concentration:.0f}× (canopy → trunk base)")
    print(f"    Stemflow velocity:        {stemflow_velocity*1e6:.0f} μm/s at trunk")
    print(f"    J_stemflow (downward):    {J_stemflow_down:.2e} A/m²")

    print(f"""
    Drip point concentration:   {drippoint_concentration:.1f}× at canopy edges
    Diffuse throughfall:        {diffuse_fraction*100:.0f}% of rain (the only truly diffuse part)

  THE CIRCUIT IS CLOSED AND STRUCTURED ON BOTH LEGS:

    ┌─── UPWARD LEG (3000 root wires per m²) ─────────────────┐
    │  Soil pores → root surface (radial convergence)          │
    │  → Xylem → Trunk → Leaves → Atmosphere                  │
    │  STREAMING CURRENT: {J_streaming_vertical:.2e} A/m² (UPWARD)       │
    │  Geometry: 3000 parallel wires, 18 mm spacing            │
    ├─── ATMOSPHERE (flying rivers, Hadley cell) ──────────────┤
    │  Transpiration → clouds → 100-200 km transport           │
    │  The forest makes ~50% of its own rain                   │
    ├─── DOWNWARD LEG (tree-structured) ───────────────────────┤
    │  Stemflow: {stemflow_fraction*100:.0f}% → concentrated at trunk ({stemflow_concentration:.0f}×)    │
    │  Drip points: {drippoint_fraction*100:.0f}% → concentrated at canopy edge ({drippoint_concentration:.1f}×)│
    │  Diffuse: only {diffuse_fraction*100:.0f}%                                  │
    │  STREAMING CURRENT: {J_stemflow_down:.2e} A/m² (DOWNWARD, at trunk) │
    └──────────────────────────────────────────────────────────┘

  The net current is NOT zero because:
    - Upward leg: distributed over 1,256 m² root zone, 1.5 m deep
    - Downward stemflow: concentrated at {trunk_area:.2f} m² trunk base
    - The upward current has ~{canopy_area/trunk_area:.0f}× less concentration but
      ~{N_roots_per_m2}× more parallel paths
    - The spatial asymmetry creates a NET CIRCULATION of streaming current:
      broadly UP in the root zone, narrowly DOWN at trunks

  This is an ELECTROKINETIC CONVECTION CELL driven by solar evaporation.
  The forest is not a passive element in the water cycle.
  It is a SELF-ORGANIZED ELECTROMAGNETIC PUMP that:
    1. Creates its own rain (moisture recycling)
    2. Structures the return flow (stemflow, drip points)
    3. Drives streaming current through charged pores
    4. Couples to B through the grade-3 pseudoscalar
    5. Recycles with a period of ~hours to days

  The Sun drives the pump. The forest IS the circuit.
    """)

    # ─── Grade-3 of the complete circuit ──────────────────────────────

    B_earth = 50e-6  # T
    inc_deg = 63  # mid-latitude inclination
    inc = np.radians(inc_deg)

    # Grade-3 coupling of vertical streaming current with B
    g3_streaming = J_streaming_vertical * B_earth * np.sin(inc)

    # The downward stemflow also couples — but OPPOSITE sign (downward)
    g3_stemflow = J_stemflow_down * B_earth * np.sin(inc)

    # Net grade-3 from the circulation
    g3_net = g3_streaming - g3_stemflow  # upward dominates (more area)

    # Compare to telluric grade-3
    J_telluric = 0.03 * 1.3e-3
    g3_telluric = J_telluric * B_earth * 0.5  # telluric is mostly horizontal

    print(f"  GRADE-3 BUDGET OF THE COMPLETE CIRCUIT:")
    print("  " + "-" * 60)
    print(f"    {{J_up, B}}₃ (root pump):     {g3_streaming:.2e} T·A/m² (upward, over root zone)")
    print(f"    {{J_down, B}}₃ (stemflow):     {g3_stemflow:.2e} T·A/m² (downward, at trunks)")
    print(f"    Net {{J, B}}₃ circulation:     {g3_net:.2e} T·A/m²")
    print(f"    {{J_telluric, B}}₃:            {g3_telluric:.2e} T·A/m²")
    print(f"    Ratio (net circuit / telluric): {g3_net/g3_telluric:.0f}×")

    print(f"""
  Even accounting for the structured return flow, the net grade-3
  from the forest circuit exceeds telluric by ~{g3_net/g3_telluric:.0f}×.

  The asymmetry is the key:
    UP:   distributed (1,256 m² per tree), MANY roots, modest J
    DOWN: concentrated (0.28 m² at trunk), FEW paths, high J
    They don't cancel because they occur at DIFFERENT LOCATIONS.

  The forest's grade-3 field has spatial TEXTURE:
    - Over the root zone: upward pseudoscalar (positive at N latitudes)
    - At trunk bases: downward pseudoscalar (negative)
    - At canopy drip lines: downward, concentrated
    - Net: the distributed upward field dominates

  A magnetotelluric survey in a forest should see this texture:
  the streaming-current signal varies on the scale of tree spacing
  (~5-10 m), with a sign reversal between root zone and trunk base.

  Deforestation destroys this entire circuit. The grade-3 texture
  collapses. The streaming current vanishes. The pore network dries.
  In the Amazon, the rainfall itself collapses (tipping point).
  The electromagnetic pump and the hydrological pump are the SAME pump.
    """)

    return {
        "J_streaming_vertical": J_streaming_vertical,
        "g3_streaming": g3_streaming,
        "g3_telluric": g3_telluric,
        "wire_spacing_mm": wire_spacing * 1e3,
        "N_roots_per_m2": N_roots_per_m2,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  THE BIOLOGICAL ANTENNA: Root Currents × Telluric Currents × Grade-3")
    print("=" * 80)

    global_I = root_telluric_comparison()
    grade3_root_field()
    root_anomaly_interaction()
    budget = collective_electromagnetic_budget()
    pore = pore_pressure_budget()
    pump = vertical_pump_model()

    print("\n" + "=" * 80)
    print("  SYNTHESIS: The Living Crust")
    print("=" * 80)
    print(f"""
  The vegetated land surface is not electromagnetically passive.
  Root currents are {budget['ratio']:.0f}× larger than storm-driven telluric currents
  in the upper 2 meters of soil.

  Three systems superpose in the soil:
    1. TELLURIC (geophysical): J = σ × E_storm     ({budget['J_telluric_storm']:.2e} A/m²)
    2. ROOT (biological):      J = N × I_tip × f   ({budget['J_root_coherent']:.2e} A/m²)
    3. ELECTROKINETIC:         J from fluid flow    (pore pressure driven)

  All three couple to B through the grade-3 anti-commutator.
  All three carry CHIRALITY information (root = L-amino, mineral = L/D quartz).

  The grade-3 field of the biosphere is not a metaphor.
  It is a measurable electromagnetic quantity:
    {{J_root, B}}₃ = {budget['g3_root']:.2e} T·A/m² per m² of forest

  Predictions:
    1. Root growth patterns should correlate with B inclination (latitude test)
    2. Vegetation anomalies over ore deposits are partly ELECTROMAGNETIC,
       not purely geochemical (test: compare conductive vs non-conductive ores)
    3. Deforestation changes the local Schumann field pattern
       (removes the biological current layer)
    4. The circadian rhythm of {{J_root, B}}₃ should be measurable with
       a buried induction coil magnetometer in a forest

  The iron thread:
    Core → dynamo → surface B → magnetite in soil → root electrotropism
    → biological current → {{J_root, B}}₃ → CISS in chiral root tissue
    → chirality-dependent growth → feedback to soil conductivity
    → modified telluric pattern → back to the iron in the soil

  The circle closes. Iron mediates the coupling at every step.
    """)


if __name__ == "__main__":
    main()
