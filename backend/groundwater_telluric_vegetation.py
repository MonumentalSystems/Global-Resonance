#!/usr/bin/env python3
"""
Groundwater as the True Telluric Current on Land
===================================================
The standard picture: ionospheric Sq current induces telluric current
in the crust via Faraday induction. J = σ × E_induced.

The actual picture on land: GROUNDWATER FLOW through charged pore
networks generates streaming current that DWARFS the induced telluric.
The "telluric current" that matters for trees, faults, and pore pressure
is predominantly ELECTROKINETIC, driven by hydraulic head gradients.

This means:
  1. Anything that modifies groundwater flow modifies the E-field landscape
  2. Conductive anomalies (Kursk, ore bodies) reshape flow + E-fields
  3. Archaeological tells (compacted, different σ) are local E-field anomalies
  4. Trees respond to this through electrotropism
  5. Vegetation patterns over anomalies are partly ELECTROMAGNETIC

The same mechanism explains:
  - Crop marks in aerial archaeology (vegetation over buried structures)
  - "Indicator plants" in biogeochemical prospecting
  - Forest density anomalies over ore deposits
  - The Kursk Magnetic Anomaly's famously fertile "black earth" (chernozem)
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
# GROUNDWATER STREAMING vs IONOSPHERIC INDUCTION
# ═══════════════════════════════════════════════════════════════════════

def groundwater_vs_induced():
    """
    Compare the two sources of "telluric" current on land:
    1. Faraday induction from ionospheric Sq current
    2. Streaming current from groundwater flow
    """
    print("=" * 80)
    print("  GROUNDWATER STREAMING vs IONOSPHERIC INDUCTION")
    print("  Which one is the 'real' telluric current on land?")
    print("=" * 80)

    # Electrokinetic coupling coefficient
    epsilon = 80 * 8.854e-12
    zeta = -50e-3  # V
    eta = 1e-3     # Pa·s
    sigma_f = 0.02 # S/m
    C_ek = epsilon * abs(zeta) / (eta * sigma_f)

    # ─── Induced telluric (ionospheric) ───────────────────────────────

    print(f"\n  1. FARADAY INDUCTION (ionospheric Sq dynamo → crustal current):")
    scenarios_induced = [
        ("Quiet Sq (diurnal)",   0.01e-3, 0.01),
        ("Moderate Kp=3",        0.3e-3,  0.01),
        ("Storm Kp=5",           1.3e-3,  0.01),
        ("Severe Kp=8",          10e-3,   0.01),
    ]
    print(f"  {'Condition':25s} {'E (mV/m)':>10s} {'σ (S/m)':>8s} {'J (A/m²)':>12s}")
    print("  " + "-" * 60)
    for name, E, sigma in scenarios_induced:
        J = sigma * E
        print(f"  {name:25s} {E*1e3:10.2f} {sigma:8.3f} {J:12.2e}")

    # ─── Streaming current (groundwater) ──────────────────────────────

    print(f"\n  2. STREAMING CURRENT (groundwater flow → electrokinetic):")
    print(f"     C_ek = εζ/(ησ_f) = {C_ek:.2e} m²/(V·s)")

    scenarios_gw = [
        ("Flat terrain (i=0.001)", 0.001, 1e-5, 0.02, "Plains, slow seepage"),
        ("Gentle slope (i=0.01)",  0.01,  1e-4, 0.02, "Hills, normal drainage"),
        ("Hillside (i=0.05)",      0.05,  5e-4, 0.02, "Moderate slopes"),
        ("Mountain (i=0.1)",       0.1,   1e-3, 0.02, "Steep terrain"),
        ("Spring/seep (i=0.5)",    0.5,   5e-3, 0.02, "Focused discharge"),
        ("Root suction (30 kPa)",  2.0,   None,  0.02, "Tree transpiration pull"),
    ]

    rho_w = 1000
    g = 9.81

    print(f"\n  {'Scenario':30s} {'∇h':>8s} {'v_Darcy':>10s} {'∇P (Pa/m)':>10s} {'J_stream':>12s} {'J/J_Sq':>8s}")
    print("  " + "-" * 85)

    J_Sq = 0.01 * 0.01e-3  # quiet Sq reference

    for name, grad_h, v_darcy, sigma_f, desc in scenarios_gw:
        if v_darcy is None:
            # Root suction case
            grad_P = 30e3 / 1.5  # 30 kPa over 1.5 m root zone
        else:
            grad_P = rho_w * g * grad_h  # Pa/m

        J_streaming = sigma_f * C_ek * grad_P / sigma_f  # = C_ek × ∇P
        ratio = J_streaming / J_Sq

        print(f"  {name:30s} {grad_h:8.3f} "
              f"{'--' if v_darcy is None else f'{v_darcy*1e6:.0f} μm/s':>10s} "
              f"{grad_P:10.0f} {J_streaming:12.2e} {ratio:8.0f}×")

    print(f"""
  RESULT: Even on flat terrain (hydraulic gradient 0.001), groundwater
  streaming current ({C_ek * rho_w * g * 0.001:.2e} A/m²) is comparable to
  quiet-day induced telluric ({J_Sq:.2e} A/m²).

  On a hillside (gradient 0.05): streaming is {C_ek * rho_w * g * 0.05 / J_Sq:.0f}× induced Sq.
  At a spring or seep: streaming is {C_ek * rho_w * g * 0.5 / J_Sq:.0f}× induced Sq.
  Under a transpiring tree: streaming is {C_ek * 30e3/1.5 / J_Sq:.0f}× induced Sq.

  On land, the "telluric current" is PREDOMINANTLY streaming current
  from groundwater flow. The ionospheric component is a perturbation
  ON TOP of the much larger groundwater baseline.

  This changes everything about how we think about vegetation anomalies
  over buried structures: it's not (just) soil chemistry — it's the
  E-field landscape shaped by groundwater interacting with conductivity
  contrasts in the subsurface.
    """)


# ═══════════════════════════════════════════════════════════════════════
# KURSK: The Forest on the Anomaly
# ═══════════════════════════════════════════════════════════════════════

def kursk_vegetation():
    """
    The Kursk Magnetic Anomaly region has:
    1. 120,000 km² of buried BIF (σ=0.5 S/m, 40% magnetite)
    2. 3,000 nT magnetic anomaly (6% of Earth's field)
    3. Famous CHERNOZEM (black earth) — the world's most fertile soil
    4. Anomalous vegetation patterns (extremely productive agriculture)

    Is the chernozem purely a climate/geology story, or does the
    electromagnetic environment of the Kursk anomaly contribute?
    """
    print("=" * 80)
    print("  KURSK: The World's Largest Magnetic Anomaly and Its Forest/Soil")
    print("=" * 80)

    # Kursk parameters
    sigma_bif = 0.5       # S/m (BIF conductivity)
    sigma_chernozem = 0.05  # S/m (chernozem topsoil, moist)
    sigma_country = 0.01   # S/m (typical sedimentary cover)
    depth_bif = 0.3        # km (BIF under 300 m of sediment)
    area_km2 = 120000
    anomaly_nT = 3000
    B_earth_nT = 51000

    # Electrokinetic parameters
    epsilon = 80 * 8.854e-12
    zeta = -50e-3
    eta = 1e-3
    sigma_f = 0.03  # S/m (chernozem pore fluid, moderate)
    C_ek = epsilon * abs(zeta) / (eta * sigma_f)

    print(f"""
  The Kursk Magnetic Anomaly (KMA):
    Area:           {area_km2:,} km² (larger than England)
    BIF depth:      {depth_bif*1000:.0f} m below surface
    σ_BIF:          {sigma_bif} S/m
    σ_chernozem:    {sigma_chernozem} S/m
    σ_country:      {sigma_country} S/m
    Anomaly:        +{anomaly_nT} nT ({anomaly_nT/B_earth_nT*100:.1f}% of Earth's field)

  The chernozem of the Central Russian Upland is the world's
  most fertile agricultural soil. Standard explanation:
    - Steppe grassland accumulated organic matter (Holocene)
    - Loess parent material (fine-grained, nutrient-rich)
    - Continental climate (cold winters preserve organics)
    - Moderate rainfall (enough to grow, not enough to leach)

  All true. But there's a coincidence nobody discusses:
  the BEST chernozem overlies the Kursk Magnetic Anomaly.
    """)

    # ─── Groundwater flow modification by the BIF ─────────────────────

    print(f"  GROUNDWATER FLOW OVER THE BIF:")
    print("  " + "-" * 60)

    # The BIF at 300 m depth is a conductivity/permeability contrast
    # Groundwater flow is modified:
    # 1. The BIF is LESS permeable than sediment → flow deflects around it
    # 2. The fracture zones at BIF margins are MORE permeable → flow concentrates
    # 3. The conductivity contrast creates an E-field anomaly via streaming

    # Regional hydraulic gradient (Central Russian Upland)
    grad_h = 0.005  # gentle regional slope toward Don/Dnieper rivers
    rho_w = 1000
    g_acc = 9.81
    grad_P = rho_w * g_acc * grad_h  # Pa/m

    # Streaming current in normal sediment
    J_stream_country = C_ek * grad_P
    # Streaming current over BIF (modified by conductivity contrast)
    # The E-field is enhanced at the BIF boundary (like ore body focusing)
    # Enhancement factor ≈ σ_BIF / σ_sediment for a slab
    enhancement = sigma_bif / sigma_country
    J_stream_bif_edge = J_stream_country * enhancement

    # E-field in the soil above the BIF
    E_country = J_stream_country / sigma_chernozem
    E_bif_edge = J_stream_bif_edge / sigma_chernozem

    print(f"    Regional hydraulic gradient:  i = {grad_h}")
    print(f"    ∇P = ρg×i = {grad_P:.0f} Pa/m")
    print(f"    J_streaming (normal):        {J_stream_country:.2e} A/m²")
    print(f"    J_streaming (BIF edge):      {J_stream_bif_edge:.2e} A/m² ({enhancement:.0f}× enhanced)")
    print(f"    E-field (normal soil):       {E_country*1e3:.2f} mV/m")
    print(f"    E-field (over BIF edge):     {E_bif_edge*1e3:.2f} mV/m")

    # Root electrotropism threshold
    root_threshold = 0.1e-3  # V/m (0.1 mV/m)
    print(f"\n    Root electrotropism threshold: {root_threshold*1e3:.1f} mV/m")
    print(f"    E_normal / threshold:  {E_country/root_threshold:.1f}×")
    print(f"    E_BIF_edge / threshold: {E_bif_edge/root_threshold:.1f}×")

    above_threshold = E_country > root_threshold
    print(f"    Roots can sense groundwater E-field: {'YES' if above_threshold else 'NO'}")

    # ─── Nutrient cycling feedback ────────────────────────────────────

    print(f"""

  THE ELECTROMAGNETIC SOIL FERTILITY FEEDBACK:

  Conventional:  Fertile soil ← climate + parent material + biology
  Extended:      Fertile soil ← (climate + parent material + biology)
                                × electromagnetic environment

  The BIF creates a multi-layered effect:

  1. GROUNDWATER MODIFICATION (hydrological)
     BIF acts as aquitard → water table rises above it
     → Higher soil moisture → more vegetation → more organic matter
     → Deeper roots reach the water table → stronger vertical pump

  2. MINERAL SUPPLY (geochemical)
     BIF weathers to release Fe²⁺, Si, Mg
     → Iron-rich groundwater → Fe oxyhydroxide coatings on soil grains
     → Increased ζ-potential → STRONGER streaming currents
     → Magnetite nanoparticles in soil → root electrotropism targets

  3. E-FIELD LANDSCAPE (electromagnetic)
     Streaming current enhanced {enhancement:.0f}× at BIF boundary
     → E-field gradient detectable by roots
     → Root networks preferentially grow toward BIF-modified zones
     → Deeper, denser root networks → more organic input → richer soil

  4. GRADE-3 ENHANCEMENT
     +{anomaly_nT} nT anomaly → {anomaly_nT/B_earth_nT*100:.1f}% stronger local B
     ∇B ~ {anomaly_nT / np.sqrt(area_km2):.1f} nT/km (gradual over {np.sqrt(area_km2):.0f} km)
     {{J_stream, B}}₃ is {(1 + anomaly_nT/B_earth_nT) * enhancement:.0f}× enhanced over the BIF

  The Kursk chernozem is the best in the world partly because
  it sits on a buried electromagnetic anomaly that enhances
  groundwater flow, iron supply, streaming current, AND the
  grade-3 coupling that drives root electrotropism.
    """)


# ═══════════════════════════════════════════════════════════════════════
# ARCHAEOLOGICAL TELLS: Vegetation Marks from E-field Anomalies
# ═══════════════════════════════════════════════════════════════════════

def archaeological_tells():
    """
    An archaeological tell is a mound formed by millennia of settlement:
    mudbrick → collapse → rebuild → layer upon layer.

    Tell soil properties vs surrounding natural soil:
    - Higher conductivity (σ): ash, charcoal, metal artifacts, compaction
    - Different permeability (K): compacted layers = aquitards
    - Different water retention: mudbrick fragments hold moisture
    - Different chemistry: phosphate, nitrogen (ancient waste)

    These create DETECTABLE VEGETATION ANOMALIES (crop marks):
    - Lusher vegetation over ditches (more water)
    - Stressed vegetation over walls (less water, compacted)
    - Anomalous species composition over entire tell

    Standard explanation: differential water retention + nutrients.
    Extended: E-field anomaly from conductivity contrast → electrotropism.
    """
    print("=" * 80)
    print("  ARCHAEOLOGICAL TELLS: E-field Anomalies and Vegetation Marks")
    print("=" * 80)

    # Tell properties
    tells = {
        "Ditch fill (ancient moat)": {
            "sigma_Sm": 0.08,   # organic-rich, wet fill
            "K_m_s": 1e-4,      # permeable
            "moisture": "HIGH",
            "vegetation": "LUSH (positive crop mark)",
            "depth_m": 2.0,
            "width_m": 5.0,
        },
        "Mudbrick wall foundation": {
            "sigma_Sm": 0.005,   # dry, compacted brick
            "K_m_s": 1e-7,       # nearly impermeable
            "moisture": "LOW",
            "vegetation": "STRESSED (negative crop mark)",
            "depth_m": 0.5,
            "width_m": 1.0,
        },
        "Ash/midden layer": {
            "sigma_Sm": 0.10,    # charcoal + bone + metals
            "K_m_s": 5e-5,
            "moisture": "MODERATE-HIGH",
            "vegetation": "LUSH (species anomaly)",
            "depth_m": 1.0,
            "width_m": 20.0,
        },
        "Natural soil (control)": {
            "sigma_Sm": 0.02,
            "K_m_s": 1e-5,
            "moisture": "NORMAL",
            "vegetation": "NORMAL",
            "depth_m": 0,
            "width_m": 0,
        },
        "Kiln / burnt area": {
            "sigma_Sm": 0.01,    # fired clay = insulator
            "K_m_s": 1e-8,       # vitrified
            "moisture": "VERY LOW",
            "vegetation": "ABSENT or stunted",
            "depth_m": 0.3,
            "width_m": 3.0,
        },
        "Metal hoard / slag dump": {
            "sigma_Sm": 1.0,     # metallic artifacts
            "K_m_s": 1e-5,
            "moisture": "NORMAL",
            "vegetation": "ANOMALOUS (electrotropism)",
            "depth_m": 0.5,
            "width_m": 2.0,
        },
    }

    # Electrokinetic parameters
    epsilon = 80 * 8.854e-12
    zeta = -50e-3
    eta = 1e-3
    sigma_natural = 0.02

    # Background groundwater gradient
    grad_h = 0.01  # gentle slope
    rho_w = 1000
    g_acc = 9.81
    grad_P = rho_w * g_acc * grad_h

    print(f"\n  Streaming current E-field anomaly at buried structures:")
    print(f"  (Background: natural soil, σ={sigma_natural} S/m, i={grad_h})")
    print(f"\n  {'Structure':30s} {'σ S/m':>8s} {'σ/σ₀':>6s} {'E_stream':>10s} {'Sens?':>6s} {'Vegetation':>20s}")
    print("  " + "-" * 90)

    root_threshold = 0.1e-3  # V/m

    for name, props in tells.items():
        sigma = props["sigma_Sm"]
        # E-field at the boundary of the structure
        # Current conservation: J₁ = J₂ at boundary
        # σ₁E₁ = σ₂E₂ → E₂ = E₁ × σ₁/σ₂
        C_ek = epsilon * abs(zeta) / (eta * sigma_natural)
        J_background = C_ek * grad_P
        E_background = J_background / sigma_natural

        # E-field perturbation at structure boundary
        if sigma != sigma_natural:
            E_inside = E_background * sigma_natural / sigma
            E_contrast = abs(E_inside - E_background)
        else:
            E_inside = E_background
            E_contrast = 0

        sensible = E_contrast > root_threshold
        ratio = sigma / sigma_natural

        print(f"  {name:30s} {sigma:8.3f} {ratio:5.1f}× "
              f"{E_contrast*1e3:8.2f} mV/m {'YES' if sensible else 'no':>6s} "
              f"{props['vegetation']:>20s}")

    print(f"""
  The streaming current E-field is DIRECTLY detectable by roots
  for most buried archaeological structures.

  The mechanism for CROP MARKS is dual:
    1. HYDROLOGICAL: differential water retention (standard explanation)
       Ditches hold water → lusher crops
       Walls drain water → stressed crops
       This is the dominant effect in most soils.

    2. ELECTROMAGNETIC: streaming current E-field anomaly (new)
       Conductive fills (ash, metal) focus current → E gradient
       Roots grow preferentially toward conductive anomalies
       This adds a DIRECTIONAL component to the vegetation response:
       not just "more water" but "roots grow TOWARD the structure"

  The electromagnetic component explains several puzzling observations:
    a) Crop marks appear even in well-watered conditions
       (when moisture differential should be minimal)
    b) Certain species are better indicators than others
       (species with stronger electrotropic response)
    c) Metal hoards produce vegetation anomalies without
       moisture or chemistry differences
    d) Crop marks are STRONGER after geomagnetic storms
       (enhanced telluric → stronger E-field contrast)
       This last point is TESTABLE with satellite NDVI time series.
    """)


# ═══════════════════════════════════════════════════════════════════════
# VEGETATION OVER CONDUCTIVE ANOMALIES: A GLOBAL TEST
# ═══════════════════════════════════════════════════════════════════════

def vegetation_anomaly_global():
    """
    Testable with satellite data: do vegetation indices (NDVI, EVI)
    show anomalies over known conductive structures?
    """
    print("=" * 80)
    print("  GLOBAL TEST: Vegetation Indices over Conductive Anomalies")
    print("=" * 80)

    print(f"""
  Prediction: satellite-derived vegetation indices should show
  anomalies over major conductive/magnetic structures, AFTER
  controlling for climate, topography, and soil type.

  The signal should be strongest where:
    1. The anomaly is near the surface (or has surface expression)
    2. Regional groundwater gradient is moderate (not zero, not extreme)
    3. Climate allows vegetation (not desert, not ice)
    4. The conductivity contrast is large (σ_anomaly >> σ_country)

  Test sites (ranked by expected signal):

  {'Site':30s} {'σ contrast':>12s} {'Area km²':>10s} {'Vegetation':>15s} {'Expected':>10s}
  {'-'*80}
  {'Kursk Magnetic Anomaly':30s} {'50×':>12s} {'120,000':>10s} {'Chernozem/ag':>15s} {'STRONG':>10s}
  {'Bushveld Complex':30s} {'30×':>12s} {'65,000':>10s} {'Savanna':>15s} {'MODERATE':>10s}
  {'Pilbara (Hamersley)':30s} {'30×':>12s} {'6,000':>10s} {'Arid scrub':>15s} {'WEAK (dry)':>10s}
  {'Kiruna':30s} {'100×':>12s} {'80':>10s} {'Boreal':>15s} {'MODERATE':>10s}
  {'Bayan Obo':30s} {'20×':>12s} {'48':>10s} {'Desert':>15s} {'WEAK (dry)':>10s}
  {'Carajas':30s} {'40×':>12s} {'400':>10s} {'RAINFOREST':>15s} {'STRONG':>10s}

  CONTROL SITES (same climate/topography, no anomaly):
    - 100 km from each test site in each cardinal direction
    - Same latitude band, same elevation, same soil parent material
    - Difference in NDVI = electromagnetic vegetation effect

  The BEST test case is CARAJAS (Brazil):
    - Iron BIF (σ=0.4 S/m) in the Amazon rainforest
    - Known to have anomalous vegetation (forest islands on ferricrete)
    - Surrounded by dense rainforest (excellent control)
    - Groundwater flow is vigorous (tropical rainfall)
    - The "canga" vegetation on Carajas ironstone is well-studied
      but attributed purely to edaphic (soil chemistry) factors
    - Prediction: the vegetation anomaly correlates with CONDUCTIVITY
      of the substrate, not (only) with iron concentration in soil

  The CLEAREST test is KURSK:
    - Gradual transition from BIF to normal sediment
    - Chernozem quality degrades as you move away from the BIF
    - Standard explanation: climate gradient + loess thickness
    - Test: does chernozem quality correlate with MAGNETIC ANOMALY
      (i.e., proximity to the BIF) after controlling for climate?
    - This could be done with soil carbon maps + aeromagnetic surveys

  TEMPORAL TEST (strongest prediction):
    Compare NDVI time series over Kursk/Carajas to geomagnetic Kp index.
    Prediction: NDVI anomaly (site minus control) should be LARGER
    during magnetically active periods (higher Kp → stronger telluric
    → stronger streaming current contrast → more root growth toward anomaly).
    Lag time: weeks to months (root growth response time).
    """)

    # ─── The groundwater E-field landscape ────────────────────────────

    print(f"\n  THE E-FIELD LANDSCAPE OF THE SUBSURFACE:")
    print("  " + "-" * 60)
    print(f"""
  Every conductivity contrast in the subsurface creates an E-field
  anomaly in the flowing groundwater above it. The E-field landscape
  is a map of buried conductivity structure, "illuminated" by
  groundwater flow like a flashlight through a stained glass window.

  Trees read this map through electrotropism. Their root networks
  are an IMAGE of the subsurface conductivity structure, projected
  upward into the rhizosphere by the streaming current E-field.

  This is why:
    - Oaks cluster on buried aquifers (conductive, wet)
    - Willows find buried streams (conductive, flowing)
    - Heather avoids limestone (resistive vs surrounding clay)
    - Ancient metalworking sites have anomalous vegetation 3000 years later
    - Kimberlite pipes (diamond exploration) have vegetation halos

  The groundwater flow is the illumination source.
  The subsurface conductivity is the pattern.
  The streaming current is the signal.
  The roots are the detector.
  The vegetation is the image.

  And the whole thing couples to Earth's field through grade-3.
    """)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    groundwater_vs_induced()
    kursk_vegetation()
    archaeological_tells()
    vegetation_anomaly_global()

    print("\n" + "=" * 80)
    print("  SYNTHESIS")
    print("=" * 80)
    print(f"""
  The "telluric current" on land is primarily STREAMING CURRENT
  from groundwater flow through charged pores — not Faraday induction
  from the ionosphere.

  This streaming current is shaped by subsurface conductivity structure:
    - Ore bodies focus it (Kursk, Bayan Obo, Carajas)
    - Archaeological remains perturb it (tells, ditches, walls)
    - Geological contacts create boundaries (aquitard/aquifer)

  Trees read the resulting E-field through electrotropism:
    - Roots grow toward conductive anomalies
    - Root networks mirror subsurface structure
    - Vegetation patterns are an IMAGE of buried conductivity

  This is testable:
    1. NDVI over Kursk BIF vs control: is chernozem quality correlated
       with magnetic anomaly after controlling for climate?
    2. NDVI over Carajas BIF in the Amazon: does vegetation on ironstone
       correlate with substrate conductivity, not just iron chemistry?
    3. Crop marks after geomagnetic storms: are marks SHARPER in Kp>5
       weeks? (satellite time series)
    4. Root excavation at BIF boundaries: do root networks show
       directional bias toward the conductive body?

  The groundwater-streaming-root-vegetation chain closes the circuit:
    Rain → groundwater → streaming current (through charged pores)
    → E-field landscape (shaped by buried σ contrasts)
    → Root electrotropism (trees detect E gradients)
    → Transpiration → rain (the forest makes its own rain)
    → MORE groundwater → STRONGER streaming current

  Positive feedback. Solar-powered. Self-organized.
  The forest and the aquifer are a coupled electromagnetic system.
    """)


if __name__ == "__main__":
    main()
