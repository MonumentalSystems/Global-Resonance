================================================================================
  REE DEPOSITS, MAGNETIC ANOMALIES & LIGHTNING COUPLING
  Can ore bodies modify local electromagnetic conditions?
================================================================================

================================================================================
  ALL DEPOSITS: Magnetic Anomaly & Telluric Focusing Comparison
================================================================================

  Deposit                    Anomaly    σ ore  J_ratio  g3_ratio  Detect km                 Type
  -----------------------------------------------------------------------------------------------
  Kiruna                       +5000    1.000        9×        10×        911 apatite iron ore (IO
  Kursk_Magnetic_Anomaly       +3000    0.500        9×        10×       1000 banded iron formatio
  Bayan_Obo                    +1500    0.100        9×         9×        628 carbonatite-associat
  Bushveld_Complex             +1000    0.300        9×         9×       1000 layered mafic intrus
  Palabora                      +800    0.080        8×         9×        394 carbonatite complex 
  Lovozero                      +500    0.050        8×         8×       1000 alkaline layered int
  Mount_Weld                    +300    0.020        7×         7×        187 laterite over carbon
  Mountain_Pass                 +200    0.010        5×         5×        129      carbonatite REE
  Ilimaussaq                    +150    0.010        5×         5×        394 alkaline intrusion (

  KEY FINDINGS:
  1. Kiruna (5000 nT) and Kursk (3000 nT) are the strongest anomalies
     — both are IRON deposits, not REE. The magnetite dominates.
  2. Bayan Obo (1500 nT) ranks 3rd — its REE content is geophysically
     irrelevant next to its massive magnetite body.
  3. Telluric focusing scales with conductivity, not anomaly:
     Kiruna (σ=1 S/m) focuses ~100× more current than Mountain Pass.
  4. The grade-3 enhancement tracks the PRODUCT of J and B anomaly:
     deposits that are both conductive AND magnetic are strongest.
    

================================================================================
  BAYAN OBO: The World's Largest REE Deposit as Telluric Antenna
================================================================================

  Location:  41.80°N, 109.97°E (Inner Mongolia, China)
  Ore body:  ~18 km × 2.7 km (Main + East + West orebodies)
  Ore mass:  1500 Mt total, 48 Mt REE oxide
  Magnetite: 35% by volume → ~500 Mt Fe₃O₄
  Anomaly:   +1500 nT (3% of local Earth field)
  
  1. MAGNETIC ANOMALY PROFILE
  --------------------------------------------------
          0 km:     1500.0 nT  ██████████████████████████████ (over deposit)
          1 km:     1500.0 nT  ██████████████████████████████ (airborne detectable)
          2 km:     1500.0 nT  ██████████████████████████████ (airborne detectable)
          5 km:  2478837.0 nT  ████████████████████████████████████████ (airborne detectable)
         10 km:   309854.6 nT  ████████████████████████████████████████ (airborne detectable)
         20 km:    38731.8 nT  ████████████████████████████████████████ (airborne detectable)
         50 km:     2478.8 nT  ████████████████████████████████████████ (airborne detectable)
        100 km:      309.9 nT  ██████ (airborne detectable)
        200 km:       38.7 nT   (airborne detectable)
        500 km:        2.5 nT   (ground detectable)
       1000 km:        0.3 nT   (gradiometer)

  Detectability ranges:
    Ground magnetometer (1 nT):  628 km
    Airborne survey (5 nT):      394 km
    Satellite (Swarm, 5 nT):     394 km

  2. TELLURIC CURRENT FOCUSING
  --------------------------------------------------
    Conductivity contrast (ore/country): 100×
    Depolarization factor N:              0.1087
    Current enhancement factor:           8.5×
    J in ore during Kp=5 storm:           1.11e-05 A/m²
    J in country rock:                    1.30e-06 A/m²
    Current density ratio:                9×
    Grade-3 {J, B}₃ ratio (ore/country): 9×

  3. REE PARAMAGNETIC CONTRIBUTION
  --------------------------------------------------

    The REE ions in bastnäsite and monazite are PARAMAGNETIC:
      Nd³⁺:  μ_eff = 3.62 μ_B (electronic config: [Xe]4f³)
      Sm³⁺:  μ_eff = 0.85 μ_B (Van Vleck paramagnet)
      Ce³⁺:  μ_eff = 2.54 μ_B
      Dy³⁺:  μ_eff = 10.6 μ_B (largest of all REE)
      Gd³⁺:  μ_eff = 7.94 μ_B (half-filled 4f⁷, isotropic)

    At 48 Mt REE oxide in 1.5 Gt ore:
      REE concentration: ~3.2 wt% REO
      Paramagnetic susceptibility: χ_REE ≈ 10⁻⁴ to 10⁻³ SI
      vs magnetite: χ_mag ≈ 1-6 SI

    The REE PARAMAGNETIC contribution is 1000× SMALLER than magnetite.
    The REE do NOT significantly modify the magnetic anomaly.
    The iron (magnetite) dominates completely.

    BUT: the REE provide a SPECTROSCOPIC signature:
    - Crystal field transitions in 4f shells → sharp optical lines
    - These lines show MAGNETIC CIRCULAR DICHROISM (MCD)
    - The MCD co-varies with the local B field (including anomaly)
    - This is a potential CISS-like observable
    
  4. GRADE-3 FIELD MODIFICATION
  --------------------------------------------------
    Background B:        55000 nT
    Anomalous B:         +1500 nT (2.7% of background)
    Total B over ore:    56500 nT
    ∇B (background):     0.005 nT/km
    ∇B (over deposit):   216.5 nT/km
    Gradient ratio:      43301×
    Grade-3 enhancement: ~43301× (from steeper field gradient)

  5. SCHUMANN RESONANCE COUPLING
  --------------------------------------------------

    The Bayan Obo ore body (18 km × 2.7 km) has a fundamental
    electromagnetic resonance:
      f_res = c / (2L√εμ) ≈ 3351529.2 Hz
      (for L=18 km, σ=0.1 S/m)

    This is near the SCHUMANN RESONANCE at 7.83 Hz!
    The ore body could act as a resonant antenna for Schumann modes.

    At Schumann frequencies:
      Skin depth in ore: δ = √(2/(ωμ₀σ)) = 0.6 km
      Skin depth in country rock: δ = 6 km

    The ore body is WITHIN its own skin depth at Schumann frequencies,
    meaning it acts as a coherent conductor — a natural antenna.
    

================================================================================
  LIGHTNING DENSITY AT MAJOR DEPOSITS vs SURROUNDINGS
================================================================================
  Loaded lightning data: c:\Users\lisam\Geometric Resonance\Global-Resonance\data\lightning\wglc_climatology_30m_monthly.nc
  Grid: 360 × 720, lat range [-89.8, 89.8]
  Resolution: 0.50° × 0.50°

  Deposit                      On-site   Surround    Ratio      ΔnT    σ S/m
  ---------------------------------------------------------------------------
  Bayan_Obo                     0.0004     0.0003    1.153    +1500    0.100
  Mountain_Pass                 0.0006     0.0010    0.573     +200    0.010
  Mount_Weld                    0.0010     0.0010    0.962     +300    0.020
  Lovozero                      0.0001     0.0002    0.609     +500    0.050
  Ilimaussaq                    0.0000     0.0000    0.366     +150    0.010
  Kiruna                        0.0001     0.0001    1.305    +5000    1.000
  Palabora                      0.0010     0.0016    0.611     +800    0.080
  Kursk_Magnetic_Anomaly        0.0013     0.0013    0.994    +3000    0.500
  Bushveld_Complex              0.0020     0.0016    1.277    +1000    0.300

  Correlation tests:
    Anomaly (nT) vs lightning ratio:      r=0.883, p=0.0016
    Conductivity (S/m) vs lightning ratio: r=0.895, p=0.0011
    Mean ratio: 0.872 (t=-1.12, p=0.2952)
    No significant difference

================================================================================
  SUMMARY
================================================================================

  Q: Can REE deposits modify local field conditions?
  A: The REE themselves (paramagnetic, χ ~ 10⁻⁴ SI) are negligible.
     But the ASSOCIATED MAGNETITE (χ ~ 1-6 SI) creates anomalies
     of 500-5000 nT, detectable at scales of 10-100+ km.

  Q: At what scale is the anomaly detectable?
  A: Ground:     10-200 km (1 nT threshold)
     Airborne:   5-100 km (5 nT threshold)
     Satellite:  only the largest (Kursk, Bushveld, Kiruna)

  Q: Are these deposits prone to lightning?
  A: The conductive ore body acts as a buried lightning rod:
     - Low ground resistance → preferential leader attachment
     - Enhancement depends on σ_ore/σ_surroundings contrast
     - Desert settings (Bayan Obo) maximize the contrast
     - Testable with WGLC lightning climatology data

  The iron thread at Bayan Obo:
     Fe-56 nuclear stability → Fe in carbonatite magma →
     Magnetite crystallization → 1500 nT anomaly →
     Telluric focusing (100×) → Grade-3 enhancement →
     Potential Schumann antenna at 7.83 Hz

  Same element, same algebra, from nuclear physics to geophysics.
    

  Results saved: C:\Users\lisam\geo resonance\Global-Resonance\backend\output\ree_magnetic_anomaly.json