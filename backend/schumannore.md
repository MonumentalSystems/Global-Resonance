================================================================================
  SCHUMANN RESONANCE ABSORPTION BY ORE BODIES
================================================================================

  Skin depth (km) at each Schumann mode:
  Deposit                      7.8 Hz   14.1 Hz   20.3 Hz   26.4 Hz   32.5 Hz   Body/δ₁
  -------------------------------------------------------------------------------------
  Kiruna                      0.18     0.13     0.11     0.10     0.09        49.7
  Kursk_Magnetic_Anomaly      0.25     0.19     0.16     0.14     0.12      1361.9
  Bayan_Obo                   0.57     0.42     0.35     0.31     0.28        12.2
  Bushveld_Complex            0.33     0.24     0.20     0.18     0.16       776.4
  Palabora                    0.64     0.47     0.39     0.35     0.31         7.0
  Lovozero                    0.80     0.60     0.50     0.44     0.39        31.7
  Mount_Weld                  1.27     0.95     0.79     0.69     0.62         1.8
  Mountain_Pass               1.80     1.34     1.12     0.98     0.88         1.0
  Ilimaussaq                  1.80     1.34     1.12     0.98     0.88         6.5

  Deposit                     δ(7.8Hz)   Body/δ  Abs_eff  σ_abs km²    P_abs W    P/P_cav
  ------------------------------------------------------------------------------------------
  Kursk_Magnetic_Anomaly          0.25   1361.9    0.994   119276.3   1.95e+07   3.96e+02 SCATTERER
  Bushveld_Complex                0.33    776.4    0.952    61907.0   4.88e+06   9.91e+01 SCATTERER
  Kiruna                          0.18     49.7    0.996       79.7   2.00e+04   4.07e-01 SCATTERER
  Lovozero                        0.80     31.7    0.712      462.5   8.12e+03   1.65e-01 SCATTERER
  Bayan_Obo                       0.57     12.2    0.828       39.7   1.20e+03   2.44e-02 SCATTERER
  Palabora                        0.64      7.0    0.792       15.8   4.00e+02   8.13e-03 SCATTERER
  Ilimaussaq                      1.80      6.5    0.426       58.0   3.40e+02   6.91e-03 SCATTERER
  Mount_Weld                      1.27      1.8    0.544        2.7   2.50e+01   5.08e-04 absorber
  Mountain_Pass                   1.80      1.0    0.426        1.3   7.50e+00   1.52e-04

  INTERPRETATION:

  Body/δ ratio determines the interaction regime:
    Body/δ < 1:  EM penetrates through → weak interaction (transparent)
    Body/δ ~ 1:  Maximum absorption per unit volume (resonant absorber)
    Body/δ > 5:  EM can't penetrate → scattering dominates (reflector)

  At 7.83 Hz (Schumann fundamental):
    - Kiruna (σ=1 S/m): δ=0.18 km, body=9 km → Body/δ=50 → SCATTERER
      The ore body casts an EM shadow, creating a Schumann dead zone
      and secondary scattered field. Detectable with magnetotellurics.

    - Kursk (σ=0.5 S/m): δ=0.25 km, body=346 km → Body/δ=1377 → MASSIVE SCATTERER
      The Kursk BIF extends over 120,000 km² — it is a continent-scale
      perturbation to the Schumann waveguide. The cavity modes are
      genuinely modified by its presence.

    - Bayan Obo (σ=0.1 S/m): δ=0.57 km, body=7 km → Body/δ=12 → SCATTERER
      The ore body is ~12 skin depths across at 7.83 Hz.
      It should create a measurable Schumann field distortion.

  No single deposit absorbs a significant fraction of total cavity power
  (P/P_cav ~ 10⁻⁶ to 10⁻³). The Schumann resonances persist globally.
  But the LOCAL field pattern is measurably distorted over large deposits.

  TESTABLE: Deploy a pair of Schumann receivers (induction coil magnetometers):
    - One over the deposit (e.g., Bayan Obo main orebody)
    - One 50+ km away in country rock (same latitude)
    Compare: amplitude ratio, phase shift, and polarization at 7.83 Hz.
    Prediction: the deposit site shows REDUCED amplitude (absorption)
    and SHIFTED phase (scattering) at all Schumann modes.
    

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