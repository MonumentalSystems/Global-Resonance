#!/usr/bin/env python3
"""
The Ice King's Electromagnetic Footprint
==========================================
Frederic Tudor (1783-1864) launched the global ice trade in 1806.
By the 1880s, the US harvested ~5-15 million tons of ice per year
from freshwater lakes and rivers.

Scale at peak (1880-1900):
  - US annual consumption: 5-15 million tons (5-15 × 10⁹ kg)
  - Hudson River alone: 4 million tons stored, 135 warehouses, 20,000 workers
  - Norway: 1 million tons/year exported
  - 90,000 workers + 25,000 horses in the US
  - Industry capitalized at $28 million ($660 million in 2010 dollars)

The question: what was the electromagnetic impact of removing
5-15 million tons of ice from lakes every winter?

Ice is an electrical INSULATOR (σ ~ 10⁻⁷ S/m).
Liquid water beneath ice is a CONDUCTOR (σ ~ 0.01-0.05 S/m for fresh).
Ice cover SHIELDS the lake from atmospheric E-fields and telluric coupling.

Removing the ice:
  1. Exposes conductive water to atmospheric E-field (Schumann, storms)
  2. Changes the lake's thermal regime (less insulation → more freezing)
  3. Modifies the streaming potential (ice-free surface → evaporation → flow)
  4. Alters the local conductivity structure (resistive cap removed)

This is a measurable anthropogenic modification of the electromagnetic
environment — predating radio, power grids, and industrial EM pollution
by decades.
"""

import numpy as np
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PI = np.pi
MU0 = 4 * PI * 1e-7


def ice_trade_scale():
    """Quantify the scale of the ice trade."""
    print("=" * 80)
    print("  THE ICE KING'S FOOTPRINT: Scale of 19th Century Ice Harvesting")
    print("=" * 80)

    # ─── Production timeline ──────────────────────────────────────────

    timeline = [
        (1806, 0.001, "Tudor's first shipment to Martinique (mostly melted)"),
        (1820, 0.003, "Boston exports ~3,000 tons/yr, 2/3 from Tudor"),
        (1847, 0.075, "Boston alone exports 75,000 tons + 27,000 tons local"),
        (1856, 0.146, "Peak India exports: 146,000 tons"),
        (1870, 1.0,   "~1 million tons (estimated US total)"),
        (1880, 5.0,   "~5 million tons consumed annually in US"),
        (1890, 8.0,   "Hudson harvests fail — ice famine → surge in Maine"),
        (1900, 12.0,  "Norway exports 1M tons; US ~10-12M tons"),
        (1907, 15.0,  "Peak: 15 million tons consumed in US"),
        (1914, 24.0,  "24M tons natural ice + 26M tons artificial = 50M total"),
        (1930, 5.0,   "Natural ice collapses; mechanical refrigeration dominates"),
    ]

    print(f"\n  {'Year':>6s} {'Mt/yr':>8s} {'Note':50s}")
    print("  " + "-" * 70)
    for year, mt, note in timeline:
        bar = "█" * int(min(mt * 3, 50))
        print(f"  {year:6d} {mt:7.1f}M {bar}")
        if note:
            print(f"  {'':6s} {'':7s} {note}")

    # ─── Convert to physical dimensions ───────────────────────────────

    print(f"\n\n  PHYSICAL DIMENSIONS AT PEAK (1907: 15 million tons)")
    print("  " + "-" * 60)

    mass_kg = 15e6 * 1000  # 15 million tons → kg
    rho_ice = 917  # kg/m³
    volume_m3 = mass_kg / rho_ice

    # Harvested ice was typically 18 inches (45 cm) thick
    thickness_m = 0.45
    area_m2 = volume_m3 / thickness_m
    area_km2 = area_m2 / 1e6

    print(f"    Total mass:              {mass_kg:.2e} kg ({mass_kg/1e9:.1f} billion kg)")
    print(f"    Volume:                  {volume_m3:.2e} m³ ({volume_m3/1e6:.1f} million m³)")
    print(f"    At 45 cm thickness:      {area_km2:.0f} km² of lake surface harvested")
    print(f"    Equivalent:              {area_km2:.0f} km² = {area_km2/2.59:.0f} square miles")
    print(f"    Compare: Walden Pond     = 0.25 km²")
    print(f"    Compare: Fresh Pond (MA) = 0.62 km²")
    print(f"    Compare: Lake Champlain  = 1,127 km²")

    # How many lakes?
    avg_lake_area_km2 = 2.0  # typical New England/Midwest harvesting lake
    avg_harvest_fraction = 0.5  # typically harvested half the lake
    n_lakes = area_km2 / (avg_lake_area_km2 * avg_harvest_fraction)

    print(f"\n    Typical harvest: 50% of lake surface")
    print(f"    Average lake:    ~{avg_lake_area_km2} km²")
    print(f"    Number of lakes: ~{n_lakes:.0f} lakes actively harvested")

    return {
        "mass_kg": mass_kg,
        "volume_m3": volume_m3,
        "area_km2": area_km2,
        "n_lakes": n_lakes,
    }


def electromagnetic_impact(scale):
    """Compute the EM impact of removing ice from lakes."""
    print(f"\n\n" + "=" * 80)
    print("  ELECTROMAGNETIC IMPACT OF ICE REMOVAL")
    print("=" * 80)

    # ─── Ice as electrical insulator ──────────────────────────────────

    sigma_ice = 1e-7       # S/m (pure ice)
    sigma_fresh = 0.02     # S/m (typical lake water)
    sigma_soil = 0.03      # S/m (surrounding soil)

    # Ice thickness
    d_ice = 0.45  # m (18 inches, harvesting minimum)

    print(f"""
  Ice is an electrical INSULATOR:
    σ_ice    = {sigma_ice:.0e} S/m
    σ_water  = {sigma_fresh} S/m (fresh lake water)
    σ_soil   = {sigma_soil} S/m (surrounding ground)

    Contrast: water/ice = {sigma_fresh/sigma_ice:.0e} (200,000×)

  A frozen lake has a RESISTIVE CAP:
    R_ice = d/(σ×A) = {d_ice}/{sigma_ice}×1 = {d_ice/sigma_ice:.0e} Ω·m²
    R_water (same depth) = {d_ice}/{sigma_fresh}×1 = {d_ice/sigma_fresh:.0f} Ω·m²

  The ice cap blocks 99.9999% of vertical current flow.
  It is an almost perfect electromagnetic shield for the lake.
    """)

    # ─── What the ice blocks ──────────────────────────────────────────

    print(f"  WHAT ICE BLOCKS:")
    print("  " + "-" * 60)

    # 1. Atmospheric electric field coupling
    E_fair = 130  # V/m (fair-weather atmospheric E-field)
    # With ice: E doesn't penetrate → no current in lake
    # Without ice: J = σ_water × E_surface_component
    # But E doesn't directly couple — it's the GLOBAL ELECTRIC CIRCUIT:
    # Fair-weather current density: J_z ≈ 2 pA/m² (Wilson 1920)
    J_gec = 2e-12  # A/m² (global electric circuit fair-weather current)

    # Through ice: J_through_ice = J_gec × σ_ice / σ_water ≈ 0
    J_through_ice = J_gec * sigma_ice / sigma_fresh

    print(f"    1. GLOBAL ELECTRIC CIRCUIT (fair-weather current)")
    print(f"       J_z (fair weather) = {J_gec:.0e} A/m² (downward)")
    print(f"       Through ice:        {J_through_ice:.2e} A/m² (blocked)")
    print(f"       Without ice:        {J_gec:.0e} A/m² (full coupling)")
    print(f"       Ice blocks {(1-J_through_ice/J_gec)*100:.4f}% of GEC current to the lake")

    # 2. Schumann resonance coupling
    E_schumann = 0.5e-3  # V/m (Schumann E-field at surface, ~0.5 mV/m)
    # Ice attenuates Schumann fields
    skin_ice = np.sqrt(2 / (2*PI*7.83 * MU0 * sigma_ice)) / 1000  # km
    skin_water = np.sqrt(2 / (2*PI*7.83 * MU0 * sigma_fresh)) / 1000  # km
    attenuation_ice = np.exp(-d_ice / (skin_ice * 1000))

    print(f"\n    2. SCHUMANN RESONANCE COUPLING (7.83 Hz)")
    print(f"       Skin depth in ice:    {skin_ice*1000:.0f} km (ice is transparent!)")
    print(f"       Skin depth in water:  {skin_water:.1f} km")
    print(f"       Attenuation through {d_ice*100:.0f} cm ice: {(1-attenuation_ice)*100:.6f}%")
    print(f"       Ice is TRANSPARENT to Schumann frequencies.")
    print(f"       The EM shield is for DC/low-freq, not Schumann.")

    # 3. Telluric current coupling
    # Telluric currents in the ground couple to lakes through the banks
    # Ice on the surface doesn't directly block horizontal telluric
    # BUT: ice changes the thermal/density structure → changes streaming
    E_telluric = 1.3e-3  # V/m (Kp=5)
    J_lake_telluric = sigma_fresh * E_telluric

    print(f"\n    3. TELLURIC CURRENT COUPLING")
    print(f"       Horizontal telluric: not directly blocked by surface ice")
    print(f"       But ice changes thermal stratification → changes convection")
    print(f"       → changes streaming current pattern in lake sediment")

    # 4. The BIG impact: evaporation and streaming potential
    print(f"\n    4. EVAPORATION AND STREAMING (the main impact)")

    # Ice-covered lake: no evaporation → no vertical water flow → no streaming current
    # Open lake: evaporation drives vertical flow → streaming current
    # Removing ice MID-WINTER exposes the lake to evaporation in cold, dry air

    evap_rate_open = 2.0  # mm/day (winter open water, cold dry air)
    evap_rate_ice = 0.1   # mm/day (sublimation from ice, much slower)

    # Streaming current from evaporation-driven flow through lake bed sediment
    epsilon = 80 * 8.854e-12
    zeta = -30e-3  # V (lake sediment, lower than soil)
    eta = 1e-3
    sigma_f_lake = 0.02
    C_ek = epsilon * abs(zeta) / (eta * sigma_f_lake)

    # Evaporation creates a downward flow through lake sediment (replacement)
    # ΔP ≈ ρ × g × evap_loss over the lake depth
    # For a shallow lake (3 m), losing 2 mm/day:
    grad_P_evap = 1000 * 9.81 * evap_rate_open / 1000 / 86400 / 3  # Pa/m/s... simplified
    # Actually: the evaporative loss creates a pressure deficit that drives
    # groundwater inflow through the lake bed
    # ΔP ≈ ρg × (head_difference) ≈ 100-1000 Pa for typical lake/aquifer
    delta_P_gw = 500  # Pa (typical hydraulic head driving GW into lake)

    J_streaming_open = C_ek * delta_P_gw / sigma_f_lake * sigma_f_lake
    J_streaming_ice = J_streaming_open * 0.1  # ice reduces GW exchange by ~90%

    print(f"       Evaporation (open water): {evap_rate_open} mm/day")
    print(f"       Sublimation (ice cover):  {evap_rate_ice} mm/day")
    print(f"       Ratio:                    {evap_rate_open/evap_rate_ice:.0f}×")
    print(f"")
    print(f"       Streaming current (open lake):  {J_streaming_open:.2e} A/m²")
    print(f"       Streaming current (ice-covered): {J_streaming_ice:.2e} A/m²")
    print(f"       Removal of ice INCREASES streaming current by {J_streaming_open/J_streaming_ice:.0f}×")

    # ─── Scale to the industry ────────────────────────────────────────

    print(f"\n\n  SCALE OF THE ICE TRADE'S EM IMPACT:")
    print("  " + "-" * 60)

    area = scale["area_km2"]
    area_m2 = area * 1e6

    # Total additional streaming current from exposed lakes
    delta_J = J_streaming_open - J_streaming_ice
    total_I = delta_J * area_m2

    print(f"    Lake surface exposed:     {area:.0f} km²")
    print(f"    Additional J per m²:      {delta_J:.2e} A/m²")
    print(f"    Total additional current: {total_I:.1f} A")
    print(f"    Duration:                 ~3 months (Jan-Mar harvest season)")
    print(f"    (Compare: a single lightning stroke = 30,000 A for 1 ms)")
    print(f"    (Compare: total Sq ionospheric current = ~100,000 A)")

    # Grade-3 coupling
    B = 50e-6  # T (New England latitude)
    g3_delta = delta_J * B * 0.9  # sin(inc) at 45°N ≈ 0.9

    print(f"\n    Grade-3 perturbation:")
    print(f"    Δ{{J, B}}₃ per m² = {g3_delta:.2e} T·A/m²")
    print(f"    Over {area:.0f} km²: total = {g3_delta * area_m2:.2e} T·A·m")

    return {
        "delta_J": delta_J,
        "total_I": total_I,
        "J_streaming_open": J_streaming_open,
    }


def comparative_context(scale, em):
    """Put the ice trade impact in context."""
    print(f"\n\n" + "=" * 80)
    print("  CONTEXT: The Ice Trade vs Other Anthropogenic EM Perturbations")
    print("=" * 80)

    print(f"""
  The ice trade was ONE of several 19th-century activities that modified
  the electromagnetic environment before anyone knew EM existed:

  ┌────────────────────────────────────────────────────────────────────┐
  │  Activity              Scale           EM mechanism               │
  ├────────────────────────────────────────────────────────────────────┤
  │  ICE HARVESTING        {scale['area_km2']:.0f} km²/yr     Removed insulating cap        │
  │  (1806-1930)           15 Mt/yr peak   → exposed conductive water │
  │                                        → enhanced streaming + GEC │
  │                                                                    │
  │  DEFORESTATION         ~100,000 km²/yr Removed root current array │
  │  (continuous)          in 19th century → eliminated streaming pump │
  │                                        → collapsed grade-3 field  │
  │                                                                    │
  │  CANAL BUILDING        ~10,000 km      Created conductive channels│
  │  (1790-1850)           Erie, Suez etc  → new telluric pathways    │
  │                                                                    │
  │  RAILROAD (iron rails) ~300,000 km     Laid conductive network    │
  │  (1830-1900)           by 1900         → grounded antenna array   │
  │                                        → modified telluric field  │
  │                                                                    │
  │  TELEGRAPH WIRES       ~1,000,000 km   First artificial EM network│
  │  (1840-1900)           by 1880         → detected telluric storms │
  │                                        → Carrington Event 1859   │
  │                                                                    │
  │  DRAINAGE/WETLANDS     ~100,000 km²    Removed conductive water   │
  │  (1800-1900)           drained in US   → reduced streaming current│
  │                                        → changed soil EM regime   │
  └────────────────────────────────────────────────────────────────────┘

  The ice trade ({scale['area_km2']:.0f} km²) was SMALLER than deforestation or
  wetland drainage in area. But it was UNIQUE in being:
    1. SEASONAL: removed and replaced every winter
    2. CONCENTRATED: mostly New England, Hudson River, Great Lakes
    3. REVERSIBLE: lakes refreeze (unlike deforestation)
    4. At CRITICAL LATITUDE: 40-45°N = high B inclination → strong {'{J,B}'}₃

  The ice trade's EM impact was modest compared to deforestation
  (which eliminates the 10⁻² A/m² streaming current from the root pump).
  But it was the first large-scale SEASONAL perturbation to the
  freshwater electromagnetic environment.
    """)

    # ─── The deeper question: ice as EM regulator ─────────────────────

    print(f"  THE DEEPER QUESTION: Ice as Natural EM Regulator")
    print("  " + "-" * 60)
    print(f"""
  Natural lake ice serves as a seasonal electromagnetic SWITCH:

    WINTER (ice on):
      - Resistive cap blocks GEC vertical current
      - No evaporation → no evaporative streaming
      - Reduced groundwater exchange → reduced streaming current
      - Lake is electromagnetically QUIET

    SPRING (ice off):
      - Conductive surface exposed to atmosphere
      - Evaporation begins → streaming current rises
      - Groundwater exchange increases → telluric coupling increases
      - Lake becomes electromagnetically ACTIVE

  This is a NATURAL seasonal modulation of the local EM environment.
  The ice trade DISRUPTED this cycle by removing ice 1-3 months early,
  extending the "electromagnetically active" season.

  For the {scale['n_lakes']:.0f} lakes harvested at peak:
    - Ice removed: January-March (normally ice-covered until April)
    - Electromagnetic exposure extended by ~2-3 months
    - Total additional "active days": ~60-90 × {scale['area_km2']:.0f} km²
                                    = {60 * scale['area_km2']:.0f} km²·days of extra EM exposure

  Was this detectable? Probably not with 19th-century instruments.
  But it was a real, physical modification of the electromagnetic
  landscape — and it happened decades before Maxwell's equations
  were published (1865), before Hertz detected radio waves (1888),
  and long before anyone imagined that lakes had an EM environment.

  The ice kings didn't know they were electromagnetic engineers.
  But they were: removing ~{scale['area_km2']:.0f} km² of natural insulation from
  the conducting surface of thousands of lakes, every winter,
  for a century.
    """)


def main():
    scale = ice_trade_scale()
    em = electromagnetic_impact(scale)
    comparative_context(scale, em)


if __name__ == "__main__":
    main()
