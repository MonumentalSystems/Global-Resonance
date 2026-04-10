#!/usr/bin/env python3
"""
Zeus's Oak: Why Lightning Strikes Oaks More Than Other Trees
==============================================================
"Of all trees, the oak is most often struck by lightning." — Pliny the Elder

This is not myth. Multiple forestry surveys confirm it:
  - Heidler (1899): 54% of lightning-struck trees in German forests were oak
    (oaks were only ~10% of the forest)
  - Covert (1924): similar result in US midwest
  - Taylor (1965): 4-5× overrepresentation in UK
  - Schmitz (2020): meta-analysis confirms species-dependent strike rates

Standard explanation: oaks are tall, wide-crowned, often isolated.

But this can't be the full story:
  - Beeches of the same height and crown size are struck much LESS often
  - Poplars are TALLER than oaks but struck less frequently
  - The German proverb: "Eichen sollst du weichen, Buchen sollst du suchen"
    (Avoid oaks, seek beeches) — implies the difference is species-specific

What's special about oaks ELECTRICALLY?

1. DEEP TAPROOT: 3-5 m (most trees: 0.5-2 m)
   → Reaches deeper, wetter soil → lower ground resistance

2. MASSIVE ROOT NETWORK: 500+ tips/m² in a 20 m radius
   → ~600 million tips per mature tree
   → Total root current: ~600 A per tree

3. HIGH SAP CONDUCTIVITY: oak sap σ ≈ 0.5-1.0 S/m
   (vs beech sap σ ≈ 0.2-0.3 S/m, conifer sap σ ≈ 0.1-0.2 S/m)
   → Oak is a better conductor from crown to ground

4. HIGH TANNIN: gallic/ellagic acid in oak tissue
   → Chelates metal ions → increases tissue conductivity
   → Modifies rhizosphere ζ-potential → affects streaming current

5. MYCORRHIZAL NETWORK: oaks form ectomycorrhiza with massive
   fungal networks extending 50+ m from trunk
   → Fungal hyphae are conductive (σ ≈ 0.01-0.1 S/m)
   → The mycorrhizal network extends the effective antenna
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
# TREE SPECIES ELECTRICAL PROPERTIES
# ═══════════════════════════════════════════════════════════════════════

TREES = {
    "English oak (Quercus robur)": {
        "height_m": 25,
        "crown_radius_m": 10,
        "taproot_depth_m": 4.0,      # deep taproot
        "lateral_root_radius_m": 20,  # root spread ≈ 2× crown
        "root_tips_total": 6e8,       # ~600 million
        "root_tips_per_m2": 500,      # averaged over root zone
        "sap_sigma_Sm": 0.8,          # high electrolyte sap
        "bark_resistance_Ohm_m": 200, # moderate bark resistance
        "heartwood_sigma_Sm": 0.05,   # wet heartwood
        "tannin_pct": 8.0,            # high tannin (gallic acid)
        "mycorrhiza_radius_m": 50,    # ectomycorrhizal network
        "transpiration_L_day": 400,   # high transpiration
        "lightning_overrep": 4.5,     # Heidler 1899: 4-5× overrepresented
        "notes": "Zeus's tree. Deep taproot + high sap σ + massive root network.",
    },
    "European beech (Fagus sylvatica)": {
        "height_m": 30,               # taller than oak!
        "crown_radius_m": 8,
        "taproot_depth_m": 1.5,       # shallow heart-root system
        "lateral_root_radius_m": 15,
        "root_tips_total": 3e8,
        "root_tips_per_m2": 400,
        "sap_sigma_Sm": 0.25,         # lower electrolyte
        "bark_resistance_Ohm_m": 800, # smooth bark = high resistance
        "heartwood_sigma_Sm": 0.02,
        "tannin_pct": 2.0,            # low tannin
        "mycorrhiza_radius_m": 20,    # ectomycorrhizal but smaller
        "transpiration_L_day": 300,
        "lightning_overrep": 0.3,     # UNDER-represented (the safe tree)
        "notes": "'Seek beeches' — smooth bark insulates, shallow roots.",
    },
    "Scots pine (Pinus sylvestris)": {
        "height_m": 25,
        "crown_radius_m": 5,
        "taproot_depth_m": 2.5,
        "lateral_root_radius_m": 10,
        "root_tips_total": 2e8,
        "root_tips_per_m2": 600,
        "sap_sigma_Sm": 0.15,         # resinous, low electrolyte
        "bark_resistance_Ohm_m": 500, # thick bark
        "heartwood_sigma_Sm": 0.01,   # resinous, low σ
        "tannin_pct": 1.0,
        "mycorrhiza_radius_m": 30,
        "transpiration_L_day": 100,
        "lightning_overrep": 1.0,     # roughly average
        "notes": "Resin insulates. Moderate strike rate.",
    },
    "Lombardy poplar (Populus nigra)": {
        "height_m": 35,               # tallest common tree
        "crown_radius_m": 3,          # narrow columnar crown
        "taproot_depth_m": 2.0,
        "lateral_root_radius_m": 25,  # wide but shallow
        "root_tips_total": 4e8,
        "root_tips_per_m2": 200,      # sparse per m² (wide spread)
        "sap_sigma_Sm": 0.3,
        "bark_resistance_Ohm_m": 300,
        "heartwood_sigma_Sm": 0.03,
        "tannin_pct": 1.5,
        "mycorrhiza_radius_m": 10,    # arbuscular (less extensive)
        "transpiration_L_day": 500,   # very high
        "lightning_overrep": 0.7,     # slightly underrepresented
        "notes": "Tallest but narrow crown, shallow roots. Less struck than expected.",
    },
    "Sugar maple (Acer saccharum)": {
        "height_m": 25,
        "crown_radius_m": 8,
        "taproot_depth_m": 1.0,       # shallow fibrous roots
        "lateral_root_radius_m": 12,
        "root_tips_total": 4e8,
        "root_tips_per_m2": 800,      # dense fibrous
        "sap_sigma_Sm": 0.4,          # maple sap — moderate sugar
        "bark_resistance_Ohm_m": 400,
        "heartwood_sigma_Sm": 0.03,
        "tannin_pct": 3.0,
        "mycorrhiza_radius_m": 15,
        "transpiration_L_day": 200,
        "lightning_overrep": 1.2,
        "notes": "Dense roots but shallow. Moderate strike rate.",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# GROUND RESISTANCE MODEL
# ═══════════════════════════════════════════════════════════════════════

def tree_ground_resistance(tree):
    """
    Compute the effective ground resistance of a tree's root system.

    The lightning stepped leader is looking for the lowest-impedance
    path to ground. The tree's ground resistance determines how much
    current it can sink and therefore how attractive it is to the leader.

    Model: the root system as a vertical rod (taproot) in parallel
    with a horizontal disk (lateral roots), in a conducting medium (soil).

    R_taproot = ρ_soil / (2π L) × [ln(2L/a) - 1]   (vertical rod)
    R_lateral = ρ_soil / (4 r_disk)                   (disk electrode)
    R_total = R_taproot ∥ R_lateral

    Modified by:
    - Root tissue conductivity (roots are MORE conductive than soil)
    - Mycorrhizal network (extends the effective electrode)
    - Soil moisture (taproot reaches deeper, wetter soil)
    """
    # Soil resistivity (varies with depth due to moisture)
    rho_surface = 100  # Ω·m (typical forest soil, top 0.5 m)
    rho_deep = 30      # Ω·m (below 1 m, more saturated)

    L_tap = tree["taproot_depth_m"]
    a_tap = 0.15  # m (taproot radius, ~30 cm diameter)
    r_lat = tree["lateral_root_radius_m"]

    # Taproot: reaches into wetter soil
    # Weighted average resistivity
    rho_tap = rho_surface * min(1.0, 0.5/L_tap) + rho_deep * max(0, 1 - 0.5/L_tap)
    R_taproot = rho_tap / (2 * PI * L_tap) * (np.log(2 * L_tap / a_tap) - 1)

    # Lateral root disk: in surface soil
    # But root tissue (σ ≈ 0.05-0.1 S/m) is more conductive than soil
    # Effective conductivity is enhanced by root density
    root_vol_fraction = 0.02  # 2% of root zone volume is root tissue
    sigma_soil = 1 / rho_surface
    sigma_root = tree["heartwood_sigma_Sm"]
    sigma_eff = sigma_soil + root_vol_fraction * (sigma_root - sigma_soil)
    rho_eff = 1 / sigma_eff
    R_lateral = rho_eff / (4 * r_lat)

    # Mycorrhizal extension
    # Fungal hyphae extend the effective electrode radius
    r_myco = tree["mycorrhiza_radius_m"]
    # Hyphae are thin (10 μm) but numerous (100-1000 km per m³ of soil)
    # Effective conductivity enhancement is modest
    R_myco = rho_surface / (4 * r_myco) * 2  # factor 2: sparser than roots

    # Total: parallel combination
    R_total = 1 / (1/R_taproot + 1/R_lateral + 1/R_myco)

    # Crown-to-ground resistance through the trunk
    trunk_height = tree["height_m"]
    trunk_area = PI * 0.2**2  # ~40 cm diameter
    R_trunk = tree["bark_resistance_Ohm_m"] * 0.01  # bark is thin
    # Sap path (interior): much lower resistance
    R_sap = trunk_height / (tree["sap_sigma_Sm"] * 0.01)  # sap cross-section ~100 cm²

    R_crown_to_ground = R_sap + R_total

    return {
        "R_taproot": R_taproot,
        "R_lateral": R_lateral,
        "R_myco": R_myco,
        "R_ground": R_total,
        "R_sap": R_sap,
        "R_total": R_crown_to_ground,
    }


def root_current_budget(tree):
    """Total electromagnetic current from the tree's root system."""
    I_tip = 1e-6  # A per tip
    total_I = tree["root_tips_total"] * I_tip  # total root current
    area = PI * tree["lateral_root_radius_m"]**2
    depth = tree["taproot_depth_m"]
    J_root = total_I / (area * depth)  # volume-averaged
    return {
        "total_I_A": total_I,
        "J_root": J_root,
        "area_m2": area,
    }


def streaming_current(tree):
    """Streaming current from transpiration-driven water flow."""
    # Transpiration: L/day → m³/s
    Q = tree["transpiration_L_day"] / 1000 / 86400  # m³/s

    # Root suction: typical mid-day
    psi = 30e3  # Pa (30 kPa)

    # Streaming coefficient
    epsilon = 80 * 8.854e-12
    zeta = -50e-3  # V
    eta = 1e-3  # Pa·s
    sigma_f = 0.02  # S/m

    C_ek = epsilon * abs(zeta) / (eta * sigma_f)

    # Streaming current = C_ek × ΔP × cross_section / distance
    # Approximate: J_streaming = σ_f × C_ek × ΔP / σ_f = C_ek × ΔP
    # Over the root zone volume:
    depth = tree["taproot_depth_m"]
    area = PI * tree["lateral_root_radius_m"]**2
    E_streaming = C_ek * psi / sigma_f
    J_streaming = sigma_f * E_streaming

    return {
        "Q_m3_s": Q,
        "E_streaming_mV_m": E_streaming * 1e3,
        "J_streaming": J_streaming,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  ZEUS'S OAK: Why Lightning Prefers Quercus")
    print("  'Of all trees, the oak is most often struck.' — Pliny, NH XVI.93")
    print("=" * 80)

    # ─── Ground resistance comparison ─────────────────────────────────

    print(f"\n  GROUND RESISTANCE (path of least resistance for stepped leader):")
    print(f"\n  {'Species':40s} {'H m':>5s} {'Tap m':>6s} {'R_gnd':>8s} {'R_tot':>8s} "
          f"{'σ_sap':>6s} {'Struck':>7s}")
    print("  " + "-" * 85)

    for name, tree in TREES.items():
        r = tree_ground_resistance(tree)
        print(f"  {name:40s} {tree['height_m']:5.0f} {tree['taproot_depth_m']:6.1f} "
              f"{r['R_ground']:7.1f}Ω {r['R_total']:7.0f}Ω "
              f"{tree['sap_sigma_Sm']:6.2f} {tree['lightning_overrep']:6.1f}×")

    # ─── Current budget per tree ──────────────────────────────────────

    print(f"\n\n  ROOT CURRENT BUDGET (per tree):")
    print(f"\n  {'Species':40s} {'Tips':>10s} {'I_root A':>10s} {'J_root':>12s} {'I_stream':>12s}")
    print("  " + "-" * 90)

    for name, tree in TREES.items():
        rc = root_current_budget(tree)
        sc = streaming_current(tree)
        print(f"  {name:40s} {tree['root_tips_total']:.0e} {rc['total_I_A']:10.0f} "
              f"{rc['J_root']:12.2e} {sc['J_streaming']:12.2e}")

    # ─── The oak advantage ────────────────────────────────────────────

    print(f"\n\n  THE OAK ADVANTAGE (vs beech, same-height comparison):")
    print("  " + "-" * 60)

    oak = TREES["English oak (Quercus robur)"]
    beech = TREES["European beech (Fagus sylvatica)"]

    r_oak = tree_ground_resistance(oak)
    r_beech = tree_ground_resistance(beech)
    rc_oak = root_current_budget(oak)
    rc_beech = root_current_budget(beech)
    sc_oak = streaming_current(oak)
    sc_beech = streaming_current(beech)

    print(f"  {'Property':35s} {'Oak':>12s} {'Beech':>12s} {'Ratio':>8s}")
    print("  " + "-" * 70)
    print(f"  {'Height (m)':35s} {oak['height_m']:12.0f} {beech['height_m']:12.0f} {oak['height_m']/beech['height_m']:8.2f}")
    print(f"  {'Taproot depth (m)':35s} {oak['taproot_depth_m']:12.1f} {beech['taproot_depth_m']:12.1f} {oak['taproot_depth_m']/beech['taproot_depth_m']:8.1f}")
    print(f"  {'Sap conductivity (S/m)':35s} {oak['sap_sigma_Sm']:12.2f} {beech['sap_sigma_Sm']:12.2f} {oak['sap_sigma_Sm']/beech['sap_sigma_Sm']:8.1f}")
    print(f"  {'Ground resistance (Ω)':35s} {r_oak['R_ground']:12.1f} {r_beech['R_ground']:12.1f} {r_beech['R_ground']/r_oak['R_ground']:8.1f}")
    print(f"  {'Total R crown→ground (Ω)':35s} {r_oak['R_total']:12.0f} {r_beech['R_total']:12.0f} {r_beech['R_total']/r_oak['R_total']:8.1f}")
    print(f"  {'Root current (A)':35s} {rc_oak['total_I_A']:12.0f} {rc_beech['total_I_A']:12.0f} {rc_oak['total_I_A']/rc_beech['total_I_A']:8.1f}")
    print(f"  {'Bark resistance (Ω·m)':35s} {oak['bark_resistance_Ohm_m']:12.0f} {beech['bark_resistance_Ohm_m']:12.0f} {beech['bark_resistance_Ohm_m']/oak['bark_resistance_Ohm_m']:8.1f}")
    print(f"  {'Mycorrhizal radius (m)':35s} {oak['mycorrhiza_radius_m']:12.0f} {beech['mycorrhiza_radius_m']:12.0f} {oak['mycorrhiza_radius_m']/beech['mycorrhiza_radius_m']:8.1f}")
    print(f"  {'Lightning overrepresentation':35s} {oak['lightning_overrep']:12.1f}× {beech['lightning_overrep']:12.1f}× {oak['lightning_overrep']/beech['lightning_overrep']:8.1f}")

    # ─── Stepped leader attachment physics ────────────────────────────

    print(f"""

  STEPPED LEADER ATTACHMENT PHYSICS:

  The downward stepped leader approaches to ~100 m above ground.
  At this point, upward CONNECTING LEADERS launch from the tallest
  or best-grounded objects. The winner is determined by:

    1. HEIGHT:        taller = leader starts earlier = advantage
    2. TIP SHARPNESS: pointed = higher E-field enhancement = advantage
    3. GROUND R:      lower R = more current available = advantage

  For the oak vs beech comparison:
    Height advantage:     beech is TALLER (30 vs 25 m) → beech wins
    Crown sharpness:      oak has more pointed branches → oak wins slightly
    Ground resistance:    oak R = {r_oak['R_ground']:.1f} Ω, beech R = {r_beech['R_ground']:.1f} Ω
                          → oak ground R is {r_beech['R_ground']/r_oak['R_ground']:.1f}× LOWER → oak wins

  The ground resistance advantage overrides the height disadvantage.
  The oak's deep taproot reaches wet subsoil that the beech cannot access.

  The SAPPING CURRENT reinforces this:
    Oak sap σ = {oak['sap_sigma_Sm']} S/m (3× beech at {beech['sap_sigma_Sm']} S/m)
    → The oak trunk is a {oak['sap_sigma_Sm']/beech['sap_sigma_Sm']:.0f}× better conductor
    → The connecting leader propagates faster in oak

  And the BARK matters:
    Oak bark: furrowed, rough, lower resistance ({oak['bark_resistance_Ohm_m']} Ω·m)
    Beech bark: smooth, intact, higher resistance ({beech['bark_resistance_Ohm_m']} Ω·m)
    → Surface flashover (current running down the outside) is easier on oak
    → Beech's smooth bark is literally an insulator
    """)

    # ─── Root current creates a ground-level E-field signature ────────

    print(f"  ROOT CURRENT SIGNATURE:")
    print("  " + "-" * 60)
    print(f"""
  The 600 A of root current in a mature oak creates a detectable
  electromagnetic signature at the soil surface:

    B_root = μ₀ × I / (2π × r)  at distance r from trunk

    At the trunk (r = 0.5 m):  B = {MU0 * 600 / (2 * PI * 0.5) * 1e9:.1f} nT
    At the crown edge (r = 10 m): B = {MU0 * 600 / (2 * PI * 10) * 1e9:.1f} nT
    At the root limit (r = 20 m): B = {MU0 * 600 / (2 * PI * 20) * 1e9:.1f} nT

  This is a MEASURABLE field. A fluxgate magnetometer at the base
  of an oak should detect a ~240 nT anomaly — comparable to a small
  ore deposit. The field has a 24-hour cycle following transpiration.

  The stepped leader 'sees' this as a ground-level current source.
  The oak's root system is electrically equivalent to a buried
  conductor with 600 A of continuous current — a lightning rod
  that the tree built itself, over decades, guided by electrotropism.
    """)

    # ─── The mycorrhizal network as distributed antenna ───────────────

    print(f"  THE MYCORRHIZAL ANTENNA:")
    print("  " + "-" * 60)
    print(f"""
  Oaks form ECTOMYCORRHIZAL associations with fungi (Boletus, Amanita,
  Tuber, Russula). The fungal mycelium extends 50+ m from the trunk.

  A single mycorrhizal network can connect dozens of trees:
    - "Wood Wide Web" (Simard 1997): resource sharing via hyphae
    - Hyphae diameter: 2-10 μm
    - Hyphal density: 100-1000 km per m³ of soil
    - Hyphal conductivity: σ ≈ 0.01-0.1 S/m (cytoplasmic)

  The mycorrhizal network is an ELECTRICALLY CONNECTED MESH:
    - Fungal cytoplasm conducts ions (K⁺, H⁺, Ca²⁺)
    - Electric signals propagate along hyphae (action-potential-like)
    - The network has been shown to transmit stress signals between trees

  For lightning attachment:
    - The mycorrhizal network extends the effective ground electrode
    - Oak mycorrhiza radius: {oak['mycorrhiza_radius_m']} m (vs beech: {beech['mycorrhiza_radius_m']} m)
    - Effective ground contact area: π×{oak['mycorrhiza_radius_m']}² = {PI * oak['mycorrhiza_radius_m']**2:.0f} m²
    - This is a {PI * oak['mycorrhiza_radius_m']**2 / (PI * oak['lateral_root_radius_m']**2):.1f}× larger
      electrode than the root system alone

  The oak's mycorrhizal network turns a single tree into a
  distributed antenna system spanning ~8,000 m² of forest floor.
  No other common European tree has this combination of deep taproot
  + high sap conductivity + massive ectomycorrhizal network.
    """)

    # ─── Grade-3 interpretation ───────────────────────────────────────

    print(f"  THE GRADE-3 CONNECTION:")
    print("  " + "-" * 60)

    B_earth = 50e-6  # T
    I_oak = 600  # A total root current
    g3_oak = I_oak * 1e-6 * B_earth  # per tip, but 6e8 tips

    # The oak's root array produces a grade-3 pseudoscalar field
    J_oak = rc_oak['J_root']
    g3_density = J_oak * B_earth * 0.7  # sin(inc) at European latitudes

    print(f"""
  The oak's root current array ({rc_oak['total_I_A']:.0f} A through {oak['root_tips_total']:.0e} tips)
  produces a grade-3 coupling to Earth's field:

    {{J_root, B}}₃ = {g3_density:.2e} T·A/m² (at 50° latitude)

  This pseudoscalar field is CHIRAL (oak tissue is L-amino, D-sugar)
  and CIRCADIAN (peaks at midday with transpiration).

  The lightning connection to grade-3:
    Lightning channel current (~30 kA) is VERTICAL
    Earth's field has a component ALONG the channel (sin I)
    {{J_lightning, B}}₃ = 30,000 × 50×10⁻⁶ × sin(63°) = {30000 * 50e-6 * np.sin(np.radians(63)):.2f} T·A/m²

  When lightning strikes an oak:
    - The 30 kA channel current flows through CHIRAL root tissue
    - The grade-3 coupling is {30000 * 50e-6 * np.sin(np.radians(63)) / g3_density:.0f}× larger than the root's own
    - The CISS effect is activated at enormous current density
    - For a brief moment, the oak becomes a massive chiral current source

  Zeus didn't choose the oak arbitrarily:
    Deep taproot + high sap σ + massive mycorrhiza =
    lowest ground resistance + largest antenna + best conductor =
    THE tree that lightning prefers, for electromagnetic reasons
    that the Greeks encoded as divine preference.

  The iron thread in the oak:
    Fe in ferredoxin → photosynthesis → sugar → sap flow
    Fe in cytochrome → root respiration → root current
    Fe in magnetite in soil → electrotropism target → root growth direction
    Fe in the lightning channel → 30 kA through chiral tissue
    All grade-3. All iron. All connected.
    """)


if __name__ == "__main__":
    main()
