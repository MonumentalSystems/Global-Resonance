# The Jelly Ball as Ringing Bell: Spherical Harmonic Modal Decomposition of CME-Seismicity Coupling

**Date:** April 4, 2026  
**Context:** Comet ATLAS perihelion passage, post-G3 geomagnetic storm recovery  
**Data:** 389 Kp>=5 storms, 182,967 M4.5+ earthquakes, 2000-2026

---

## 1. Summary

The Paper XXV "Jelly Ball" model predicts that CME impacts at the subsolar point produce a spatial pattern of seismicity modulation across the globe. We show that this pattern is not arbitrary but corresponds to the **nodes and antinodes of Legendre polynomial cavity modes** — the same spherical harmonics that govern Schumann resonances. Furthermore, we demonstrate that the spatial pattern **inverts between storm compression and relaxation phases**, consistent with a damped standing wave in the Earth's lithosphere.

The refined model treats the Earth as a resonating sphere struck by CME impulses, ringing at lithospheric timescales (~1 cycle/day) with the same mode shapes as electromagnetic Schumann resonances (7.83 Hz) but coupled through the global electric circuit and telluric currents.

---

## 2. Background: Paper XXV Static Zones

The original Jelly Ball model defines 10 zones by angular distance from the subsolar point at the time of CME impact:

| Zone | Angular Range | Expected Ratio | Effect |
|------|--------------|----------------|--------|
| Eye | 0-15 deg | 0.85x | Suppression |
| Inner | 15-30 deg | 0.92x | Compression |
| Transition | 30-60 deg | 0.98x | Near-neutral |
| **Wavefront** | **60-75 deg** | **1.36x** | **Peak enhancement** |
| Wavefront-tail | 75-100 deg | 1.09x | Enhancement |
| Neutral | 100-120 deg | 0.95x | Neutral |
| **Far-suppress** | **120-135 deg** | **0.82x** | **Suppression** |
| Far-neutral | 135-155 deg | 0.90x | Far neutral |
| Pre-antipodal | 155-165 deg | 1.00x | Neutral |
| Antipodal | 165-180 deg | 1.16x | Enhancement |

---

## 3. Modal Decomposition

### 3.1 Legendre Polynomial Fit

The static pattern decomposes into spherical harmonic (Legendre polynomial) modes:

**R(theta) = 1 + sum_l[ a_l * P_l(cos theta) ]**

| Mode | Coefficient a_l | Schumann f_l | Physical Role |
|------|----------------|-------------|---------------|
| l=1 | +0.102 | 10.6 Hz | Dipole (N-S asymmetry) |
| **l=2** | **-0.145** | **14.3 Hz** | **Quadrupole (eye + far-suppress)** |
| **l=3** | **-0.314** | **20.8 Hz** | **Octupole (wavefront peak)** |
| l=4 | +0.072 | 27.3 Hz | Fine structure |
| l=5 | +0.080 | 33.5 Hz | Fine structure |
| l=6 | +0.113 | 41.0 Hz | Fine structure |

The **l=3 octupole** is the strongest mode (a=-0.314), creating the wavefront peak at 60-75 degrees. The **l=2 quadrupole** (a=-0.145) creates the paired suppression at the eye (0 deg) and far-suppress zone (125 deg).

### 3.2 Node Alignment

Legendre polynomial nodes (zeros) align remarkably with the Paper XXV zone boundaries:

| P_l Node | Degrees | Nearest Zone Boundary | Distance |
|----------|---------|----------------------|----------|
| P2 | 55 deg | Transition/Wavefront (60 deg) | 5 deg |
| **P2** | **125 deg** | **Far-suppress center (127.5 deg)** | **2.5 deg** |
| P3 | 39 deg | Transition zone (30-60 deg) | inside |
| P3 | 90 deg | Wavefront-tail (75-100 deg) | boundary |
| P3 | 141 deg | Far-neutral (135-155 deg) | match |
| P4 | 70 deg | Wavefront (60-75 deg) | 5 deg |
| P5 | 57 deg | Transition/Wavefront (60 deg) | 3 deg |
| P5 | 123 deg | Far-suppress (120 deg) | 3 deg |
| P5 | 155 deg | Far-neutral/Pre-antipodal (155 deg) | exact |
| P6 | 76 deg | Wavefront/Wavefront-tail (75 deg) | 1 deg |
| P6 | 104 deg | Neutral (100-120 deg) | 4 deg |
| P6 | 131 deg | Far-suppress (120-135 deg) | 4 deg |

The clustering of nodes from multiple modes near 55-60 deg, 120-125 deg, and 155 deg suggests these are **resonant boundaries** of the spherical cavity, not arbitrary bin edges.

---

## 4. Phase-Resolved Backtest

### 4.1 Method

We identified 389 geomagnetic storms (Kp >= 5) in the 2000-2026 record and computed seismicity density in each Paper XXV zone during four storm phases:

- **Background:** Day -10 to -5 (pre-storm baseline)
- **Compression:** Day -1 to 0 (Kp rising, J increasing above J_c)
- **Peak:** Day 0 to +1 (storm maximum)
- **Relaxation early:** Day +1 to +3 (Kp falling, J dropping through J_c)
- **Relaxation late:** Day +3 to +7 (recovery)

### 4.2 Results

Phase-resolved modal coefficients (l=1 through l=4):

| Phase | l=1 | l=2 | l=3 | l=4 |
|-------|-----|-----|-----|-----|
| Compression | -0.094 | **+0.124** | +0.021 | -0.209 |
| Peak | -0.014 | -0.026 | -0.043 | +0.076 |
| Relaxation early | -0.028 | -0.063 | +0.046 | +0.127 |
| Relaxation late | -0.037 | -0.031 | -0.023 | +0.032 |

### 4.3 The l=2 Sign Flip

The l=2 (quadrupole) coefficient **changes sign** between compression (+0.124) and relaxation (-0.031 to -0.063). This means:

- **During compression**: P2 is positive at the nodes (55, 125 deg) — the far-suppress zone is **enhanced** (strain loading)
- **During relaxation**: P2 is negative at the nodes — the far-suppress zone is **suppressed** (returning to baseline)

**Statistical significance:** Paired t-test on the far-suppress zone density between compression and relaxation phases gives **t=-3.16, p=0.0017** across 389 storms. The effect is highly significant.

### 4.4 Wavefront-Tail Delayed Enhancement

The wavefront-tail zone (75-100 deg) shows **delayed enhancement** — 1.24x at peak and 1.21x during early relaxation, higher than during compression (0.91x). This is consistent with a wave propagating outward from the subsolar point with finite travel time through the crust.

---

## 5. The Ringing Bell Model

### 5.1 Physical Picture

A CME impact is an impulse to a spherical cavity. The cavity response is a superposition of normal modes:

**R(theta, t) = 1 + sum_l[ A_l * cos(omega_l * t + phi_l) * exp(-gamma_l * t) * P_l(cos theta) ]**

where:
- **A_l** = excitation amplitude (proportional to CME dynamic pressure)
- **omega_l** = cavity mode frequency (~1 cycle per 1-3 days for lithospheric modes)
- **phi_l** = initial phase (determined by impact geometry)
- **gamma_l** = damping rate (Q ~ 3-5, matching storm recovery timescale)
- **P_l(cos theta)** = Legendre polynomial (same as Schumann resonance spatial pattern)

### 5.2 Two Frequency Scales, Same Geometry

| Cavity | Medium | Frequency | Timescale | Damping |
|--------|--------|-----------|-----------|---------|
| Schumann | Ionosphere-surface (EM) | 7.83, 14.3, 20.8 Hz | Milliseconds | Q ~ 5-10 |
| Jelly Ball | Lithosphere (mechanical) | ~0.3-1 cycle/day | Hours-days | Q ~ 3-5 |

Both share **P_l(cos theta)** spatial patterns because they inhabit the same sphere. The coupling mechanism is the **global electric circuit**: Schumann resonances modulate ionospheric conductivity, which modulates telluric currents (Jz), which modulate stress at faults near criticality.

### 5.3 The Strain Accumulation Mechanism

At Legendre nodes (zeros of P_l), the standing wave has zero displacement. In a driven system:

1. **Compression phase**: CME drives the cavity. Antinodes respond (seismicity modulated). Nodes cannot respond — strain **accumulates**.
2. **Phase inversion**: As the wave completes half a cycle (~1-2 days), nodes become antinodes and vice versa. Accumulated strain at former nodes **releases**.
3. **Ring-down**: Each subsequent cycle is smaller (damped by gamma_l). After Q cycles (~3-5 days), the system returns to background.

This explains why the far-suppress zone (sitting on the P2 node at 125 deg) is **enhanced** during compression (loading) and returns to baseline during relaxation (releasing).

---

## 6. Current Event: April 4, 2026 Comet Perihelion

### 6.1 Timeline

| Time (UTC) | Event | J State |
|-----------|-------|---------|
| Mar 30 03:24 | X1.5 flare | J >> J_c (impulse) |
| Mar 31 10:40 | CME shock | J re-compressed |
| Apr 1-2 | CME body | J sustained > J_c |
| **Apr 2 14:13** | **Indonesia M5.9** | **Swarm begins at 121 deg (P2 node)** |
| Apr 3 18:00 | G3 peak (Kp=7) | J >> J_c maximum |
| Apr 3-4 | 58 events Indonesia | Strain loading at P2 node |
| Apr 4 09:00 | Recovery | J = 0.60, 6% below J_c |
| Apr 4 13:40 | Hardness spikes | Criticality rising |
| **Apr 4 14:22** | **Comet perihelion** | **Criticality = 1.0** |
| Apr 4 15:05 | Escalation -> FLARE | 17 triggers, peak fused 0.75 |
| Apr 4 15:30 | Criticality discharge | Lattice energy released |
| Apr 4 18:00+ | Rate declining | 2 events in 6h post-perihelion |

### 6.2 Interpretation

The Indonesia swarm at 121 deg sits precisely on the **P2 Legendre node** (125 deg). During the G3 storm compression, this node loaded strain (observed ratio 6.24x, expected 0.82x). The swarm is the strain accumulation at a cavity node — exactly what the ringing bell model predicts.

The comet perihelion at 14:22 UTC coincided with the criticality detector hitting 1.0 (Clifford lattice maximum stress), followed by discharge 68 minutes later. The escalation reached FLARE level with 28 hardness spikes and peak fused score 0.75.

### 6.3 Prediction

If no new CME is produced by the perihelion interaction:
- Indonesia swarm tapers by April 5-6 (Q ~ 3-5 cycles = 3-5 days from G3 peak)
- J returns to baseline (~0.55) by April 6
- l=2 mode rings down to zero

If a new CME arrives from perihelion-triggered activity (April 6-7):
- Cavity re-excited: new impulse restarts the ringing
- Indonesia swarm could re-intensify (P2 node reloads)
- Compound event: larger eventual relaxation burst

---

## 7. Testable Predictions

1. **Indonesia swarm decay rate** should match exp(-gamma_2 * t) with gamma_2 corresponding to Q ~ 3-5 for the l=2 mode. The swarm rate should halve every ~1.5 days.

2. **The wavefront-tail zone (75-100 deg)** should show enhanced seismicity ~1-3 days after impulse (delayed wave arrival), then decay. Currently at 1.21x (early relaxation) — should decline to ~1.0x by April 6.

3. **Future Kp>=5 storms** should show the same l=2 sign flip: far-suppress enhanced during compression, suppressed during relaxation. This is testable with the next storm.

4. **Schumann resonance intensity** should show correlated modulation in the l=2 mode during the same storm phases. If the ringing bell model is correct, the 14.3 Hz Schumann mode should show amplitude changes synchronized with the seismic spatial pattern inversion.

5. **Bz dependence**: The l=2 inversion should be stronger for northward Bz (shield OFF, compression transmits to crust) than southward Bz (shield ON, energy dissipated as aurora). This can be tested by splitting the 389 storms by Bz polarity.

---

## 8. Solar Harmonics: The Same Modes on the Sun

The Legendre mode structure is not unique to Earth. Analysis of 3,202 solar flares shows the **same P_l eigenmodes govern the Sun's activity distribution**:

| Mode | Solar <P_l> | t-stat | Physical meaning |
|------|------------|--------|------------------|
| **l=2** | **-0.375** | **-229** | Active region belt (butterfly diagram) |
| l=4 | +0.110 | +36 | Joy's law tilt structure |
| l=3 | +0.047 | +8 | North-South asymmetry |

The l=2 quadrupole dominates both systems: on the Sun it creates the butterfly diagram active region belt at ±15° latitude; on Earth it creates the far-suppress zone at 125° from the subsolar point. Both are **P_2 node structures**.

Flares cluster significantly closer to Legendre nodes than random (p < 0.001 for l=1, 2, 3, 4, 6). M/X class flares show **stronger** l=2 coupling (-0.380) than C/B flares (-0.352) — energetic eruptions are more tightly bound to the harmonic geometry.

The Sun and Earth are **weakly coupled oscillators** sharing the same Cl(3,0) geometry on S²:
- Kuramoto phase synchronization: r = 0.04 (not phase-locked, p=0.83)
- Daily Kp→EQ lagged correlation: r = -0.033 at lag 2-3 days (p=0.001)
- The anti-correlation confirms the Jelly Ball: storms suppress, then release

The coupling is **impulsive, not resonant**: each CME is a discrete kick to a damped oscillator. Earth rings at its own l=2 frequency (Q~3-5, ~3-5 day decay), not at the solar cycle frequency.

---

## 9. Lunar Coupling: Three-Body l=2 Resonance

The Moon adds a third l=2 mode to the system. Analysis of 165,155 shallow earthquakes shows:

**Fortnightly M2 tidal signal**: chi² = 101, p < 0.0001. Earthquakes prefer **quarter moons** over new/full — the **opposite** of direct tidal triggering. This is the same strain-storage-release mechanism: spring tides load strain at P_2 nodes, quarter moons release it.

**Depth gradient matches pore fluid coupling:**

| Depth | M2 signal | p-value |
|-------|----------|---------|
| 0-15 km | -0.002 | 0.53 (ns) |
| 15-35 km | -0.013 | 0.001 ** |
| **35-70 km** | **-0.019** | **< 0.0001** *** |
| 70-150 km | +0.001 | 0.81 (ns) |

The tidal signal peaks at **35-70 km** (brittle-ductile transition) — the same depth range where the solar EM coupling is strongest. Pore fluid is the shared medium for both solar and lunar forcing.

### Three-Body l=2 Model

Three quadrupole modes on three bodies, all acting on the same P_2(cos θ):

| Body | Period | Mechanism |
|------|--------|-----------|
| Sun | 22 years (Hale) | IMF → magnetosphere → telluric Jz → pore pressure |
| Moon | 14.77 days (M2) | Body tide → tidal stress → pore pressure |
| Earth | 3-5 days (ringdown) | Cavity mode → zone-resolved strain release |

**Combined effective coupling:**

**J_eff(θ, t) = J_tectonic + a_solar · P_2(cos θ) · f(t) + a_lunar · P_2(cos θ) · g(t) + a_storm · P_2(cos θ) · h(t) · exp(-γt)**

When all three align (storm + spring tide + P_2 node geometry), the effective J can cross J_c even for moderate individual forcings. The April 2026 Indonesia swarm exemplifies this: G3 storm (a_storm large) + Full Moon April 1 (a_lunar maximum) + comet perihelion + epicenter at 121° (P_2 node at 125°).

---

## 10. Implications

The Jelly Ball is not a static spatial filter — it is a **dynamical system** embedded in a **three-body l=2 resonance**. The Earth responds to CME impulses as a resonating sphere, with the spatial pattern of seismicity modulation determined by the instantaneous phase of the ringing cavity modes, further modulated by the lunar tidal cycle.

This connects four previously separate phenomena through a single geometric framework:
- **Solar activity** (l=2 butterfly diagram, flare clustering at P_l nodes)
- **Schumann resonances** (electromagnetic cavity modes, P_l spatial patterns, Hz timescale)
- **Geomagnetic storm recovery** (magnetospheric dynamics, days timescale)
- **Post-storm seismicity patterns** (lithospheric strain release, same P_l spatial patterns, days timescale)
- **Lunar tidal modulation** (fortnightly M2, body tide stress, same P_2 depth profile)

The coupling mechanism is the **global electric circuit**: ionospheric conductivity changes (driven by solar wind) and tidal body forces (driven by the Moon) both modulate pore pressure at faults near criticality, with the spatial pattern governed by Legendre polynomial eigenmodes — the natural modes of Clifford algebra on a sphere.

The critical threshold J_c = 2/π appears to be a **universal property of Cl(3,0) on S²**, not specific to any one system. It manifests as the Kuramoto critical coupling for phase synchronization, the KT phase transition for vortex unbinding, and the onset of collective behavior in coupled oscillator networks — all unified by the same geometric algebra on the same spherical geometry.
