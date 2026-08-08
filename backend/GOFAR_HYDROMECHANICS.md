# Gofar rupture barriers: model update

## Evidence incorporated

The Gofar transform fault produces approximately M6 earthquakes on roughly
5-6 year cycles, but persistent, highly damaged structural barriers repeatedly
stop rupture. The 2026 Science study identifies multistrand faulting and
transtensional stepovers with roughly 100-400 m offsets at these barriers. Its
interpretation is that seawater infiltration and damage-enhanced porosity allow
rapid dilatancy to lower pore pressure during rupture, increasing effective
normal stress and braking propagation.

An independent 2026 companion model of Gofar swarm chambers couples compaction,
dilatancy, permeability, pore pressure, and rate-and-state friction. Its Gofar
setup uses 140 mm/yr loading and 50 MPa effective normal stress. In its
conceptual mainshock scenario, transient drainage lowers pore pressure by about
15 MPa, a direct effective-stress increase equal to 30% of the 50 MPa reference.
The simulation and figure data are openly archived.

Primary sources:

- Gong et al. (2026), *Predictable seismic cycles result from structural rupture
  barriers on oceanic transform faults*, Science 392, 718-723.
  https://doi.org/10.1126/science.ady6190
- USGS publication record and abstract:
  https://pubs.usgs.gov/publication/70276297
- Jiang, Zhang, and Li (2026), *Hydro-Mechanical Controls on Swarm Recurrence on
  the Westernmost Gofar Transform Fault, East Pacific Rise*.
  https://doi.org/10.1029/2025GL119319
- Companion simulation and figure archive:
  https://doi.org/10.5281/zenodo.17067488
- McGuire et al. (2012), *Variations in earthquake rupture properties along the
  Gofar transform fault, East Pacific Rise*.
  https://doi.org/10.1038/ngeo1454

## Calculation boundary

The update separates two stages that the previous pore-pressure display could
blur together:

1. Before slip, a positive pore-pressure perturbation reduces effective normal
   stress (`delta sigma_eff = -delta p`) and can favor nucleation, swarms, or
   aseismic deformation.
2. During rapid slip at a porous, poorly drained structural barrier, dilatancy
   can produce a negative pore-pressure transient. That increases effective
   normal stress and can inhibit rupture propagation.

The 15 MPa / 50 MPa calculation is shown only as a Gofar companion-model
reference. It is not a direct field measurement and is not converted into a
global probability or magnitude multiplier. Consequently, the existing global
Jelly Ball zone ratios are not changed. Applying this mechanism elsewhere will
require local porosity, permeability/drainage timescale, fault geometry, stress,
and dynamic pore-pressure measurements or calibrated simulations.
