# 2025-2026 fault, solar, and core model audit

This audit separates a publishable observation from a transferable numerical
coefficient. It does not treat correlation as causation, turn a regional record
into a global rate multiplier, or use deep-Earth variability as a short-term
surface-hazard precursor.

## Cascadia and the northern San Andreas fault

Goldfinger et al. identify paired turbidites on the southern Cascadia and
northern San Andreas margins over about 3,100 years. Ten of 18 southern-
Cascadia event beds are paired with northern San Andreas beds; eight doublet
structures support an interpretation of possible stress triggering and partial
synchronization, most often with Cascadia first. The median paired age
difference is about 60 years, comparable to the dating uncertainty. Some
doublets permit separation of minutes to hours, while the full interpretation
allows lags extending to decades. Erosion, age uncertainty, and alternate
sedimentary interpretations prevent a calibrated real-time conditional
probability or universal alert window.

Model decision: `/api/earthquakes` now emits a **pending-confirmation** compound
fault candidate when a ComCat-like event passes a broad great-Cascadia screening
rule. The candidate is not marked active until authoritative attribution exists.
Magnitude 8, the bounding box, and the 70 km depth ceiling are explicit
implementation conventions, not fitted results. The candidate requires USGS or
other authoritative fault attribution and never supplies a probability. Global
Jelly Ball seismic-zone ratios are unchanged.

- Goldfinger et al. (2025), *Geosphere* 21, 1132-1180.
  https://doi.org/10.1130/GES02857.1

Shelly et al. add a separate, complementary geometry result at the Mendocino
triple junction. Tidal sensitivity and P-wave first motions indicate that a
zone of low-frequency earthquakes has dipping strike-slip motion. The authors
interpret it as a former Farallon (Pioneer) slab fragment captured by the
Pacific plate and moving north beneath westernmost North America. This extends
the inferred slab-interface geometry beyond a simple slab-window picture and
may represent previously unaccounted regional hazard.

Model decision: expose the captured-fragment interpretation as regional fault
geometry using the 27 published USGS LFE-family locations as a GeoJSON
`MultiPoint` feature. The points are observations supporting the interpretation,
not a slab surface or fault polygon. Do not add a hazard or
rupture multiplier: the paper establishes geometry and motion, not a calibrated
event probability. The open USGS LFE catalog can support a later spatial layer.

- Shelly et al. (2026), *Science* 391, 294-299.
  https://doi.org/10.1126/science.aeb2407
- USGS LFE catalog. https://doi.org/10.5066/P1TCKK7G

## Solar active-region helicity

Kim et al. used 24-hour SHARP time series and interpretable symbolic regression
for >=M-class flares. Their representative result combines flux near the
polarity-inversion line with a nonlinear interaction between total unsigned and
absolute net current helicity. The helicity interaction explained about 77% of
that model's predicted variance, but this is not a directly portable weight for
the repository's different detector and training contract.

Model decision: calculate and expose an instantaneous helicity-interaction
diagnostic in `solar-monitor/src/feeds/sharp.rs`. It is deliberately excluded
from `sharp_flare_risk` until the repository has a leakage-safe, active-region
split backtest using the required 24-hour histories and calibrated thresholds.

- Kim et al. (2026), *Astrophysical Journal Letters* 1005, L26.
  https://doi.org/10.3847/2041-8213/ae6cf8

The 2026 Cmod study similarly reinforces time-series modeling, missing-data
handling, temporal independence, and class-imbalance controls. It reports high
TSS on SWAN-SF, but its trained coefficients are not available as a drop-in
calibration for this monitor, so no score was copied.

- Azizian Foumani, Farokhi, and Qi (2026), *Solar Physics*.
  https://doi.org/10.1007/s11207-026-02612-6

Billcliff et al. combine an ensemble of ambient solar-wind solutions with a
logistic geomagnetic-storm forecast. This is useful evidence for an ensemble,
probabilistic warning architecture, but it explicitly omits CMEs and depends on
HUXt/MAS inputs that this repository does not ingest. Its reported skill and
logistic weights therefore are not copied into the operational score.

- Billcliff et al. (2026), *Space Weather*.
  https://doi.org/10.1029/2025SW004823

Two other 2026 results refine solar-dynamo interpretation rather than supply a
new operational predictor. Helioseismic torsional-oscillation evolution favors
a tachocline-origin interpretation, while candidate global magnetically
modified Rossby modes offer an additional probe of internal solar magnetism.
The monitor is observation-driven and neither study provides a calibrated
mapping from those signals to the repository's alert targets, so no live score
or forecast horizon changes.

- Mandal and Kosovichev (2026), *Scientific Reports*.
  https://doi.org/10.1038/s41598-025-34336-1
- Hanasoge and Hanson (2026), *Nature Astronomy*.
  https://doi.org/10.1038/s41550-026-02794-w

## Geomagnetic field and core dynamics

IGRF-14 is the current standard internal-field model for 2025-2030. The former
fallback labeled old approximate values as IGRF-13 and then discarded the two
off-axis degree-one terms, producing an axial field with an incorrect local
north-component sign. The fallback now uses the published IGRF-14 2025 degree-
one coefficients and their 2025-2030 secular variation in a tilted-dipole NED
synthesis. It remains only a degree-one fallback; full IGRF degree 13 is needed
for regional anomalies and precision work.

- Beggan et al. (2026), *Earth, Planets and Space*, IGRF-14 description.
  https://doi.org/10.1186/s40623-025-02360-0
- IAGA final coefficients. https://doi.org/10.5281/zenodo.14218973

Vidale et al. find that repeating-earthquake waveform changes are tentatively
explained by both differential inner-core rotation and localized deformation
near the inner-core boundary. This does **not** establish a geomagnetic-
reversal precursor, a short-term hazard signal, or an earthquake trigger. The
geomagnetic field is generated by flow in the liquid outer core, so this result
does not validate claims that the field is rigidly anchored to a tumbling solid
inner core.

- Vidale et al. (2025), *Nature Geoscience* 18, 267-272.
  https://doi.org/10.1038/s41561-025-01642-2

Model decision: inner-core rotation/deformation remains excluded from live
warning scores. IGRF-14 secular variation, observatory measurements, and
satellite field models are the appropriate state inputs. No defensible date for
the next geomagnetic reversal follows from the cited work.

Rivera et al.'s 3,300-year core-surface-flow reconstruction is useful for
long-timescale geodynamo research, while also making the inverse problem and
its uncertainty/regularization choices explicit. The early-Earth dynamo model
of Lin et al. further shows that dynamo action need not depend on a solid inner
core. Together these results reinforce removal of the repository's former
inner-core-anchored field and reversal-countdown claims. They do not provide a
present-day earthquake, eruption, reversal, or storm-alert coefficient.

- Rivera et al. (2026), *Geochemistry, Geophysics, Geosystems*.
  https://doi.org/10.1029/2025GC012475
- Lin et al. (2025), *Nature*.
  https://doi.org/10.1038/s41586-025-09334-y

## Deliberately unchanged

- No solar-to-earthquake causal coefficient was added.
- No global seismic-zone ratio or earthquake probability was changed.
- No Cascadia-to-San-Andreas conditional probability or countdown was invented.
- No solar flare score was changed without a matching 24-hour SHARP backtest.
- No inner-core motion was used as a reversal, volcanic, or seismic alert.
- No ambient-solar-wind ensemble weights were copied into a CME-inclusive
  monitor with different inputs and targets.
