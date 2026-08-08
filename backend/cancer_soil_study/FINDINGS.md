# Preliminary nationwide screen (2026-07-13)

## What is assembled

- NCI State Cancer Profiles 2018-2022 age-adjusted county incidence for ten
  cancer-site groupings (up to 3,142 county-equivalent areas before suppression).
- EPA county radon-potential zones.
- USDA SOLUS100 predicted saturated-paste electrical conductivity at 0 cm,
  summarized within Census county polygons.
- USGS 2.45 km predicted cobalt, iron, manganese, selenium, and zinc rasters,
  summarized within Census county polygons.
- A map-ready GeoJSON containing all outcome and exposure fields.

The USGS ScienceBase record advertises a copper raster, but its published file
URL returned HTTP 404 during this run. Copper is therefore not silently imputed
or replaced by a weaker nearest-sample proxy.

## First result

There is no positive nationwide conductivity signal in this first model. The
joint screen controlled for EPA radon category, state fixed effects, rurality,
and five orthogonal metal factors. A one-standard-deviation increase in log
surface conductivity was associated with:

- all-site incidence: -3.29 cases per 100,000 (BH q=0.104);
- lung incidence: -1.80 cases per 100,000 (BH q=0.00045); and
- esophagus incidence: -0.35 cases per 100,000 (BH q=0.00014).

The other tested cancer sites had no conductivity association after correction.
These negative coefficients do not imply protection. They show that the simple
positive-conductivity claim is not reproduced by this proxy and model.

Several metal factors correlate with several cancer outcomes, but the pattern is
not specific. For example, the selenium-dominant second factor is associated
with all-site, bladder, lung, and thyroid incidence. The third factor (higher
manganese and lower iron/zinc) is associated with all-site, esophagus, and lung
incidence. Such broad patterns can arise from regional diet, smoking, industry,
income, healthcare access, geology, or spatial autocorrelation and are not yet
evidence of a soil-metal pathway.

The EPA radon-zone-only screen also produced an implausible negative lung
gradient after state and rurality adjustment. EPA zones are building-code
planning categories, not household exposure measurements; this result is a
warning that county categories are too coarse for radon disambiguation by
themselves.

## Interpretation

There is not yet a defensible “cluster around the water pump” moment. The current
maps are useful for finding candidate regions, but a candidate cluster must
persist after adding smoking, poverty/income, race/ethnicity, screening and care
access, industrial/mining activity, drinking-water source, population weighting,
spatial residual structure, and leave-one-state-out checks.

Machine-readable results are written locally to:

- `backend/output/cancer_soil/radon_screen.csv`
- `backend/output/cancer_soil/joint_exposure_screen.csv`
- `backend/output/cancer_soil/metal_factor_loadings.csv`
- `backend/output/cancer_soil/county_cancer_exposures.geojson`
