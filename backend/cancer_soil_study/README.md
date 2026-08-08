# Cancer, soil conductivity, metals, and radon study

This module tests whether county cancer incidence covaries with mapped soil
electrical properties after separating three candidate explanations:

1. soil electrical conductivity (USDA SOLUS100 saturated-paste EC);
2. specific soil metals (USGS observed and predicted geochemistry); and
3. radon potential (EPA/USGS), especially for lung cancer.

It is an ecological correlation study. A county association cannot establish an
individual exposure or a cellular electromagnetic mechanism.

## Important claim audit

The 1940 Nature URL is a short review of Georges Lakhovsky's 1939 book, not an
experimental research article. It describes cells as hypothetical high-frequency
electromagnetic resonators but the web-accessible review does not present the
soil/cancer analysis. The geographic claim should therefore be treated as a new,
pre-specified hypothesis rather than a replication of a validated result.

The 2017 Spain study used 861,440 cancer deaths in 7,917 towns, ten metals at
13,317 sample locations, compositional transforms, spatial Bayesian models, and
socio-demographic adjustment. It reported different associations by cancer site
and sex. This pipeline retains cancer sites separately and will use centered
log-ratio metal features rather than testing each raw concentration as if the
parts were independent.

## Run the first real-data screen

```powershell
python -m backend.cancer_soil_study.pipeline --sites all,lung,bladder,brain,colon_rectum,esophagus,liver,pancreas,stomach,thyroid
```

Cached source files go under `data/cancer_soil/` (gitignored). Results go under
`backend/output/cancer_soil/` (gitignored). The first screen is deliberately
limited: it tests EPA radon category with state fixed effects and rurality, using
HC3 robust uncertainty and Benjamini-Hochberg correction across cancer sites.

To emit a globe-ready layer, supply a Census county GeoJSON containing `GEOID`:

```powershell
python -m backend.cancer_soil_study.pipeline --county-geojson path/to/counties.geojson
```

## Add conductivity and metals

Download the USDA SOLUS100 electrical-conductivity raster(s), USGS trace-element
rasters, and Census county polygons. Install the optional GIS stack and summarize
each raster inside the actual county polygon:

```powershell
pip install -r backend/cancer_soil_study/requirements-geospatial.txt
python backend/cancer_soil_study/zonal_covariates.py counties.geojson copper.tif copper data/cancer_soil/normalized/county_copper.csv
python -m backend.cancer_soil_study.exposure_screen
python -m backend.cancer_soil_study.map_layer data/cancer_soil/raw/geography/cb_2024_us_county_500k/cb_2024_us_county_500k.shp
```

Use at least mean, median, and upper-tail exposure. Copper veins or ore deposits
must not be represented by county centroid distance alone; both the proportion of
county area affected and population-weighted exposure should be tested.

## Pre-specified analysis ladder

- Outcome: age-adjusted incidence, separately for all cancer, lung, bladder,
  brain, esophagus, stomach, colorectal, liver, pancreas, and thyroid.
- Base controls: state effects, rurality, age-adjustment already in outcome.
- Full controls: smoking, poverty/income, race/ethnicity composition, population
  density, healthcare access/screening, industry/mining, drinking-water source,
  and spatial smooth/random effect.
- Exposure models: EC only; metals only (centered log-ratio factors); radon only;
  EC + metals + radon; and interactions specified before inspecting results.
- Spatial checks: Moran's I on residuals, spatial block cross-validation, local
  cluster maps with false-discovery control, and leave-one-state-out stability.
- Positive control: radon should be most informative for lung cancer. Negative
  controls: outcomes without a plausible radon pathway and shuffled/spatially
  rotated exposures.
- Sensitivity: surface EC by depth, deep Earth conductivity separately, latency
  windows, urban/rural strata, population-weighted versus area-weighted exposure,
  and measured versus modeled metals.

The “cases cluster around the water pump” moment should require the same cluster
to persist after population denominators, smoothing/suppression, known risk
factors, radon, and spatial autocorrelation are accounted for. A raw choropleth
overlap is only the hypothesis-generating view.
