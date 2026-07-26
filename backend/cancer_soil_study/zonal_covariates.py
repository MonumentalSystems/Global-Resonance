#!/usr/bin/env python3
"""Summarize conductivity or metals rasters within county polygons.

Install the optional geospatial dependencies in requirements-geospatial.txt.
The output is one area-weighted raster summary per county, suitable for joining
to pipeline.py output by FIPS. A raster must have a valid CRS and nodata value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(counties_path: Path, raster_path: Path, variable: str, output_path: Path) -> None:
    try:
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask
    except ImportError as exc:
        raise SystemExit("Install backend/cancer_soil_study/requirements-geospatial.txt") from exc

    counties = gpd.read_file(counties_path)
    if "GEOID" not in counties:
        raise ValueError("County file must contain a GEOID field")
    rows = []
    with rasterio.open(raster_path) as src:
        projected = counties.to_crs(src.crs)
        for _, county in projected.iterrows():
            try:
                clipped, _ = mask(src, [county.geometry], crop=True, filled=False, indexes=1)
                values = clipped.compressed().astype(float)
            except ValueError:  # polygon does not overlap raster
                values = np.array([], dtype=float)
            rows.append(
                {
                    "fips": str(county["GEOID"]).zfill(5),
                    f"{variable}_mean": np.mean(values) if values.size else np.nan,
                    f"{variable}_median": np.median(values) if values.size else np.nan,
                    f"{variable}_p90": np.quantile(values, 0.9) if values.size else np.nan,
                    f"{variable}_n_cells": values.size,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counties", type=Path)
    parser.add_argument("raster", type=Path)
    parser.add_argument("variable")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.counties, args.raster, args.variable, args.output)


if __name__ == "__main__":
    main()
