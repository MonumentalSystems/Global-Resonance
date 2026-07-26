#!/usr/bin/env python3
"""Build a compact county GeoJSON containing outcomes and exposure layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .exposure_screen import load_covariates
from .pipeline import DATA_DIR, OUTPUT_DIR


def build(counties: Path, cancer_path: Path, covariate_paths: list[Path], output: Path) -> None:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit("Install backend/cancer_soil_study/requirements-geospatial.txt") from exc

    geography = gpd.read_file(counties)
    geography["fips"] = geography["GEOID"].astype(str).str.zfill(5)
    cancer = pd.read_csv(cancer_path, dtype={"fips": str})
    cancer["fips"] = cancer["fips"].str.zfill(5)
    rates = cancer.pivot_table(index="fips", columns="site", values="incidence_rate", aggfunc="first")
    rates.columns = [f"cancer_{column}" for column in rates.columns]
    radon = cancer.groupby("fips", as_index=True)["radon_zone"].first().rename("radon_zone")
    covariates, _ = load_covariates(covariate_paths)
    attributes = rates.join(radon).reset_index().merge(covariates, on="fips", how="outer")
    mapped = geography.merge(attributes, on="fips", how="left", validate="one_to_one")
    output.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_file(output, driver="GeoJSON")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counties", type=Path)
    parser.add_argument(
        "--cancer", type=Path, default=DATA_DIR / "normalized" / "county_cancer_radon.csv"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "county_cancer_exposures.geojson")
    parser.add_argument("covariates", nargs="*", type=Path)
    args = parser.parse_args()
    covariates = args.covariates or [
        path
        for path in sorted((DATA_DIR / "normalized").glob("county_*.csv"))
        if "cancer" not in path.name
    ]
    build(args.counties, args.cancer, covariates, args.output)


if __name__ == "__main__":
    main()
