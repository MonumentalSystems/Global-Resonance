#!/usr/bin/env python3
"""Jointly screen county raster covariates while controlling for radon and geography."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .pipeline import DATA_DIR, OUTPUT_DIR, _bh_adjust, _hc3_ols


def load_covariates(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    merged = None
    columns: list[str] = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"fips": str})
        frame["fips"] = frame["fips"].str.zfill(5)
        means = [column for column in frame if column.endswith("_mean")]
        if len(means) != 1:
            raise ValueError(f"{path} must have exactly one *_mean column")
        columns.extend(means)
        keep = frame[["fips", means[0]]]
        merged = keep if merged is None else merged.merge(keep, on="fips", how="outer", validate="one_to_one")
    if merged is None:
        raise ValueError("At least one covariate CSV is required")
    return merged, columns


def screen(cancer_path: Path, covariate_paths: list[Path], output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    cancer = pd.read_csv(cancer_path, dtype={"fips": str, "state_fips": str})
    cancer["fips"] = cancer["fips"].str.zfill(5)
    cancer["state_fips"] = cancer["state_fips"].str.zfill(2)
    covariates, exposure_columns = load_covariates(covariate_paths)
    conductivity_columns = [column for column in exposure_columns if "conductivity" in column]
    metal_columns = [column for column in exposure_columns if column not in conductivity_columns]
    if len(conductivity_columns) > 1:
        raise ValueError("Use one conductivity depth per screen so its coefficient is identifiable")
    complete = covariates.dropna(subset=exposure_columns).copy()
    log_exposure = np.log(complete[metal_columns].clip(lower=1e-12))
    standardized = (log_exposure - log_exposure.mean()) / log_exposure.std(ddof=0)
    _, _, vt = np.linalg.svd(standardized.to_numpy(float), full_matrices=False)
    scores = standardized.to_numpy(float) @ vt.T
    scores = scores / scores.std(axis=0, ddof=0)
    factor_columns = [f"metal_pc{index + 1}" for index in range(scores.shape[1])]
    complete[factor_columns] = scores
    model_exposures = factor_columns.copy()
    if conductivity_columns:
        conductivity = conductivity_columns[0]
        logged = np.log(complete[conductivity].clip(lower=1e-12))
        complete["conductivity_z"] = (logged - logged.mean()) / logged.std(ddof=0)
        model_exposures.insert(0, "conductivity_z")
    data = cancer.merge(complete[["fips", *model_exposures]], on="fips", how="left", validate="many_to_one")
    rows = []
    for site, group in data.groupby("site"):
        required = ["incidence_rate", "radon_potential", *model_exposures]
        valid = group.dropna(subset=required).copy()
        state = pd.get_dummies(valid["state_fips"], drop_first=True, dtype=float)
        controls = pd.concat(
            [
                pd.Series(1.0, index=valid.index, name="intercept"),
                valid["radon_potential"].astype(float),
                valid["rural_urban"].eq("Rural").astype(float).rename("rural"),
                state,
            ],
            axis=1,
        ).to_numpy(float)
        # Fit conductivity and all orthogonal metal factors together. This is the
        # first disambiguation model; spatial and socioeconomic controls come next.
        design = np.column_stack([controls, valid[model_exposures].to_numpy(float)])
        beta, se = _hc3_ols(valid["incidence_rate"].to_numpy(float), design)
        for exposure_index, exposure in enumerate(model_exposures):
            coefficient = beta[-len(model_exposures) + exposure_index]
            standard_error = se[-len(model_exposures) + exposure_index]
            pvalue = 2 * stats.norm.sf(abs(coefficient / standard_error))
            rows.append(
                {
                    "site": site,
                    "exposure": exposure,
                    "n_counties": len(valid),
                    "rate_change_per_exposure_sd": coefficient,
                    "hc3_se": standard_error,
                    "p": pvalue,
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh"] = _bh_adjust(result["p"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "joint_exposure_screen.csv", index=False)
    loading_rows = [
        {
            "factor": factor_column,
            "metal": exposure.removesuffix("_mean"),
            "loading": vt[factor_index, exposure_index],
        }
        for factor_index, factor_column in enumerate(factor_columns)
        for exposure_index, exposure in enumerate(metal_columns)
    ]
    pd.DataFrame(loading_rows).to_csv(output_dir / "metal_factor_loadings.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "covariates",
        nargs="*",
        type=Path,
        default=sorted((DATA_DIR / "normalized").glob("county_*.csv")),
    )
    parser.add_argument(
        "--cancer",
        type=Path,
        default=DATA_DIR / "normalized" / "county_cancer_radon.csv",
    )
    args = parser.parse_args()
    covariates = [
        path for path in args.covariates if "cancer" not in path.name and path.name != "county_cancer_radon.csv"
    ]
    result = screen(args.cancer, covariates)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
