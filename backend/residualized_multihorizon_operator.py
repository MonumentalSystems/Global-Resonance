#!/usr/bin/env python3
"""Test temporal controls after training-only calendar residualization.

The calendar baseline is a fixed low-dimensional harmonic model fitted only on
the H1 training block.  It uses neither geomagnetic activity indices nor held-
out coefficients.  Raw and residualized 1/3/6/12-hour controls are then run
through the same frozen linear gate for a direct diagnostic comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from .annual_vector_spherical_operator import run_controls, storm_masks
    from .geomagnetic_operator_dataset import save_dataset
    from .multihorizon_vector_spherical_operator import parse_horizons
    from .operator_data_sources import parse_utc
except ImportError:
    from annual_vector_spherical_operator import run_controls, storm_masks
    from geomagnetic_operator_dataset import save_dataset
    from multihorizon_vector_spherical_operator import parse_horizons
    from operator_data_sources import parse_utc


DEFAULT_HORIZONS = (1, 3, 6, 12)
CALENDAR_RIDGE = 1e-6
HUBER_DELTA = 2.5
IRLS_ITERATIONS = 5


def calendar_harmonic_design(
    timestamps: Sequence[datetime],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return a fixed daily/seasonal basis with seasonal daily modulation."""

    if not timestamps:
        raise ValueError("timestamps must not be empty")
    seconds_per_day = 24.0 * 60.0 * 60.0
    daily_phase = np.asarray(
        [
            2.0
            * np.pi
            * (
                value.hour * 3600.0
                + value.minute * 60.0
                + value.second
                + value.microsecond / 1e6
            )
            / seconds_per_day
            for value in timestamps
        ]
    )
    annual_phase = np.asarray(
        [
            2.0
            * np.pi
            * (
                value.timetuple().tm_yday
                - 1
                + (
                    value.hour * 3600.0
                    + value.minute * 60.0
                    + value.second
                    + value.microsecond / 1e6
                )
                / seconds_per_day
            )
            / 365.2425
            for value in timestamps
        ]
    )
    columns = [np.ones(len(timestamps), dtype=np.float64)]
    names = ["intercept"]
    daily_columns: list[tuple[str, np.ndarray]] = []
    for harmonic, period in ((1, 24), (2, 12)):
        for function_name, function in (("sin", np.sin), ("cos", np.cos)):
            name = f"{function_name}_{period}h"
            values = function(harmonic * daily_phase)
            columns.append(values)
            names.append(name)
            daily_columns.append((name, values))
    annual_columns = (
        ("sin_annual", np.sin(annual_phase)),
        ("cos_annual", np.cos(annual_phase)),
    )
    for name, values in annual_columns:
        columns.append(values)
        names.append(name)
    for daily_name, daily_values in daily_columns:
        for annual_name, annual_values in annual_columns:
            columns.append(daily_values * annual_values)
            names.append(f"{daily_name}_x_{annual_name}")
    return np.column_stack(columns), tuple(names)


def fit_robust_calendar_baseline(
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    design: np.ndarray,
    train_split: tuple[int, int],
    *,
    ridge: float = CALENDAR_RIDGE,
    huber_delta: float = HUBER_DELTA,
    iterations: int = IRLS_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a multivariate robust harmonic regression on training rows only."""

    if coefficients.shape != coefficient_mask.shape or coefficients.ndim != 2:
        raise ValueError("coefficients and mask must match as (time, coefficient)")
    if design.ndim != 2 or len(design) != len(coefficients):
        raise ValueError("design must have the same time axis as coefficients")
    if ridge < 0.0 or huber_delta <= 0.0 or iterations <= 0:
        raise ValueError("invalid robust-fit hyperparameters")
    start, end = train_split
    if not 0 <= start < end <= len(coefficients):
        raise ValueError("invalid training split")
    train_rows = np.zeros(len(coefficients), dtype=bool)
    train_rows[start:end] = True
    train_rows &= coefficient_mask.all(axis=1)
    if train_rows.sum() <= design.shape[1]:
        raise ValueError("insufficient complete training rows for calendar fit")

    x = design[train_rows].astype(np.float64)
    y = coefficients[train_rows].astype(np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    row_weights = np.ones(len(x), dtype=np.float64)
    output_median = np.median(y, axis=0)
    output_scale = 1.4826 * np.median(np.abs(y - output_median), axis=0)
    std_fallback = y.std(axis=0)
    output_scale = np.where(output_scale >= 1e-8, output_scale, std_fallback)
    output_scale = np.maximum(output_scale, 1e-8)

    def solve(weights: np.ndarray) -> np.ndarray:
        weighted_x = weights[:, None] * x
        return np.linalg.solve(
            x.T @ weighted_x + ridge * penalty,
            x.T @ (weights[:, None] * y),
        )

    weights = solve(row_weights)
    for _ in range(iterations):
        normalized = (x @ weights - y) / output_scale
        score = np.sqrt(np.mean(normalized**2, axis=1))
        row_weights = np.minimum(1.0, huber_delta / np.maximum(score, 1e-12))
        weights = solve(row_weights)

    baseline = design.astype(np.float64) @ weights
    residuals = np.where(
        coefficient_mask,
        coefficients.astype(np.float64) - baseline,
        0.0,
    )
    fit_error = x @ weights - y
    diagnostics = {
        "fit_scope": "complete coefficient rows inside the H1 training split only",
        "train_split": [int(start), int(end)],
        "train_rows": int(train_rows.sum()),
        "design_features": int(design.shape[1]),
        "outputs": int(coefficients.shape[1]),
        "parameters": int(design.shape[1] * coefficients.shape[1]),
        "ridge": float(ridge),
        "huber_delta": float(huber_delta),
        "irls_iterations": int(iterations),
        "minimum_row_weight": float(row_weights.min()),
        "median_row_weight": float(np.median(row_weights)),
        "fraction_downweighted": float(np.mean(row_weights < 1.0)),
        "training_mse": float(np.mean(fit_error**2)),
        "uses_activity_indices": False,
        "uses_validation_or_test_coefficients": False,
    }
    return baseline, residuals, diagnostics


def _model_comparison(raw: dict[str, Any], residualized: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("markov", "single_pole", "orthogonal_recurrent"):
        raw_model = raw["models"][name]
        residual_model = residualized["models"][name]
        result[name] = {
            "raw_mse": raw_model["mse"],
            "residualized_reconstructed_mse": residual_model["mse"],
            "change_from_raw_fraction": 1.0
            - residual_model["mse"] / raw_model["mse"],
        }
    result["raw_single_pole_improvement_fraction"] = raw[
        "single_pole_improvement_fraction"
    ]
    result["residualized_single_pole_improvement_fraction"] = residualized[
        "single_pole_improvement_fraction"
    ]
    return result


def run_residualized_experiment(
    source: Path,
    output: Path,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    horizons = tuple(int(value) for value in horizons)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must be positive")
    with np.load(source) as source_data:
        source_metadata = json.loads(source_data["metadata_json"].item())
        timestamp_strings = source_data["timestamps"].copy()
        coefficients = source_data["coefficients"].copy()
        coefficient_mask = source_data["coefficient_mask"].copy()
        drivers = source_data["drivers"].copy()
        driver_mask = source_data["driver_mask"].copy()
        kp = source_data["kp_evaluation_only"].copy()
        dst = source_data["dst_evaluation_only"].copy()
        evaluation_index_mask = source_data["evaluation_index_mask"].copy()

    timestamps = [parse_utc(str(value)) for value in timestamp_strings]
    labels = tuple(tuple(label) for label in source_metadata["labels"])
    splits = {
        name: tuple(indices) for name, indices in source_metadata["splits"].items()
    }
    design, basis_names = calendar_harmonic_design(timestamps)
    baseline, residuals, fit_diagnostics = fit_robust_calendar_baseline(
        coefficients,
        coefficient_mask,
        design,
        splits["train"],
    )
    evaluation_masks = storm_masks(kp, dst, evaluation_index_mask)
    raw_controls = {}
    residualized_controls = {}
    comparisons = {}
    for horizon in horizons:
        key = str(horizon)
        raw_controls[key] = run_controls(
            coefficients,
            coefficient_mask,
            drivers,
            driver_mask,
            splits,
            labels,
            evaluation_masks,
            lead_hours=horizon,
        )
        residualized_controls[key] = run_controls(
            residuals,
            coefficient_mask,
            drivers,
            driver_mask,
            splits,
            labels,
            evaluation_masks,
            lead_hours=horizon,
        )
        comparisons[key] = _model_comparison(
            raw_controls[key], residualized_controls[key]
        )

    metadata = {
        "schema_version": 1,
        "dataset_kind": "annual_vector_spherical_calendar_residualized_controls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(source),
        "source_created_at": source_metadata["created_at"],
        "horizons_hours": horizons,
        "splits": splits,
        "split_timestamps": source_metadata["split_timestamps"],
        "calendar_basis": {
            "names": basis_names,
            "description": (
                "Intercept; 24h and 12h sine/cosine terms; one annual "
                "sine/cosine pair; and daily-by-annual interactions."
            ),
            "fit": fit_diagnostics,
        },
        "raw_controls": raw_controls,
        "residualized_controls": residualized_controls,
        "raw_vs_residualized": comparisons,
        "metric_note": (
            "Residual-space MSE equals reconstructed original-coefficient MSE "
            "because the deterministic target-time calendar baseline would be "
            "added back to every residual prediction."
        ),
        "activity_label_guard": (
            "Kp and Dst are used only to stratify held-out errors after prediction; "
            "they are absent from the baseline fit and all predictors."
        ),
        "comparison": (
            "Raw and residualized controls share the model family, candidate scales, "
            "ridge grid, calendar split, masks, and 24-hour causal warm-up."
        ),
        "warning": source_metadata["warning"],
    }
    save_dataset(
        {
            "timestamps": timestamp_strings,
            "calendar_design": design.astype(np.float32),
            "calendar_baseline": baseline.astype(np.float32),
            "residual_coefficients": residuals.astype(np.float32),
            "coefficient_mask": coefficient_mask,
            "horizons_hours": np.asarray(horizons, dtype=np.int16),
            "metadata": metadata,
        },
        output,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent.parent / "data" / "operator"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "geomagnetic_vector_global_2024_hourly.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "geomagnetic_vector_global_2024_residualized_multihorizon.npz",
    )
    parser.add_argument("--horizons", default="1,3,6,12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_residualized_experiment(
        args.source, args.output, parse_horizons(args.horizons)
    )
    print(json.dumps(metadata, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
