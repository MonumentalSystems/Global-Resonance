#!/usr/bin/env python3
"""Replicate the frozen 2024 residual pole clue on an independent year.

The confirmatory target is fixed to the discovery result: a 24-hour
exponential driver state, a 6-hour lead, and the poloidal VSH sector.  Each
year independently fits the same training-only calendar baseline and selects
ridge strength on Q3 before one Q4 evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .annual_vector_spherical_operator import (
        COMMON_WARMUP_HOURS,
        common_ready_mask,
        evaluate_control,
        exponential_state_features,
        paired_daily_block_bootstrap,
        paired_weekly_block_bootstrap,
        select_ridge,
        storm_masks,
    )
    from .geomagnetic_operator_dataset import save_dataset
    from .operator_data_sources import parse_utc
    from .residualized_multihorizon_operator import (
        calendar_harmonic_design,
        fit_robust_calendar_baseline,
    )
    from .vector_spherical_harmonics import sector_indices
except ImportError:
    from annual_vector_spherical_operator import (
        COMMON_WARMUP_HOURS,
        common_ready_mask,
        evaluate_control,
        exponential_state_features,
        paired_daily_block_bootstrap,
        paired_weekly_block_bootstrap,
        select_ridge,
        storm_masks,
    )
    from geomagnetic_operator_dataset import save_dataset
    from operator_data_sources import parse_utc
    from residualized_multihorizon_operator import (
        calendar_harmonic_design,
        fit_robust_calendar_baseline,
    )
    from vector_spherical_harmonics import sector_indices


FROZEN_LEAD_HOURS = 6
FROZEN_POLE_HALF_LIFE_HOURS = 24.0
FROZEN_SECTOR = "poloidal"


def _strip_private(value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key in list(value):
        if key.startswith("_"):
            value.pop(key)
        else:
            _strip_private(value[key])


def evaluate_frozen_hypothesis(source: Path) -> dict[str, Any]:
    """Evaluate one fixed sector/lead/pole hypothesis from an annual artifact."""

    with np.load(source) as source_data:
        metadata = json.loads(source_data["metadata_json"].item())
        timestamp_strings = source_data["timestamps"].copy()
        station_latitudes = source_data["station_latitudes"].copy()
        station_longitudes = source_data["station_longitudes"].copy()
        coefficients = source_data["coefficients"].copy()
        coefficient_mask = source_data["coefficient_mask"].copy()
        drivers = source_data["drivers"].copy()
        driver_mask = source_data["driver_mask"].copy()
        kp = source_data["kp_evaluation_only"].copy()
        dst = source_data["dst_evaluation_only"].copy()
        evaluation_index_mask = source_data["evaluation_index_mask"].copy()

    labels = tuple(tuple(label) for label in metadata["labels"])
    splits = {name: tuple(indices) for name, indices in metadata["splits"].items()}
    timestamps = [parse_utc(str(value)) for value in timestamp_strings]
    calendar_design, basis_names = calendar_harmonic_design(timestamps)
    _, residuals, calendar_fit = fit_robust_calendar_baseline(
        coefficients,
        coefficient_mask,
        calendar_design,
        splits["train"],
    )
    evaluation_masks = storm_masks(kp, dst, evaluation_index_mask)
    targets = sector_indices(labels)[FROZEN_SECTOR]
    ready = common_ready_mask(driver_mask, COMMON_WARMUP_HOURS)
    markov_features = drivers.astype(np.float64)
    pole_features = exponential_state_features(
        drivers, driver_mask, FROZEN_POLE_HALF_LIFE_HOURS
    )

    markov_ridge, markov_validation_mse = select_ridge(
        markov_features,
        ready,
        residuals,
        coefficient_mask,
        splits,
        targets,
        FROZEN_LEAD_HOURS,
    )
    pole_ridge, pole_validation_mse = select_ridge(
        pole_features,
        ready,
        residuals,
        coefficient_mask,
        splits,
        targets,
        FROZEN_LEAD_HOURS,
    )
    markov = evaluate_control(
        markov_features,
        ready,
        residuals,
        coefficient_mask,
        splits,
        targets,
        labels,
        evaluation_masks,
        markov_ridge,
        FROZEN_LEAD_HOURS,
    )
    pole = evaluate_control(
        pole_features,
        ready,
        residuals,
        coefficient_mask,
        splits,
        targets,
        labels,
        evaluation_masks,
        pole_ridge,
        FROZEN_LEAD_HOURS,
    )
    markov["selection_validation_mse"] = markov_validation_mse
    pole["selection_validation_mse"] = pole_validation_mse
    daily = paired_daily_block_bootstrap(markov, pole, evaluation_masks)
    weekly = paired_weekly_block_bootstrap(markov, pole, evaluation_masks)
    improvement = 1.0 - pole["mse"] / markov["mse"]
    result = {
        "year": timestamps[0].year,
        "source_artifact": str(source),
        "source_created_at": metadata["created_at"],
        "station_codes": metadata["station_codes"],
        "station_latitudes": station_latitudes.tolist(),
        "station_longitudes": station_longitudes.tolist(),
        "lmax": metadata["lmax"],
        "labels": metadata["labels"],
        "coordinate_convention": metadata["coordinate_convention"],
        "source_cadence_seconds": metadata["source_cadence_seconds"],
        "target_cadence_seconds": metadata["target_cadence_seconds"],
        "coverage": metadata["coverage"],
        "design_condition_numbers": metadata["design_condition_numbers"],
        "storm_hours": metadata["storm_hours"],
        "split_timestamps": metadata["split_timestamps"],
        "calendar_basis_names": basis_names,
        "calendar_fit": calendar_fit,
        "models": {"markov": markov, "fixed_24h_pole": pole},
        "improvement_fraction": improvement,
        "validation_improvement_fraction": 1.0
        - pole_validation_mse / markov_validation_mse,
        "paired_daily_bootstrap": daily,
        "paired_weekly_bootstrap": weekly,
    }
    _strip_private(result)
    return result


def replication_decision(
    discovery: dict[str, Any], replication: dict[str, Any]
) -> dict[str, Any]:
    """Classify direction, validation agreement, and interval confirmation."""

    daily = replication["paired_daily_bootstrap"]["all"]
    weekly = replication["paired_weekly_bootstrap"]["all"]
    directional = (
        discovery["improvement_fraction"] > 0.0
        and replication["improvement_fraction"] > 0.0
    )
    validation_agrees = replication["validation_improvement_fraction"] > 0.0
    daily_confirmed = bool(daily["ci95"] and daily["ci95"][0] > 0.0)
    weekly_confirmed = bool(weekly["ci95"] and weekly["ci95"][0] > 0.0)
    confirmed = directional and validation_agrees and daily_confirmed and weekly_confirmed
    if confirmed:
        status = "confirmed"
    elif directional and validation_agrees:
        status = "directional_not_interval_confirmed"
    else:
        status = "not_replicated"
    return {
        "status": status,
        "same_test_direction": directional,
        "replication_validation_agrees": validation_agrees,
        "replication_daily_ci95_excludes_zero": daily_confirmed,
        "replication_weekly_ci95_excludes_zero": weekly_confirmed,
        "effect_ratio_replication_to_discovery": (
            replication["improvement_fraction"]
            / discovery["improvement_fraction"]
            if discovery["improvement_fraction"] != 0.0
            else None
        ),
    }


def run_crossyear_replication(
    discovery_source: Path,
    replication_source: Path,
    output: Path,
) -> dict[str, Any]:
    discovery = evaluate_frozen_hypothesis(discovery_source)
    replication = evaluate_frozen_hypothesis(replication_source)
    compatibility_fields = (
        "station_codes",
        "station_latitudes",
        "station_longitudes",
        "lmax",
        "labels",
        "coordinate_convention",
        "source_cadence_seconds",
        "target_cadence_seconds",
        "calendar_basis_names",
    )
    mismatches = [
        field
        for field in compatibility_fields
        if discovery[field] != replication[field]
    ]
    if mismatches:
        raise ValueError(
            "discovery and replication artifacts differ in frozen fields: "
            + ", ".join(mismatches)
        )
    result = {
        "schema_version": 1,
        "dataset_kind": "crossyear_frozen_vector_spherical_replication",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_hypothesis": {
            "sector": FROZEN_SECTOR,
            "lead_hours": FROZEN_LEAD_HOURS,
            "pole_half_life_hours": FROZEN_POLE_HALF_LIFE_HOURS,
            "baseline": (
                "The same fixed 15-term robust calendar procedure is fitted "
                "independently on each year's H1 block."
            ),
            "ridge_selection": "Each year uses H1 training and Q3 selection only.",
            "evaluation": "One untouched Q4 evaluation per year.",
        },
        "discovery": discovery,
        "replication": replication,
        "compatibility": {
            "matched": True,
            "fields": compatibility_fields,
        },
        "decision": replication_decision(discovery, replication),
        "warning": (
            "Research diagnostic, not an operational geomagnetic or hazard forecast."
        ),
    }
    save_dataset(
        {
            "years": np.asarray([discovery["year"], replication["year"]]),
            "metadata": result,
        },
        output,
    )
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent.parent / "data" / "operator"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-source",
        type=Path,
        default=root / "geomagnetic_vector_global_2024_hourly.npz",
    )
    parser.add_argument(
        "--replication-source",
        type=Path,
        default=root / "geomagnetic_vector_global_2023_hourly.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "geomagnetic_vector_crossyear_replication_2023_2024.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_crossyear_replication(
        args.discovery_source, args.replication_source, args.output
    )
    print(json.dumps(result, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
