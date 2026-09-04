#!/usr/bin/env python3
"""Evaluate sector-specific fixed poles on vector spherical harmonics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .geomagnetic_operator_dataset import baseline_metrics, save_dataset
    from .historical_geomagnetic_operator import (
        DEFAULT_HALF_LIVES,
        _ridge_metric,
        causal_pole_features,
        historical_ridge_controls,
    )
    from .vector_spherical_harmonics import (
        fit_vector_network_coefficients,
        geomagnetic_xyz_to_spherical,
        sector_indices,
    )
except ImportError:
    from geomagnetic_operator_dataset import baseline_metrics, save_dataset
    from historical_geomagnetic_operator import (
        DEFAULT_HALF_LIVES,
        _ridge_metric,
        causal_pole_features,
        historical_ridge_controls,
    )
    from vector_spherical_harmonics import (
        fit_vector_network_coefficients,
        geomagnetic_xyz_to_spherical,
        sector_indices,
    )


def _select_sector_models(
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    drivers: np.ndarray,
    driver_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    sectors: dict[str, np.ndarray],
    half_lives: Iterable[float] = DEFAULT_HALF_LIVES,
) -> dict[str, Any]:
    """Select equal-size Markov or one-pole models independently per sector."""

    markov_sectors = {}
    pole_sectors = {}
    gated_sectors = {}
    for sector, indices in sectors.items():
        markov_sectors[sector] = _ridge_metric(
            drivers,
            driver_mask.all(axis=1),
            coefficients,
            coefficient_mask,
            splits,
            target_indices=indices,
        )

        candidates = []
        for half_life in half_lives:
            features, feature_mask = causal_pole_features(
                drivers,
                driver_mask,
                half_lives=(half_life,),
            )
            candidate = _ridge_metric(
                features,
                feature_mask,
                coefficients,
                coefficient_mask,
                splits,
                target_indices=indices,
            )
            candidate["half_life_hours"] = float(half_life)
            candidates.append(candidate)
        available = [row for row in candidates if row.get("status") == "ok"]
        pole_sectors[sector] = (
            min(available, key=lambda row: row["selection_validation_mse"])
            if available
            else {"status": "unavailable", "reason": "no valid half-life"}
        )

        gated_candidates = [{**markov_sectors[sector], "direct_driver_gate": 1.0}]
        for half_life in half_lives:
            pole_features, pole_mask = causal_pole_features(
                drivers,
                driver_mask,
                half_lives=(half_life,),
            )
            for direct_gate in (0.0, 0.25, 0.5, 0.75):
                features = direct_gate * drivers + (1.0 - direct_gate) * pole_features
                candidate = _ridge_metric(
                    features,
                    pole_mask,
                    coefficients,
                    coefficient_mask,
                    splits,
                    target_indices=indices,
                )
                candidate["half_life_hours"] = float(half_life)
                candidate["direct_driver_gate"] = direct_gate
                gated_candidates.append(candidate)
        available_gates = [
            row for row in gated_candidates if row.get("status") == "ok"
        ]
        gated_sectors[sector] = min(
            available_gates,
            key=lambda row: row["selection_validation_mse"],
        )

    def aggregate(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if any(row.get("status") != "ok" for row in rows.values()):
            return {"status": "unavailable", "sectors": rows}
        weighted_errors = []
        weights = []
        for sector, row in rows.items():
            weight = len(sectors[sector]) * row["test_rows"]
            weighted_errors.append(row["mse"] * weight)
            weights.append(weight)
        return {
            "status": "ok",
            "mse": float(sum(weighted_errors) / sum(weights)),
            "parameters": int(sum(row["parameters"] for row in rows.values())),
            "sectors": rows,
        }

    markov = aggregate(markov_sectors)
    poles = aggregate(pole_sectors)
    gated = aggregate(gated_sectors)
    result = {
        "sector_specific_markov": markov,
        "sector_specific_single_poles": poles,
        "sector_specific_gated_poles": gated,
    }
    if markov.get("status") == poles.get("status") == "ok":
        result["pole_improvement_fraction"] = 1.0 - poles["mse"] / markov["mse"]
        result["parameter_matched"] = markov["parameters"] == poles["parameters"]
    if markov.get("status") == gated.get("status") == "ok":
        result["gated_pole_improvement_fraction"] = 1.0 - (
            gated["mse"] / markov["mse"]
        )
        result["gated_parameter_matched"] = (
            markov["parameters"] == gated["parameters"]
        )
    return result


def run_vector_pilot(source: Path, output: Path, lmax: int = 2) -> dict[str, Any]:
    with np.load(source) as source_data:
        source_metadata = json.loads(source_data["metadata_json"].item())
        timestamps = source_data["timestamps"].copy()
        drivers = source_data["drivers"].copy()
        driver_mask = source_data["driver_mask"].copy()
        station_deltas = source_data["station_deltas"].copy()
        station_delta_mask = source_data["station_delta_mask"].copy()
        latitudes = source_data["station_latitudes"].copy()
        longitudes = source_data["station_longitudes"].copy()

    spherical_values, spherical_mask = geomagnetic_xyz_to_spherical(
        station_deltas, station_delta_mask
    )
    coefficients, coefficient_mask, labels, condition_summary = (
        fit_vector_network_coefficients(
            spherical_values,
            spherical_mask,
            latitudes,
            longitudes,
            lmax=lmax,
        )
    )
    splits = {
        key: tuple(value) for key, value in source_metadata["splits"].items()
    }
    sectors = sector_indices(labels)
    controls = baseline_metrics(
        coefficients,
        coefficient_mask,
        drivers,
        driver_mask,
        splits,
    )
    controls["shared_sector_models"] = historical_ridge_controls(
        coefficients,
        coefficient_mask,
        drivers,
        driver_mask,
        splits,
    )
    shared = controls["shared_sector_models"]
    for model_name in (
        "markov_ridge",
        "parameter_matched_single_pole_ridge",
        "fixed_pole_ridge",
    ):
        model = shared[model_name]
        per_target_values = model.pop("mse_by_target_index", None)
        if per_target_values is None:
            continue
        per_target = np.asarray(per_target_values)
        model["mse_by_sector"] = {
            sector: float(per_target[indices].mean())
            for sector, indices in sectors.items()
        }
    shared["single_pole_improvement_by_sector"] = {
        sector: 1.0
        - shared["parameter_matched_single_pole_ridge"]["mse_by_sector"][sector]
        / shared["markov_ridge"]["mse_by_sector"][sector]
        for sector in sectors
    }
    shared["fixed_pole_improvement_by_sector"] = {
        sector: 1.0
        - shared["fixed_pole_ridge"]["mse_by_sector"][sector]
        / shared["markov_ridge"]["mse_by_sector"][sector]
        for sector in sectors
    }
    controls["sector_specific_models"] = _select_sector_models(
        coefficients,
        coefficient_mask,
        drivers,
        driver_mask,
        splits,
        sectors,
    )

    metadata = {
        "schema_version": 1,
        "dataset_kind": "historical_vector_spherical_harmonics",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(source),
        "source_created_at": source_metadata["created_at"],
        "source_provenance": source_metadata["provenance"],
        "start": source_metadata["start"],
        "end_exclusive": source_metadata["end_exclusive"],
        "lmax": lmax,
        "coordinate_convention": {
            "source": ["X north", "Y east", "Z down"],
            "vsh": ["radial outward = -Z", "theta south = -X", "phi east = Y"],
        },
        "basis_definition": {
            "radial": "Y_lm e_r",
            "poloidal": "surface_gradient(Y_lm) / sqrt(l(l+1))",
            "toroidal": "e_r cross surface_gradient(Y_lm) / sqrt(l(l+1))",
        },
        "labels": labels,
        "sector_indices": {key: value.tolist() for key, value in sectors.items()},
        "design_condition_numbers": condition_summary,
        "coefficient_fraction": float(coefficient_mask.mean()),
        "splits": splits,
        "controls": controls,
        "warning": (
            "Low-degree coefficients are irregular-network descriptors. This "
            "surface VSH analysis does not impose a divergence-free 3D field."
        ),
    }
    dataset = {
        "timestamps": timestamps,
        "drivers": drivers,
        "driver_mask": driver_mask,
        "spherical_station_deltas": spherical_values,
        "spherical_station_mask": spherical_mask,
        "station_latitudes": latitudes,
        "station_longitudes": longitudes,
        "coefficients": coefficients,
        "coefficient_mask": coefficient_mask,
        "metadata": metadata,
    }
    save_dataset(dataset, output)
    return metadata


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent.parent / "data" / "operator"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "geomagnetic_historical_2024_january.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "geomagnetic_vector_historical_2024_january.npz",
    )
    parser.add_argument("--lmax", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_vector_pilot(args.source, args.output, lmax=args.lmax)
    print(json.dumps(metadata, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
