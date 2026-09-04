#!/usr/bin/env python3
"""Evaluate fixed 3/6/12-hour leads on the annual VSH artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .annual_vector_spherical_operator import run_controls, storm_masks
    from .geomagnetic_operator_dataset import save_dataset
except ImportError:
    from annual_vector_spherical_operator import run_controls, storm_masks
    from geomagnetic_operator_dataset import save_dataset


DEFAULT_HORIZONS = (3, 6, 12)


def parse_horizons(text: str) -> tuple[int, ...]:
    horizons = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    return horizons


def run_multihorizon_experiment(
    source: Path,
    output: Path,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict:
    horizons = tuple(int(value) for value in horizons)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must be positive")
    with np.load(source) as source_data:
        source_metadata = json.loads(source_data["metadata_json"].item())
        coefficients = source_data["coefficients"].copy()
        coefficient_mask = source_data["coefficient_mask"].copy()
        drivers = source_data["drivers"].copy()
        driver_mask = source_data["driver_mask"].copy()
        kp = source_data["kp_evaluation_only"].copy()
        dst = source_data["dst_evaluation_only"].copy()
        evaluation_index_mask = source_data["evaluation_index_mask"].copy()

    labels = tuple(tuple(label) for label in source_metadata["labels"])
    splits = {
        name: tuple(indices) for name, indices in source_metadata["splits"].items()
    }
    evaluation_masks = storm_masks(kp, dst, evaluation_index_mask)
    controls = {
        str(horizon): run_controls(
            coefficients,
            coefficient_mask,
            drivers,
            driver_mask,
            splits,
            labels,
            evaluation_masks,
            lead_hours=horizon,
        )
        for horizon in horizons
    }
    metadata = {
        "schema_version": 1,
        "dataset_kind": "annual_vector_spherical_multihorizon_controls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(source),
        "source_created_at": source_metadata["created_at"],
        "horizons_hours": horizons,
        "splits": splits,
        "split_timestamps": source_metadata["split_timestamps"],
        "target": "VSH coefficient change at forecast origin plus lead",
        "controls": controls,
        "comparison": (
            "All horizons reuse the frozen model family, candidate scales, "
            "calendar split, masks, and 24-hour causal warm-up from the one-hour gate."
        ),
        "multiplicity_warning": (
            "Three horizons and three sectors are exploratory follow-ups. "
            "Do not select a claimed effect from unadjusted point estimates."
        ),
        "warning": source_metadata["warning"],
    }
    save_dataset(
        {"horizons_hours": np.asarray(horizons, dtype=np.int16), "metadata": metadata},
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
        default=root / "geomagnetic_vector_global_2024_multihorizon.npz",
    )
    parser.add_argument("--horizons", default="3,6,12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_multihorizon_experiment(
        args.source, args.output, parse_horizons(args.horizons)
    )
    print(json.dumps(metadata, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
