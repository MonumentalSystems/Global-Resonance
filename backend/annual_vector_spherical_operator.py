#!/usr/bin/env python3
"""Run the annual INTERMAGNET/OMNI vector-spherical linear gate.

The target is the next hourly change in the degree-2 vector spherical harmonic
coefficients of a global ground-magnetometer network.  Current-driver ARX, a
fixed exponential cavity, and a generic stable orthogonal recurrence use the
same five-dimensional forcing state and therefore the same learned readout
parameter count.  Model and ridge selection use validation only; the final
calendar quarter remains untouched until one final evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np

try:
    from .geomagnetic_operator_dataset import save_dataset, temporal_differences
    from .historical_geomagnetic_operator import (
        DEFAULT_HALF_LIVES,
        OMNI2_DOCUMENTATION_URL,
        OMNI_DRIVER_FEATURES,
        align_omni_drivers,
        load_omni2,
    )
    from .operator_data_sources import parse_utc
    from .vector_spherical_harmonics import (
        fit_vector_network_coefficients,
        geomagnetic_xyz_to_spherical,
        sector_indices,
    )
except ImportError:
    from geomagnetic_operator_dataset import save_dataset, temporal_differences
    from historical_geomagnetic_operator import (
        DEFAULT_HALF_LIVES,
        OMNI2_DOCUMENTATION_URL,
        OMNI_DRIVER_FEATURES,
        align_omni_drivers,
        load_omni2,
    )
    from operator_data_sources import parse_utc
    from vector_spherical_harmonics import (
        fit_vector_network_coefficients,
        geomagnetic_xyz_to_spherical,
        sector_indices,
    )


RIDGE_GRID = tuple(10.0**exponent for exponent in range(-4, 7))
NULL_RIDGE = "infinite_null_limit"
COMMON_WARMUP_HOURS = 24


def hourly_median_network(
    minute_values: np.ndarray,
    minute_mask: np.ndarray,
    *,
    minutes_per_hour: int = 60,
    minimum_samples: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce minute station vectors without treating fills as measurements."""

    if minute_values.shape != minute_mask.shape or minute_values.ndim != 3:
        raise ValueError("expected matching (time, station, component) arrays")
    if len(minute_values) % minutes_per_hour:
        raise ValueError("minute axis must contain complete hours")
    if not 1 <= minimum_samples <= minutes_per_hour:
        raise ValueError("minimum_samples must be within an hour")
    shape = (
        len(minute_values) // minutes_per_hour,
        minutes_per_hour,
        minute_values.shape[1],
        minute_values.shape[2],
    )
    values = minute_values.reshape(shape)
    mask = minute_mask.reshape(shape)
    counts = mask.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(np.where(mask, values, np.nan), axis=1)
    hourly_mask = counts >= minimum_samples
    medians = np.where(hourly_mask, medians, 0.0).astype(np.float32)
    return medians, hourly_mask, counts.astype(np.int16)


def parse_omni_evaluation_indices(text: str) -> list[dict[str, Any]]:
    """Parse Kp and Dst for evaluation labels, never predictor features."""

    records = []
    for line in text.splitlines():
        words = line.split()
        if len(words) < 55:
            continue
        try:
            year, day_of_year, hour = map(int, words[:3])
            timestamp = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
                days=day_of_year - 1, hours=hour
            )
            kp_raw = float(words[38])
            dst_raw = float(words[40])
        except (TypeError, ValueError, OverflowError):
            continue
        records.append(
            {
                "time": timestamp,
                "kp": None if kp_raw == 99 else kp_raw / 10.0,
                "dst": None if dst_raw == 99999 else dst_raw,
            }
        )
    return records


def align_evaluation_indices(
    hours: list[datetime], records: Iterable[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kp = np.zeros(len(hours), dtype=np.float32)
    dst = np.zeros(len(hours), dtype=np.float32)
    mask = np.zeros((len(hours), 2), dtype=bool)
    hour_to_index = {hour: index for index, hour in enumerate(hours)}
    for record in records:
        index = hour_to_index.get(record.get("time"))
        if index is None:
            continue
        for column, (name, target) in enumerate((("kp", kp), ("dst", dst))):
            value = record.get(name)
            if value is not None and np.isfinite(float(value)):
                target[index] = float(value)
                mask[index, column] = True
    return kp, dst, mask


def calendar_quarter_splits(
    hours: list[datetime], gap_hours: int = 24
) -> dict[str, tuple[int, int]]:
    """Use H1 for training, Q3 validation, and untouched Q4 testing."""

    if not hours or hours[0].month != 1 or hours[0].day != 1:
        raise ValueError("calendar split requires a dataset starting January 1")
    year = hours[0].year
    lookup = {timestamp: index for index, timestamp in enumerate(hours)}
    july_1 = lookup[datetime(year, 7, 1, tzinfo=timezone.utc)]
    october_1 = lookup[datetime(year, 10, 1, tzinfo=timezone.utc)]
    if july_1 + gap_hours >= october_1 or october_1 + gap_hours >= len(hours):
        raise ValueError("calendar blocks are too short for the requested gaps")
    return {
        "train": (0, july_1),
        "validation": (july_1 + gap_hours, october_1),
        "test": (october_1 + gap_hours, len(hours)),
    }


def common_ready_mask(driver_mask: np.ndarray, warmup_hours: int) -> np.ndarray:
    """Require identical post-gap causal history for every temporal control."""

    if warmup_hours <= 0:
        raise ValueError("warmup_hours must be positive")
    ready = np.zeros(len(driver_mask), dtype=bool)
    run = 0
    for index, complete in enumerate(driver_mask.all(axis=1)):
        run = run + 1 if complete else 0
        ready[index] = run >= warmup_hours
    return ready


def exponential_state_features(
    drivers: np.ndarray, driver_mask: np.ndarray, half_life_hours: float
) -> np.ndarray:
    decay = float(np.exp(-np.log(2.0) / half_life_hours))
    state = np.zeros(drivers.shape[1], dtype=np.float64)
    features = np.zeros_like(drivers, dtype=np.float64)
    for index in range(len(drivers)):
        if driver_mask[index].all():
            state = decay * state + (1.0 - decay) * drivers[index]
        else:
            state.fill(0.0)
        features[index] = state
    return features


def orthogonal_recurrent_features(
    drivers: np.ndarray, driver_mask: np.ndarray, half_life_hours: float
) -> np.ndarray:
    """Stable signed-cycle recurrence with no learned transition parameters."""

    decay = float(np.exp(-np.log(2.0) / half_life_hours))
    state = np.zeros(drivers.shape[1], dtype=np.float64)
    features = np.zeros_like(drivers, dtype=np.float64)
    for index in range(len(drivers)):
        if driver_mask[index].all():
            rotated = np.roll(state, 1)
            rotated[0] *= -1.0
            state = decay * rotated + (1.0 - decay) * drivers[index]
        else:
            state.fill(0.0)
        features[index] = state
    return features


def storm_masks(kp: np.ndarray, dst: np.ndarray, index_mask: np.ndarray) -> dict[str, np.ndarray]:
    known = index_mask.all(axis=1)
    storm = known & ((kp >= 5.0) | (dst <= -50.0))
    severe = known & ((kp >= 7.0) | (dst <= -100.0))
    return {"storm": storm, "severe": severe, "quiet": known & ~storm}


def count_episodes(mask: np.ndarray, maximum_gap_hours: int = 12) -> int:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return 0
    return 1 + int(np.sum(np.diff(indices) > maximum_gap_hours))


def _control_rows(
    features: np.ndarray,
    ready: np.ndarray,
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    target_indices: np.ndarray,
    lead_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lead_hours <= 0 or lead_hours >= len(coefficients):
        raise ValueError("lead_hours must be within the time axis")
    current_complete = coefficient_mask[:-lead_hours].all(axis=1)
    target_complete = coefficient_mask[lead_hours:, target_indices].all(axis=1)
    eligible = ready[:-lead_hours] & current_complete & target_complete
    times = np.flatnonzero(eligible) + lead_hours
    x = np.concatenate(
        (
            coefficients[:-lead_hours][eligible],
            features[:-lead_hours][eligible],
        ),
        axis=1,
    )
    y = coefficients[lead_hours:][eligible][:, target_indices]
    return x.astype(np.float64), y.astype(np.float64), times


def _split_rows(
    target_times: np.ndarray, splits: dict[str, tuple[int, int]]
) -> dict[str, np.ndarray]:
    return {
        name: (target_times >= start) & (target_times < end)
        for name, (start, end) in splits.items()
    }


def select_ridge(
    features: np.ndarray,
    ready: np.ndarray,
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    target_indices: np.ndarray,
    lead_hours: int,
    ridge_grid: Iterable[float] = RIDGE_GRID,
) -> tuple[float | str, float]:
    """Select regularization using train-to-validation error only."""

    x, y, times = _control_rows(
        features,
        ready,
        coefficients,
        coefficient_mask,
        target_indices,
        lead_hours,
    )
    rows = _split_rows(times, splits)
    train = rows["train"]
    validation = rows["validation"]
    if train.sum() <= x.shape[1] or not validation.any():
        raise ValueError("insufficient common train/validation rows")
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0)
    std[std < 1e-8] = 1.0
    design = np.concatenate(((x - mean) / std, np.ones((len(x), 1))), axis=1)
    gram = design[train].T @ design[train]
    best = (float("inf"), 0.0)
    for ridge in ridge_grid:
        weights = np.linalg.solve(
            gram + float(ridge) * np.eye(gram.shape[0]),
            design[train].T @ y[train],
        )
        error = design[validation] @ weights - y[validation]
        validation_mse = float(np.mean(error**2))
        if validation_mse < best[0]:
            best = (validation_mse, float(ridge))
    null_validation_mse = float(np.mean(y[validation] ** 2))
    if null_validation_mse < best[0]:
        best = (null_validation_mse, NULL_RIDGE)
    return best[1], best[0]


def _error_breakdown(
    squared_error: np.ndarray,
    target_times: np.ndarray,
    target_indices: np.ndarray,
    labels: tuple[tuple[str, int, int], ...],
    evaluation_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mse": float(squared_error.mean()),
        "mse_by_sector": {},
        "mse_by_degree": {},
        "evaluation_strata": {},
    }
    target_labels = [labels[index] for index in target_indices]
    for sector in sorted({label[0] for label in target_labels}):
        columns = [index for index, label in enumerate(target_labels) if label[0] == sector]
        result["mse_by_sector"][sector] = float(squared_error[:, columns].mean())
    for degree in sorted({label[1] for label in target_labels}):
        columns = [index for index, label in enumerate(target_labels) if label[1] == degree]
        result["mse_by_degree"][str(degree)] = float(squared_error[:, columns].mean())
    for name, mask in evaluation_masks.items():
        selected = mask[target_times]
        result["evaluation_strata"][name] = {
            "rows": int(selected.sum()),
            "mse": float(squared_error[selected].mean()) if selected.any() else None,
            "episodes": count_episodes(mask & np.isin(np.arange(len(mask)), target_times)),
        }
    return result


def evaluate_control(
    features: np.ndarray,
    ready: np.ndarray,
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    target_indices: np.ndarray,
    labels: tuple[tuple[str, int, int], ...],
    evaluation_masks: dict[str, np.ndarray],
    ridge: float | str,
    lead_hours: int,
) -> dict[str, Any]:
    """Refit train+validation and evaluate the selected model once on test."""

    x, y, times = _control_rows(
        features,
        ready,
        coefficients,
        coefficient_mask,
        target_indices,
        lead_hours,
    )
    rows = _split_rows(times, splits)
    fit = rows["train"] | rows["validation"]
    test = rows["test"]
    mean = x[fit].mean(axis=0)
    std = x[fit].std(axis=0)
    std[std < 1e-8] = 1.0
    design = np.concatenate(((x - mean) / std, np.ones((len(x), 1))), axis=1)
    if ridge == NULL_RIDGE:
        weights = np.zeros((design.shape[1], y.shape[1]), dtype=np.float64)
    else:
        numeric_ridge = float(ridge)
        gram = design[fit].T @ design[fit]
        weights = np.linalg.solve(
            gram + numeric_ridge * np.eye(gram.shape[0]),
            design[fit].T @ y[fit],
        )
    squared_error = (design[test] @ weights - y[test]) ** 2
    result = _error_breakdown(
        squared_error, times[test], target_indices, labels, evaluation_masks
    )
    result.update(
        {
            "status": "ok",
            "selected_ridge": ridge,
            "train_rows": int(rows["train"].sum()),
            "validation_rows": int(rows["validation"].sum()),
            "fit_rows": int(fit.sum()),
            "test_rows": int(test.sum()),
            "features": int(x.shape[1]),
            "parameters": int((x.shape[1] + 1) * y.shape[1]),
            "lead_hours": lead_hours,
        }
    )
    result["_test_target_times"] = times[test]
    result["_test_row_mse"] = squared_error.mean(axis=1)
    return result


def paired_block_bootstrap(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    evaluation_masks: dict[str, np.ndarray],
    *,
    block_hours: int,
    block_label: str,
    samples: int = 5_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap paired fractional MSE improvement over contiguous blocks."""

    if block_hours <= 0:
        raise ValueError("block_hours must be positive")

    reference_times = reference["_test_target_times"]
    candidate_times = candidate["_test_target_times"]
    if not np.array_equal(reference_times, candidate_times):
        raise ValueError("paired controls must use identical test rows")
    reference_error = reference["_test_row_mse"]
    candidate_error = candidate["_test_row_mse"]
    strata = {"all": np.ones(len(reference_times), dtype=bool)}
    strata.update(
        {name: mask[reference_times] for name, mask in evaluation_masks.items()}
    )
    result = {}
    for stratum, selected in strata.items():
        times = reference_times[selected]
        if not len(times):
            result[stratum] = {"rows": 0, "days": 0, "estimate": None}
            continue
        blocks = times // block_hours
        unique_blocks, inverse = np.unique(blocks, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        reference_sums = np.bincount(inverse, weights=reference_error[selected])
        candidate_sums = np.bincount(inverse, weights=candidate_error[selected])
        estimate = 1.0 - candidate_error[selected].mean() / reference_error[selected].mean()
        if len(unique_blocks) == 1:
            result[stratum] = {
                "rows": int(len(times)),
                "blocks": 1,
                "estimate": float(estimate),
                "ci95": None,
                "ci98_33": None,
                "ci98_75": None,
                "ci99_44": None,
                "ci99_58": None,
                "probability_improvement": None,
            }
            continue
        generator = np.random.default_rng(seed)
        draws = generator.integers(
            0, len(unique_blocks), size=(samples, len(unique_blocks))
        )
        sampled_counts = counts[draws].sum(axis=1)
        reference_means = reference_sums[draws].sum(axis=1) / sampled_counts
        candidate_means = candidate_sums[draws].sum(axis=1) / sampled_counts
        improvement = 1.0 - candidate_means / reference_means
        result[stratum] = {
            "rows": int(len(times)),
            "blocks": int(len(unique_blocks)),
            "estimate": float(estimate),
            "ci95": [
                float(np.quantile(improvement, 0.025)),
                float(np.quantile(improvement, 0.975)),
            ],
            "ci98_33": [
                float(np.quantile(improvement, 1.0 / 120.0)),
                float(np.quantile(improvement, 119.0 / 120.0)),
            ],
            "ci98_75": [
                float(np.quantile(improvement, 1.0 / 160.0)),
                float(np.quantile(improvement, 159.0 / 160.0)),
            ],
            "ci99_44": [
                float(np.quantile(improvement, 1.0 / 360.0)),
                float(np.quantile(improvement, 359.0 / 360.0)),
            ],
            "ci99_58": [
                float(np.quantile(improvement, 1.0 / 480.0)),
                float(np.quantile(improvement, 479.0 / 480.0)),
            ],
            "probability_improvement": float(np.mean(improvement > 0.0)),
            "samples": samples,
            "block": block_label,
        }
    return result


def paired_daily_block_bootstrap(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    evaluation_masks: dict[str, np.ndarray],
    *,
    samples: int = 5_000,
    seed: int = 42,
) -> dict[str, Any]:
    return paired_block_bootstrap(
        reference,
        candidate,
        evaluation_masks,
        block_hours=24,
        block_label="UTC day",
        samples=samples,
        seed=seed,
    )


def paired_weekly_block_bootstrap(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    evaluation_masks: dict[str, np.ndarray],
    *,
    samples: int = 5_000,
    seed: int = 42,
) -> dict[str, Any]:
    return paired_block_bootstrap(
        reference,
        candidate,
        evaluation_masks,
        block_hours=24 * 7,
        block_label="UTC week",
        samples=samples,
        seed=seed,
    )


def _remove_private_diagnostics(value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key in list(value):
        if key.startswith("_"):
            value.pop(key)
        else:
            _remove_private_diagnostics(value[key])


def _select_temporal_control(
    candidates: dict[str, np.ndarray],
    ready: np.ndarray,
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    target_indices: np.ndarray,
    labels: tuple[tuple[str, int, int], ...],
    evaluation_masks: dict[str, np.ndarray],
    lead_hours: int,
) -> dict[str, Any]:
    selections = []
    for name, features in candidates.items():
        ridge, validation_mse = select_ridge(
            features,
            ready,
            coefficients,
            coefficient_mask,
            splits,
            target_indices,
            lead_hours,
        )
        selections.append((validation_mse, name, ridge))
    validation_mse, name, ridge = min(selections)
    result = evaluate_control(
        candidates[name],
        ready,
        coefficients,
        coefficient_mask,
        splits,
        target_indices,
        labels,
        evaluation_masks,
        ridge,
        lead_hours,
    )
    result["selected_candidate"] = "null" if ridge == NULL_RIDGE else name
    result["selection_validation_mse"] = validation_mse
    result["candidate_validation_mse"] = {
        candidate_name: mse for mse, candidate_name, _ in selections
    }
    return result


def persistence_and_climatology(
    ready: np.ndarray,
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    labels: tuple[tuple[str, int, int], ...],
    evaluation_masks: dict[str, np.ndarray],
    lead_hours: int,
) -> dict[str, Any]:
    targets = np.arange(coefficients.shape[1])
    x, y, times = _control_rows(
        np.empty((len(coefficients), 0)),
        ready,
        coefficients,
        coefficient_mask,
        targets,
        lead_hours,
    )
    rows = _split_rows(times, splits)
    test = rows["test"]
    train = rows["train"]
    current = x[:, : coefficients.shape[1]]
    climatology = y[train].mean(axis=0)
    return {
        "persistence": _error_breakdown(
            (current[test] - y[test]) ** 2,
            times[test],
            targets,
            labels,
            evaluation_masks,
        ),
        "training_climatology": _error_breakdown(
            (np.broadcast_to(climatology, y[test].shape) - y[test]) ** 2,
            times[test],
            targets,
            labels,
            evaluation_masks,
        ),
    }


def run_controls(
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    drivers: np.ndarray,
    driver_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    labels: tuple[tuple[str, int, int], ...],
    evaluation_masks: dict[str, np.ndarray],
    *,
    half_lives: Iterable[float] = DEFAULT_HALF_LIVES,
    warmup_hours: int = COMMON_WARMUP_HOURS,
    lead_hours: int = 1,
) -> dict[str, Any]:
    half_lives = tuple(float(value) for value in half_lives)
    ready = common_ready_mask(driver_mask, warmup_hours)
    all_targets = np.arange(coefficients.shape[1])
    markov_features = drivers.astype(np.float64)
    pole_candidates = {
        f"{half_life:g}h": exponential_state_features(
            drivers, driver_mask, half_life
        )
        for half_life in half_lives
    }
    recurrent_candidates = {
        f"{half_life:g}h": orthogonal_recurrent_features(
            drivers, driver_mask, half_life
        )
        for half_life in half_lives
    }
    markov_ridge, markov_validation = select_ridge(
        markov_features,
        ready,
        coefficients,
        coefficient_mask,
        splits,
        all_targets,
        lead_hours,
    )
    markov = evaluate_control(
        markov_features,
        ready,
        coefficients,
        coefficient_mask,
        splits,
        all_targets,
        labels,
        evaluation_masks,
        markov_ridge,
        lead_hours,
    )
    markov["selection_validation_mse"] = markov_validation
    pole = _select_temporal_control(
        pole_candidates,
        ready,
        coefficients,
        coefficient_mask,
        splits,
        all_targets,
        labels,
        evaluation_masks,
        lead_hours,
    )
    recurrent = _select_temporal_control(
        recurrent_candidates,
        ready,
        coefficients,
        coefficient_mask,
        splits,
        all_targets,
        labels,
        evaluation_masks,
        lead_hours,
    )

    sectors = sector_indices(labels)
    sector_models = {}
    for sector, indices in sectors.items():
        sector_markov_ridge, sector_markov_validation = select_ridge(
            markov_features,
            ready,
            coefficients,
            coefficient_mask,
            splits,
            indices,
            lead_hours,
        )
        sector_markov = evaluate_control(
            markov_features,
            ready,
            coefficients,
            coefficient_mask,
            splits,
            indices,
            labels,
            evaluation_masks,
            sector_markov_ridge,
            lead_hours,
        )
        sector_markov["selection_validation_mse"] = sector_markov_validation
        sector_pole = _select_temporal_control(
            pole_candidates,
            ready,
            coefficients,
            coefficient_mask,
            splits,
            indices,
            labels,
            evaluation_masks,
            lead_hours,
        )
        sector_recurrent = _select_temporal_control(
            recurrent_candidates,
            ready,
            coefficients,
            coefficient_mask,
            splits,
            indices,
            labels,
            evaluation_masks,
            lead_hours,
        )
        sector_models[sector] = {
            "markov": sector_markov,
            "single_pole": sector_pole,
            "orthogonal_recurrent": sector_recurrent,
            "single_pole_improvement_fraction": 1.0
            - sector_pole["mse"] / sector_markov["mse"],
            "orthogonal_recurrent_improvement_fraction": 1.0
            - sector_recurrent["mse"] / sector_markov["mse"],
        }

    models = {
        "markov": markov,
        "single_pole": pole,
        "orthogonal_recurrent": recurrent,
    }
    paired_uncertainty = {
        name: paired_daily_block_bootstrap(
            markov, model, evaluation_masks
        )
        for name, model in models.items()
        if name != "markov"
    }
    paired_weekly_uncertainty = {
        name: paired_weekly_block_bootstrap(
            markov, model, evaluation_masks
        )
        for name, model in models.items()
        if name != "markov"
    }
    for row in sector_models.values():
        row["paired_daily_bootstrap"] = {
            "single_pole": paired_daily_block_bootstrap(
                row["markov"], row["single_pole"], evaluation_masks
            ),
            "orthogonal_recurrent": paired_daily_block_bootstrap(
                row["markov"], row["orthogonal_recurrent"], evaluation_masks
            ),
        }
        row["paired_weekly_bootstrap"] = {
            "single_pole": paired_weekly_block_bootstrap(
                row["markov"], row["single_pole"], evaluation_masks
            ),
            "orthogonal_recurrent": paired_weekly_block_bootstrap(
                row["markov"], row["orthogonal_recurrent"], evaluation_masks
            ),
        }
    result = {
        "common_warmup_hours": warmup_hours,
        "lead_hours": lead_hours,
        "ridge_grid": RIDGE_GRID,
        "null_ridge_limit_available": True,
        "common_ready_fraction": float(ready.mean()),
        "half_lives_hours": half_lives,
        "models": models,
        "parameter_matched": len({model["parameters"] for model in models.values()}) == 1,
        "single_pole_improvement_fraction": 1.0 - pole["mse"] / markov["mse"],
        "orthogonal_recurrent_improvement_fraction": 1.0
        - recurrent["mse"] / markov["mse"],
        "paired_daily_bootstrap": paired_uncertainty,
        "paired_weekly_bootstrap": paired_weekly_uncertainty,
        "sector_models": sector_models,
        "baselines": persistence_and_climatology(
            ready,
            coefficients,
            coefficient_mask,
            splits,
            labels,
            evaluation_masks,
            lead_hours,
        ),
    }
    _remove_private_diagnostics(result)
    return result


def run_annual_experiment(
    source: Path,
    output: Path,
    omni_cache: Path,
    *,
    lmax: int = 2,
    minimum_samples: int = 30,
) -> dict[str, Any]:
    with np.load(source) as source_data:
        source_metadata = json.loads(source_data["metadata_json"].item())
        minute_timestamps = source_data["timestamps"].copy()
        station_values = source_data["station_values"].copy()
        station_mask = source_data["station_mask"].copy()
        latitudes = source_data["station_latitudes"].copy()
        longitudes = source_data["station_longitudes"].copy()
        station_codes = source_data["station_codes"].copy()

    hourly_values, hourly_mask, hourly_counts = hourly_median_network(
        station_values, station_mask, minimum_samples=minimum_samples
    )
    first = parse_utc(str(minute_timestamps[0]))
    hours = [first + timedelta(hours=index) for index in range(len(hourly_values))]
    hourly_deltas, hourly_delta_mask = temporal_differences(hourly_values, hourly_mask)
    spherical_values, spherical_mask = geomagnetic_xyz_to_spherical(
        hourly_deltas, hourly_delta_mask
    )
    coefficients, coefficient_mask, labels, conditions = (
        fit_vector_network_coefficients(
            spherical_values,
            spherical_mask,
            latitudes,
            longitudes,
            lmax=lmax,
        )
    )

    end = hours[-1] + timedelta(hours=1)
    omni_records, omni_provenance = load_omni2(first, end, omni_cache)
    drivers, driver_mask = align_omni_drivers(hours, omni_records)
    omni_path = omni_cache / f"omni2_{first.year}.dat"
    kp, dst, evaluation_index_mask = align_evaluation_indices(
        hours, parse_omni_evaluation_indices(omni_path.read_text())
    )
    evaluation_masks = storm_masks(kp, dst, evaluation_index_mask)
    splits = calendar_quarter_splits(hours)
    controls = run_controls(
        coefficients,
        coefficient_mask,
        drivers,
        driver_mask,
        splits,
        labels,
        evaluation_masks,
    )

    metadata = {
        "schema_version": 1,
        "dataset_kind": "annual_intermagnet_omni_vector_spherical_controls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(source),
        "source_created_at": source_metadata["created_at"],
        "intermagnet_source": source_metadata["source"],
        "intermagnet_citation": source_metadata["citation"],
        "omni_documentation": OMNI2_DOCUMENTATION_URL,
        "omni_provenance": omni_provenance,
        "start": hours[0].isoformat(),
        "end_exclusive": end.isoformat(),
        "station_codes": station_codes.tolist(),
        "driver_features": OMNI_DRIVER_FEATURES,
        "evaluation_only_indices": {
            "names": ["Kp", "Dst"],
            "storm": "Kp >= 5 or Dst <= -50 nT",
            "severe": "Kp >= 7 or Dst <= -100 nT",
            "warning": "Never used as features, targets, split boundaries, or selectors.",
        },
        "source_cadence_seconds": 60,
        "target_cadence_seconds": 3600,
        "hourly_minimum_samples": minimum_samples,
        "target": "next-hour change in VSH coefficients",
        "lmax": lmax,
        "labels": labels,
        "coordinate_convention": {
            "source": ["X north", "Y east", "Z down"],
            "vsh": ["radial outward = -Z", "theta south = -X", "phi east = Y"],
        },
        "splits": splits,
        "split_timestamps": {
            name: [hours[start].isoformat(), hours[end_index - 1].isoformat()]
            for name, (start, end_index) in splits.items()
        },
        "coverage": {
            "hourly_station_component_fraction": float(hourly_mask.mean()),
            "hourly_delta_component_fraction": float(hourly_delta_mask.mean()),
            "coefficient_fraction": float(coefficient_mask.mean()),
            "driver_component_fraction": float(driver_mask.mean()),
            "complete_driver_fraction": float(driver_mask.all(axis=1).mean()),
        },
        "design_condition_numbers": conditions,
        "storm_hours": {
            name: {
                stratum: int(mask[start:end_index].sum())
                for stratum, mask in evaluation_masks.items()
            }
            for name, (start, end_index) in splits.items()
        },
        "controls": controls,
        "warning": (
            "Exploratory linear diagnostic, not an operational geomagnetic or "
            "hazard forecast. best-avail INTERMAGNET data can mix publication quality."
        ),
    }
    dataset = {
        "timestamps": np.asarray([hour.isoformat() for hour in hours]),
        "station_codes": station_codes,
        "station_latitudes": latitudes,
        "station_longitudes": longitudes,
        "station_hourly_values": hourly_values,
        "station_hourly_mask": hourly_mask,
        "station_hourly_counts": hourly_counts,
        "station_hourly_deltas": hourly_deltas,
        "station_hourly_delta_mask": hourly_delta_mask,
        "spherical_station_deltas": spherical_values,
        "spherical_station_mask": spherical_mask,
        "coefficients": coefficients,
        "coefficient_mask": coefficient_mask,
        "drivers": drivers,
        "driver_mask": driver_mask,
        "kp_evaluation_only": kp,
        "dst_evaluation_only": dst,
        "evaluation_index_mask": evaluation_index_mask,
        "metadata": metadata,
    }
    save_dataset(dataset, output)
    return metadata


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent.parent / "data" / "operator"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "geomagnetic_global_2024_minute.npz")
    parser.add_argument("--output", type=Path, default=root / "geomagnetic_vector_global_2024_hourly.npz")
    parser.add_argument("--omni-cache", type=Path, default=root / "cache" / "omni")
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--minimum-samples", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_annual_experiment(
        args.source,
        args.output,
        args.omni_cache,
        lmax=args.lmax,
        minimum_samples=args.minimum_samples,
    )
    print(json.dumps(metadata, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
