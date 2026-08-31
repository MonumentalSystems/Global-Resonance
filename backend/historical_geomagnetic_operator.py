#!/usr/bin/env python3
"""Build a resumable historical OMNI-to-geomagnetic spherical data slice.

NASA OMNI2 supplies upstream hourly solar-wind drivers. USGS one-minute X/Y/Z
variation measurements supply the response field at an irregular station
network and are reduced to hourly medians. All raw downloads are cached with
hashes, missing values remain masked, and diagnostics use chronological splits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.request import Request, urlopen

import numpy as np

try:
    from .geomagnetic_operator_dataset import (
        COMPONENTS,
        DEFAULT_STATIONS,
        align_stations,
        baseline_metrics,
        fit_network_coefficients,
        floor_hour,
        forward_chaining_splits,
        hourly_axis,
        fetch_usgs_station,
        save_dataset,
        temporal_differences,
    )
    from .operator_data_sources import USGS_GEOMAG_URL, parse_utc
except ImportError:
    from geomagnetic_operator_dataset import (
        COMPONENTS,
        DEFAULT_STATIONS,
        align_stations,
        baseline_metrics,
        fit_network_coefficients,
        floor_hour,
        forward_chaining_splits,
        hourly_axis,
        fetch_usgs_station,
        save_dataset,
        temporal_differences,
    )
    from operator_data_sources import USGS_GEOMAG_URL, parse_utc


OMNI2_URL = (
    "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
    "omni2_{year}.dat"
)
OMNI2_DOCUMENTATION_URL = (
    "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text"
)
OMNI_DRIVER_FEATURES = ("bt", "by_gsm", "bz_gsm", "speed", "density")
DEFAULT_HALF_LIVES = (1.0, 3.0, 6.0, 12.0, 24.0)
HISTORICAL_DEFAULT_STATIONS = tuple(
    code for code in DEFAULT_STATIONS if code not in {"DED", "SHU"}
)


def _clean_omni(value: str, fill_value: float) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or math.isclose(parsed, fill_value):
        return None
    return parsed


def parse_omni2(text: str) -> list[dict[str, Any]]:
    """Parse documented OMNI2 hourly words without response-derived indices."""

    records = []
    for line in text.splitlines():
        words = line.split()
        if len(words) < 55:
            continue
        try:
            year = int(words[0])
            day_of_year = int(words[1])
            hour = int(words[2])
            timestamp = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
                days=day_of_year - 1,
                hours=hour,
            )
        except (ValueError, OverflowError):
            continue
        records.append(
            {
                "time": timestamp,
                "bt": _clean_omni(words[8], 999.9),
                "by_gsm": _clean_omni(words[15], 999.9),
                "bz_gsm": _clean_omni(words[16], 999.9),
                "density": _clean_omni(words[23], 999.9),
                "speed": _clean_omni(words[24], 9999.0),
                "imf_spacecraft_id": int(words[4]),
                "plasma_spacecraft_id": int(words[5]),
            }
        )
    return records


def _download_bytes(url: str, retries: int = 2) -> bytes:
    request = Request(url, headers={"User-Agent": "GlobalResonanceOperator/0.2"})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def load_omni2(
    start: datetime,
    end: datetime,
    cache_dir: Path,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load annual OMNI2 files through an atomic, hash-audited disk cache."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    provenance = []
    for year in range(start.year, (end - timedelta(microseconds=1)).year + 1):
        url = OMNI2_URL.format(year=year)
        cache_path = cache_dir / f"omni2_{year}.dat"
        if refresh or not cache_path.exists():
            payload = _download_bytes(url)
            parsed = parse_omni2(payload.decode("ascii", errors="strict"))
            if not parsed:
                raise ValueError(f"NASA OMNI2 file for {year} contained no records")
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(cache_path)
        else:
            payload = cache_path.read_bytes()
            parsed = parse_omni2(payload.decode("ascii", errors="strict"))
            if not parsed:
                raise ValueError(f"cached OMNI2 file for {year} contained no records")
        records.extend(row for row in parsed if start <= row["time"] < end)
        provenance.append(
            {
                "year": year,
                "url": url,
                "cache_file": str(cache_path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    deduplicated = {row["time"]: row for row in records}
    return [deduplicated[key] for key in sorted(deduplicated)], provenance


def align_omni_drivers(
    hours: list[datetime], records: Iterable[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(hours), len(OMNI_DRIVER_FEATURES)), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    hour_to_index = {timestamp: index for index, timestamp in enumerate(hours)}
    for record in records:
        timestamp = record.get("time")
        if not isinstance(timestamp, datetime):
            continue
        index = hour_to_index.get(floor_hour(timestamp))
        if index is None:
            continue
        for feature_index, feature in enumerate(OMNI_DRIVER_FEATURES):
            value = record.get(feature)
            if value is not None and math.isfinite(float(value)):
                values[index, feature_index] = float(value)
                mask[index, feature_index] = True
    return values, mask


def causal_pole_features(
    drivers: np.ndarray,
    driver_mask: np.ndarray,
    half_lives: Iterable[float] = DEFAULT_HALF_LIVES,
    burn_in_multiple: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed exponential-cavity features without crossing source gaps."""

    half_lives = np.asarray(tuple(half_lives), dtype=np.float64)
    decays = np.exp(-math.log(2.0) / half_lives)
    state = np.zeros((len(decays), drivers.shape[1]), dtype=np.float64)
    features = np.zeros((len(drivers), state.size), dtype=np.float64)
    mask = np.zeros(len(drivers), dtype=bool)
    required_history = max(1, math.ceil(float(half_lives.max()) * burn_in_multiple))
    complete_run = 0
    for time_index in range(len(drivers)):
        if driver_mask[time_index].all():
            state = decays[:, None] * state + (1.0 - decays[:, None]) * drivers[
                time_index
            ]
            complete_run += 1
            mask[time_index] = complete_run >= required_history
        else:
            state.fill(0.0)
            complete_run = 0
        features[time_index] = state.reshape(-1)
    return features, mask


def _ridge_metric(
    features: np.ndarray,
    feature_mask: np.ndarray,
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    ridge_grid: Iterable[float] = (1e-4, 1e-2, 1.0, 100.0, 10_000.0),
) -> dict[str, Any]:
    flat_coefficients = coefficients.reshape(len(coefficients), -1)
    flat_mask = coefficient_mask.reshape(len(coefficient_mask), -1)
    x_rows, y_rows, target_times = [], [], []
    for time_index in range(len(coefficients) - 1):
        if not (
            feature_mask[time_index]
            and flat_mask[time_index].all()
            and flat_mask[time_index + 1].all()
        ):
            continue
        x_rows.append(np.concatenate((flat_coefficients[time_index], features[time_index])))
        y_rows.append(flat_coefficients[time_index + 1])
        target_times.append(time_index + 1)
    if not x_rows:
        return {"status": "unavailable", "reason": "no complete causal rows"}

    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    target_times = np.asarray(target_times)
    train_start, train_end = splits["train"]
    validation_start, validation_end = splits["validation"]
    test_start, test_end = splits["test"]
    train_rows = (target_times >= train_start) & (target_times < train_end)
    validation_rows = (target_times >= validation_start) & (
        target_times < validation_end
    )
    test_rows = (target_times >= test_start) & (target_times < test_end)
    if (
        train_rows.sum() <= x.shape[1]
        or not validation_rows.any()
        or not test_rows.any()
    ):
        return {
            "status": "unavailable",
            "reason": "insufficient complete train/validation/test rows",
            "train_rows": int(train_rows.sum()),
            "validation_rows": int(validation_rows.sum()),
            "test_rows": int(test_rows.sum()),
            "features": int(x.shape[1]),
        }

    train_mean = x[train_rows].mean(axis=0)
    train_std = x[train_rows].std(axis=0)
    train_std[train_std < 1e-8] = 1.0
    train_scaled = (x - train_mean) / train_std
    train_design = np.concatenate(
        (train_scaled, np.ones((len(train_scaled), 1))), axis=1
    )
    train_gram = train_design[train_rows].T @ train_design[train_rows]
    best_ridge = None
    best_validation_mse = float("inf")
    for ridge in ridge_grid:
        weights = np.linalg.solve(
            train_gram + float(ridge) * np.eye(train_gram.shape[0]),
            train_design[train_rows].T @ y[train_rows],
        )
        validation_prediction = train_design[validation_rows] @ weights
        validation_mse = float(
            np.mean((validation_prediction - y[validation_rows]) ** 2)
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_ridge = float(ridge)
    assert best_ridge is not None

    fit_rows = train_rows | validation_rows
    mean = x[fit_rows].mean(axis=0)
    std = x[fit_rows].std(axis=0)
    std[std < 1e-8] = 1.0
    x_scaled = (x - mean) / std
    design = np.concatenate((x_scaled, np.ones((len(x_scaled), 1))), axis=1)
    gram = design[fit_rows].T @ design[fit_rows]
    weights = np.linalg.solve(
        gram + best_ridge * np.eye(gram.shape[0]),
        design[fit_rows].T @ y[fit_rows],
    )
    prediction = design[test_rows] @ weights
    squared_error = (prediction - y[test_rows]) ** 2
    result = {
        "status": "ok",
        "mse": float(np.mean(squared_error)),
        "train_rows": int(train_rows.sum()),
        "validation_rows": int(validation_rows.sum()),
        "fit_rows": int(fit_rows.sum()),
        "test_rows": int(test_rows.sum()),
        "features": int(x.shape[1]),
        "parameters": int((x.shape[1] + 1) * y.shape[1]),
        "selected_ridge": best_ridge,
        "selection_validation_mse": best_validation_mse,
    }
    if coefficients.ndim == 3:
        shaped_error = squared_error.reshape(
            len(squared_error), coefficients.shape[1], coefficients.shape[2]
        )
        result["mse_by_component"] = {
            component: float(shaped_error[:, index].mean())
            for index, component in enumerate(COMPONENTS[: coefficients.shape[1]])
        }
        result["mse_by_mode_index"] = [
            float(shaped_error[:, :, index].mean())
            for index in range(coefficients.shape[2])
        ]
        if coefficients.shape[2] > 1:
            result["mse_nonconstant_modes"] = float(
                shaped_error[:, :, 1:].mean()
            )
    return result


def historical_ridge_controls(
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    drivers: np.ndarray,
    driver_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    half_lives: Iterable[float] = DEFAULT_HALF_LIVES,
) -> dict[str, Any]:
    """Compare current-driver and fixed-pole ridge diagnostics."""

    markov = _ridge_metric(
        drivers,
        driver_mask.all(axis=1),
        coefficients,
        coefficient_mask,
        splits,
    )
    pole_features, pole_mask = causal_pole_features(
        drivers, driver_mask, half_lives=half_lives
    )
    fixed_pole = _ridge_metric(
        pole_features,
        pole_mask,
        coefficients,
        coefficient_mask,
        splits,
    )
    single_pole_candidates = []
    for half_life in half_lives:
        single_features, single_mask = causal_pole_features(
            drivers, driver_mask, half_lives=(half_life,)
        )
        candidate = _ridge_metric(
            single_features,
            single_mask,
            coefficients,
            coefficient_mask,
            splits,
        )
        candidate["half_life_hours"] = float(half_life)
        single_pole_candidates.append(candidate)
    available_candidates = [
        candidate
        for candidate in single_pole_candidates
        if candidate.get("status") == "ok"
    ]
    parameter_matched = (
        min(
            available_candidates,
            key=lambda candidate: candidate["selection_validation_mse"],
        )
        if available_candidates
        else {"status": "unavailable", "reason": "no valid half-life candidate"}
    )
    result = {
        "markov_ridge": markov,
        "parameter_matched_single_pole_ridge": parameter_matched,
        "fixed_pole_ridge": fixed_pole,
        "half_lives_hours": list(half_lives),
        "comparison_warning": (
            "The selected single-pole ridge is parameter-matched to Markov. "
            "The multi-pole bank has more features and is diagnostic only."
        ),
    }
    if markov.get("status") == parameter_matched.get("status") == "ok":
        result["single_pole_improvement_fraction"] = 1.0 - (
            parameter_matched["mse"] / markov["mse"]
        )
        if "mse_nonconstant_modes" in markov:
            result["single_pole_nonconstant_improvement_fraction"] = 1.0 - (
                parameter_matched["mse_nonconstant_modes"]
                / markov["mse_nonconstant_modes"]
            )
    if markov.get("status") == fixed_pole.get("status") == "ok":
        result["fixed_pole_improvement_fraction"] = 1.0 - (
            fixed_pole["mse"] / markov["mse"]
        )
        if "mse_nonconstant_modes" in markov:
            result["fixed_pole_nonconstant_improvement_fraction"] = 1.0 - (
                fixed_pole["mse_nonconstant_modes"]
                / markov["mse_nonconstant_modes"]
            )
    return result


def build_historical_dataset(
    start: datetime,
    end: datetime,
    station_codes: Iterable[str] = HISTORICAL_DEFAULT_STATIONS,
    lmax: int = 2,
    data_type: str = "variation",
    chunk_days: int = 28,
    cache_dir: Path = Path("data/operator/cache"),
    refresh_omni: bool = False,
) -> dict[str, Any]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    hours = hourly_axis(start, end)
    omni_records, omni_provenance = load_omni2(
        start,
        end,
        cache_dir / "omni",
        refresh=refresh_omni,
    )
    drivers, driver_mask = align_omni_drivers(hours, omni_records)

    station_codes = tuple(station_codes)
    stations_by_code = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=min(6, len(station_codes))) as executor:
        futures = {
            executor.submit(
                fetch_usgs_station,
                code,
                start,
                end,
                60,
                data_type,
                chunk_days,
                cache_dir / "usgs" / data_type,
                2,
            ): code
            for code in station_codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                station = future.result()
                has_values = any(
                    record.get(component) is not None
                    for record in station.get("records", [])
                    for component in COMPONENTS
                )
                if not has_values:
                    errors[code] = "NoUsableData: source returned no X/Y/Z samples"
                    station["records"] = []
                stations_by_code[code] = station
            except Exception as exc:
                errors[code] = f"{type(exc).__name__}: {exc}"
                stations_by_code[code] = {"code": code, "records": []}

    stations = [stations_by_code[code] for code in station_codes]
    station_values, station_mask = align_stations(hours, stations)
    station_deltas, station_delta_mask = temporal_differences(
        station_values, station_mask
    )
    latitudes = np.asarray(
        [station.get("latitude", np.nan) for station in stations], dtype=np.float64
    )
    longitudes = np.asarray(
        [station.get("longitude", np.nan) for station in stations], dtype=np.float64
    )
    coefficients, coefficient_mask, modes = fit_network_coefficients(
        station_deltas,
        station_delta_mask,
        latitudes,
        longitudes,
        lmax=lmax,
    )
    splits = forward_chaining_splits(len(hours))
    controls = baseline_metrics(
        coefficients, coefficient_mask, drivers, driver_mask, splits
    )
    controls.update(
        historical_ridge_controls(
            coefficients, coefficient_mask, drivers, driver_mask, splits
        )
    )
    usgs_provenance = {
        station.get("code"): station.get("provenance", []) for station in stations
    }
    complete_rows = driver_mask.all(axis=1) & coefficient_mask.reshape(
        len(hours), -1
    ).all(axis=1)
    metadata = {
        "schema_version": 1,
        "dataset_kind": f"historical_omni2_usgs_{data_type}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": hours[0].isoformat(),
        "end_exclusive": end.astimezone(timezone.utc).isoformat(),
        "driver_features": OMNI_DRIVER_FEATURES,
        "excluded_response_indices": ["Dst", "AE", "Kp", "ap"],
        "components": COMPONENTS,
        "stations": [
            {
                key: station.get(key)
                for key in ("code", "name", "latitude", "longitude", "elevation_m")
            }
            for station in stations
        ],
        "default_station_exclusions": {
            "DED": "2024 January source response contained no X/Y/Z samples",
            "SHU": "2024 January source request returned HTTP 404",
        },
        "station_fetch_errors": errors,
        "data_type": data_type,
        "source_sampling_period_seconds": 60,
        "target_sampling_period_seconds": 3600,
        "modes": modes,
        "lmax": lmax,
        "splits": splits,
        "controls": controls,
        "coverage": {
            "driver_fraction": float(driver_mask.mean()),
            "station_fraction": float(station_mask.mean()),
            "coefficient_fraction": float(coefficient_mask.mean()),
            "complete_hour_fraction": float(complete_rows.mean()),
        },
        "provenance": {
            "omni2_documentation": OMNI2_DOCUMENTATION_URL,
            "omni2_files": omni_provenance,
            "usgs_endpoint": USGS_GEOMAG_URL,
            "usgs_cache_chunks": usgs_provenance,
        },
        "warning": (
            "Research dataset only. Spherical coefficients are irregular-network "
            "least-squares descriptors, not complete global-field observations."
        ),
    }
    return {
        "timestamps": np.asarray([value.isoformat() for value in hours]),
        "drivers": drivers,
        "driver_mask": driver_mask,
        "station_values": station_values,
        "station_mask": station_mask,
        "station_deltas": station_deltas,
        "station_delta_mask": station_delta_mask,
        "station_latitudes": latitudes,
        "station_longitudes": longitudes,
        "coefficients": coefficients,
        "coefficient_mask": coefficient_mask,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01T00:00:00+00:00")
    parser.add_argument("--end", default="2024-01-29T00:00:00+00:00")
    parser.add_argument("--stations", default=",".join(HISTORICAL_DEFAULT_STATIONS))
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument(
        "--data-type",
        choices=("variation", "adjusted", "quasi-definitive", "definitive"),
        default="variation",
    )
    parser.add_argument("--chunk-days", type=int, default=28)
    parser.add_argument("--refresh-omni", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "operator" / "cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent
        / "data"
        / "operator"
        / "geomagnetic_historical_2024_january.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    station_codes = tuple(code.strip().upper() for code in args.stations.split(","))
    dataset = build_historical_dataset(
        start,
        end,
        station_codes,
        lmax=args.lmax,
        data_type=args.data_type,
        chunk_days=args.chunk_days,
        cache_dir=args.cache_dir,
        refresh_omni=args.refresh_omni,
    )
    save_dataset(dataset, args.output)
    print(json.dumps(dataset["metadata"], indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
