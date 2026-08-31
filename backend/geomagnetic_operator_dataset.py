#!/usr/bin/env python3
"""Build the first real-data slice for a spherical geomagnetic operator.

The builder aligns active-spacecraft NOAA RTSW and GOES drivers with USGS
ground magnetometers on a strict hourly UTC grid. Missing values remain missing
and receive explicit masks. Station deltas are projected to low-degree real
spherical harmonics by masked ridge least squares; they are network descriptors,
not an unbiased reconstruction of the unobserved globe.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import torch

try:
    from .operator_data_sources import (
        GOES_XRAY_URL,
        RTSW_MAG_URL,
        RTSW_WIND_URL,
        USGS_GEOMAG_URL,
        parse_goes_xray,
        parse_rtsw_magnetic,
        parse_rtsw_wind,
        parse_usgs_geomag,
        parse_utc,
    )
    from .spherical_operator_experiment import real_spherical_harmonic_basis
except ImportError:
    from operator_data_sources import (
        GOES_XRAY_URL,
        RTSW_MAG_URL,
        RTSW_WIND_URL,
        USGS_GEOMAG_URL,
        parse_goes_xray,
        parse_rtsw_magnetic,
        parse_rtsw_wind,
        parse_usgs_geomag,
        parse_utc,
    )
    from spherical_operator_experiment import real_spherical_harmonic_basis


DEFAULT_STATIONS = (
    "BOU",
    "BRW",
    "CMO",
    "DED",
    "FRD",
    "FRN",
    "GUA",
    "HON",
    "NEW",
    "SHU",
    "SIT",
    "SJG",
    "TUC",
)
DRIVER_FEATURES = (
    "bt",
    "by_gsm",
    "bz_gsm",
    "speed",
    "density",
    "log10_xray_flux",
)
COMPONENTS = ("X", "Y", "Z")


def decode_json_payload(payload: bytes) -> Any:
    """Decode a JSON response while accepting SWPC's trailing NUL padding."""

    return json.loads(payload.rstrip(b"\x00 \t\r\n"))


def fetch_json(url: str, params: dict[str, str] | None = None) -> Any:
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "GlobalResonanceOperator/0.1"})
    with urlopen(request, timeout=60) as response:
        # SWPC occasionally pads an otherwise valid response with NUL bytes.
        # Treat those as transport padding, not as a second JSON document.
        return decode_json_payload(response.read())


def floor_hour(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def hourly_reduce(
    records: Iterable[dict[str, Any]], fields: Iterable[str]
) -> dict[datetime, dict[str, float | None]]:
    fields = tuple(fields)
    buckets: dict[datetime, dict[str, list[float]]] = {}
    for record in records:
        timestamp = record.get("time")
        if not isinstance(timestamp, datetime):
            continue
        bucket = buckets.setdefault(floor_hour(timestamp), {field: [] for field in fields})
        for field in fields:
            value = record.get(field)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                bucket[field].append(numeric)
    result = {}
    for timestamp, values in buckets.items():
        result[timestamp] = {
            field: float(np.median(samples)) if samples else None
            for field, samples in values.items()
        }
    return result


def fetch_usgs_station(
    code: str, start: datetime, end: datetime, sampling_period: int = 60
) -> dict[str, Any]:
    """Fetch a station in chunks while preserving every null measurement."""

    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + timedelta(days=28))
        payload = fetch_json(
            USGS_GEOMAG_URL,
            {
                "id": code,
                "elements": ",".join(COMPONENTS),
                "sampling_period": str(sampling_period),
                "type": "variation",
                "format": "json",
                "starttime": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endtime": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        chunks.append(parse_usgs_geomag(payload))
        cursor = chunk_end

    if not chunks:
        return {"code": code, "records": []}
    result = {key: value for key, value in chunks[0].items() if key != "records"}
    deduplicated = {}
    for chunk in chunks:
        for record in chunk["records"]:
            deduplicated[record["time"]] = record
    result["records"] = [deduplicated[key] for key in sorted(deduplicated)]
    return result


def fetch_live_drivers() -> dict[str, dict[datetime, dict[str, float | None]]]:
    magnetic = parse_rtsw_magnetic(fetch_json(RTSW_MAG_URL))
    wind = parse_rtsw_wind(fetch_json(RTSW_WIND_URL))
    xray = [
        row
        for row in parse_goes_xray(fetch_json(GOES_XRAY_URL))
        if not row["electron_contamination"]
    ]
    return {
        "magnetic": hourly_reduce(magnetic, ("bt", "by_gsm", "bz_gsm")),
        "wind": hourly_reduce(wind, ("speed", "density")),
        "xray": hourly_reduce(xray, ("flux",)),
    }


def hourly_axis(start: datetime, end: datetime) -> list[datetime]:
    start = floor_hour(start)
    end = floor_hour(end)
    if end <= start:
        raise ValueError("end must be after start")
    result = []
    cursor = start
    while cursor < end:
        result.append(cursor)
        cursor += timedelta(hours=1)
    return result


def align_drivers(
    hours: list[datetime],
    sources: dict[str, dict[datetime, dict[str, float | None]]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(hours), len(DRIVER_FEATURES)), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    for index, timestamp in enumerate(hours):
        combined = {
            **sources.get("magnetic", {}).get(timestamp, {}),
            **sources.get("wind", {}).get(timestamp, {}),
        }
        flux = sources.get("xray", {}).get(timestamp, {}).get("flux")
        if flux is not None and flux > 0:
            combined["log10_xray_flux"] = math.log10(flux)
        for feature_index, feature in enumerate(DRIVER_FEATURES):
            value = combined.get(feature)
            if value is not None and math.isfinite(float(value)):
                values[index, feature_index] = float(value)
                mask[index, feature_index] = True
    return values, mask


def align_stations(
    hours: list[datetime], stations: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(hours), len(stations), len(COMPONENTS)), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    hour_to_index = {timestamp: index for index, timestamp in enumerate(hours)}
    for station_index, station in enumerate(stations):
        reduced = hourly_reduce(station.get("records", []), COMPONENTS)
        for timestamp, row in reduced.items():
            time_index = hour_to_index.get(timestamp)
            if time_index is None:
                continue
            for component_index, component in enumerate(COMPONENTS):
                value = row.get(component)
                if value is not None and math.isfinite(float(value)):
                    values[time_index, station_index, component_index] = float(value)
                    mask[time_index, station_index, component_index] = True
    return values, mask


def temporal_differences(
    values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    differences = np.zeros_like(values)
    difference_mask = np.zeros_like(mask)
    differences[1:] = values[1:] - values[:-1]
    difference_mask[1:] = mask[1:] & mask[:-1]
    differences[~difference_mask] = 0.0
    return differences, difference_mask


def fit_network_coefficients(
    station_values: np.ndarray,
    station_mask: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lmax: int = 2,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Fit low-degree coefficients independently at each time/component."""

    basis_t, modes = real_spherical_harmonic_basis(
        torch.as_tensor(latitudes, dtype=torch.float64),
        torch.as_tensor(longitudes, dtype=torch.float64),
        lmax,
    )
    basis = basis_t.detach().cpu().numpy()
    n_modes = basis.shape[1]
    coefficients = np.zeros(
        (station_values.shape[0], station_values.shape[2], n_modes),
        dtype=np.float32,
    )
    coefficient_mask = np.zeros_like(coefficients, dtype=bool)
    identity = np.eye(n_modes)

    finite_locations = np.isfinite(latitudes) & np.isfinite(longitudes)
    for time_index in range(station_values.shape[0]):
        for component_index in range(station_values.shape[2]):
            valid = station_mask[time_index, :, component_index] & finite_locations
            design = basis[valid]
            if valid.sum() < n_modes or np.linalg.matrix_rank(design) < n_modes:
                continue
            target = station_values[time_index, valid, component_index]
            gram = design.T @ design
            scale = np.trace(gram) / n_modes
            solution = np.linalg.solve(
                gram + ridge * max(scale, 1e-12) * identity,
                design.T @ target,
            )
            coefficients[time_index, component_index] = solution
            coefficient_mask[time_index, component_index] = True
    return coefficients, coefficient_mask, modes


def forward_chaining_splits(
    n_steps: int, train_fraction: float = 0.6, validation_fraction: float = 0.2, gap: int = 1
) -> dict[str, tuple[int, int]]:
    if n_steps < 12:
        raise ValueError("at least 12 hourly steps are required")
    train_end = max(2, int(n_steps * train_fraction))
    validation_start = train_end + gap
    validation_end = max(validation_start + 2, int(n_steps * (train_fraction + validation_fraction)))
    test_start = validation_end + gap
    if test_start >= n_steps - 1:
        raise ValueError("not enough steps after split gaps")
    return {
        "train": (0, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, n_steps),
    }


def _masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return float(np.mean((prediction[mask] - target[mask]) ** 2))


def baseline_metrics(
    coefficients: np.ndarray,
    coefficient_mask: np.ndarray,
    drivers: np.ndarray,
    driver_mask: np.ndarray,
    splits: dict[str, tuple[int, int]],
    ridge: float = 1e-3,
) -> dict[str, Any]:
    """Persistence, train-climatology, and ridge VAR controls."""

    train_start, train_end = splits["train"]
    test_start, test_end = splits["test"]
    target = coefficients[test_start:test_end]
    target_mask = coefficient_mask[test_start:test_end]
    previous = coefficients[test_start - 1 : test_end - 1]
    previous_mask = coefficient_mask[test_start - 1 : test_end - 1]
    persistence_mask = target_mask & previous_mask

    train_values = coefficients[train_start:train_end]
    train_mask = coefficient_mask[train_start:train_end]
    counts = train_mask.sum(axis=0)
    climatology = np.divide(
        (train_values * train_mask).sum(axis=0),
        np.maximum(counts, 1),
    )
    climatology_prediction = np.broadcast_to(climatology, target.shape)
    climatology_mask = target_mask & (counts > 0)[None]
    result: dict[str, Any] = {
        "persistence_mse": _masked_mse(previous, target, persistence_mask),
        "climatology_mse": _masked_mse(
            climatology_prediction, target, climatology_mask
        ),
    }

    flat_coefficients = coefficients.reshape(coefficients.shape[0], -1)
    flat_mask = coefficient_mask.reshape(coefficient_mask.shape[0], -1)
    x_rows, y_rows, row_times = [], [], []
    for time_index in range(1, coefficients.shape[0]):
        if not (
            flat_mask[time_index - 1].all()
            and flat_mask[time_index].all()
            and driver_mask[time_index - 1].all()
        ):
            continue
        x_rows.append(
            np.concatenate((flat_coefficients[time_index - 1], drivers[time_index - 1]))
        )
        y_rows.append(flat_coefficients[time_index])
        row_times.append(time_index)
    if not x_rows:
        result["ridge_var"] = {"status": "unavailable", "reason": "no complete rows"}
        return result

    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    row_times = np.asarray(row_times)
    train_rows = row_times < train_end
    test_rows = (row_times >= test_start) & (row_times < test_end)
    if train_rows.sum() <= x.shape[1] or test_rows.sum() == 0:
        result["ridge_var"] = {
            "status": "unavailable",
            "reason": "insufficient complete train/test rows",
            "train_rows": int(train_rows.sum()),
            "test_rows": int(test_rows.sum()),
            "features": int(x.shape[1]),
        }
        return result

    mean = x[train_rows].mean(axis=0)
    std = x[train_rows].std(axis=0)
    std[std < 1e-8] = 1.0
    x_scaled = (x - mean) / std
    x_design = np.concatenate((x_scaled, np.ones((len(x_scaled), 1))), axis=1)
    gram = x_design[train_rows].T @ x_design[train_rows]
    weights = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]),
        x_design[train_rows].T @ y[train_rows],
    )
    prediction = x_design[test_rows] @ weights
    result["ridge_var"] = {
        "status": "ok",
        "mse": float(np.mean((prediction - y[test_rows]) ** 2)),
        "train_rows": int(train_rows.sum()),
        "test_rows": int(test_rows.sum()),
    }
    return result


def build_dataset(
    start: datetime,
    end: datetime,
    station_codes: Iterable[str] = DEFAULT_STATIONS,
    lmax: int = 2,
) -> dict[str, Any]:
    hours = hourly_axis(start, end)
    driver_sources = fetch_live_drivers()
    drivers, driver_mask = align_drivers(hours, driver_sources)

    station_codes = tuple(station_codes)
    stations_by_code = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=min(6, len(station_codes))) as executor:
        futures = {
            executor.submit(fetch_usgs_station, code, start, end): code
            for code in station_codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                stations_by_code[code] = future.result()
            except Exception as exc:  # retain source failure as metadata, not zeros
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
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": hours[0].isoformat(),
        "end_exclusive": end.astimezone(timezone.utc).isoformat(),
        "driver_features": DRIVER_FEATURES,
        "components": COMPONENTS,
        "stations": [
            {
                key: station.get(key)
                for key in ("code", "name", "latitude", "longitude", "elevation_m")
            }
            for station in stations
        ],
        "station_fetch_errors": errors,
        "modes": modes,
        "lmax": lmax,
        "splits": splits,
        "controls": controls,
        "coverage": {
            "driver_fraction": float(driver_mask.mean()),
            "station_fraction": float(station_mask.mean()),
            "coefficient_fraction": float(coefficient_mask.mean()),
        },
        "warning": (
            "Research dataset only. Coefficients are masked network least-squares "
            "descriptors, not complete global-field observations."
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


def save_dataset(dataset: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in dataset.items() if key != "metadata"}
    np.savez_compressed(
        output,
        **arrays,
        metadata_json=np.asarray(json.dumps(dataset["metadata"])),
    )


def parse_args() -> argparse.Namespace:
    now = floor_hour(datetime.now(timezone.utc))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=(now - timedelta(hours=24)).isoformat())
    parser.add_argument("--end", default=now.isoformat())
    parser.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent
        / "data"
        / "operator"
        / "geomagnetic_live_hourly.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    station_codes = tuple(code.strip().upper() for code in args.stations.split(","))
    dataset = build_dataset(start, end, station_codes, lmax=args.lmax)
    save_dataset(dataset, args.output)
    print(json.dumps(dataset["metadata"], indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
