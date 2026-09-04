#!/usr/bin/env python3
"""Download a globally conditioned INTERMAGNET vector-field network.

The INTERMAGNET HAPI service exposes minute XYZF observations and station
coordinates without requiring an account.  This module keeps the raw CSV
chunks, records their hashes, preserves fill values as masks, and emits a
compact NPZ suitable for the vector-spherical-harmonic experiments.

INTERMAGNET data are subject to the conditions linked in ``CITATION_URL``.
The default license is CC BY-NC 4.0, with institute-specific exceptions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import torch

try:
    from .vector_spherical_harmonics import real_vector_spherical_harmonic_basis
except ImportError:
    from vector_spherical_harmonics import real_vector_spherical_harmonic_basis


HAPI_BASE = "https://imag-data.bgs.ac.uk/GIN_V1/hapi"
CITATION_URL = "https://intermagnet.org/data_conditions.html"
COMPONENTS = ("X", "Y", "Z")

# Greedy group-D-optimal selection from stations with >=95% complete XYZ
# vectors over the 8--14 May 2024 geomagnetic-storm audit window.  At lmax=2,
# its 25-column VSH design has condition number 1.459 (USGS pilot: 674.7).
DEFAULT_GLOBAL_STATIONS = (
    "TTB",  # Brazil
    "CKI",  # Cocos-Keeling Islands
    "TSU",  # Namibia
    "TUC",  # United States
    "NUR",  # Finland
    "KAK",  # Japan
    "AIA",  # Antarctica / Argentine Islands
    "PPT",  # Tahiti
    "CNB",  # Australia
    "JAI",  # India
    "MAW",  # Antarctica
    "SHU",  # Alaska / Aleutians
    "IPM",  # Easter Island
    "GUI",  # Canary Islands
    "GUA",  # Guam
    "HBK",  # South Africa
)


def _request(url: str, params: dict[str, str], timeout: int = 120) -> bytes:
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"User-Agent": "GlobalResonanceOperator/0.2"},
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def dataset_id(
    code: str,
    data_type: str = "best-avail",
    cadence: str = "PT1M",
    orientation: str = "xyzf",
) -> str:
    return f"{code.lower()}/{data_type}/{cadence}/{orientation}"


def fetch_station_info(
    code: str,
    data_type: str = "best-avail",
) -> dict[str, Any]:
    identifier = dataset_id(code, data_type=data_type)
    payload = _request(f"{HAPI_BASE}/info", {"id": identifier})
    info = json.loads(payload)
    return {
        "code": code.upper(),
        "dataset_id": identifier,
        "start": info["startDate"],
        "stop": info["stopDate"],
        "latitude": float(info["x_latitude"]),
        "longitude": float(info["x_longitude"]),
        "elevation_m": float(info["x_elevation"]),
        "description": info.get("description"),
        "warnings": info.get("x_warnings", []),
    }


def parse_hapi_vector_csv(
    payload: bytes,
    fill_threshold: float = 90_000.0,
) -> list[dict[str, Any]]:
    """Parse headerless HAPI CSV and turn numeric fill values into nulls."""

    records = []
    for row in csv.reader(io.StringIO(payload.decode("utf-8"))):
        if len(row) < 4:
            continue
        timestamp = datetime.strptime(row[0], "%Y-%m-%dT%H:%MZ").replace(
            tzinfo=timezone.utc
        )
        record: dict[str, Any] = {"time": timestamp}
        for component, text in zip(COMPONENTS, row[1:4]):
            try:
                value = float(text)
            except ValueError:
                value = math.nan
            record[component] = (
                value
                if math.isfinite(value) and abs(value) < fill_threshold
                else None
            )
        records.append(record)
    return records


def fetch_station(
    code: str,
    start: datetime,
    end: datetime,
    *,
    data_type: str = "best-avail",
    chunk_days: int = 31,
    cache_dir: Path | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    """Fetch one station in resumable, hash-audited HAPI chunks."""

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if end <= start:
        raise ValueError("end must be after start")
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    if retries < 0:
        raise ValueError("retries must be non-negative")

    code = code.upper()
    info = fetch_station_info(code, data_type=data_type)
    records_by_time: dict[datetime, dict[str, Any]] = {}
    provenance = []
    cursor = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    while cursor < end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days))
        params = {
            "id": info["dataset_id"],
            "time.min": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time.max": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parameters": "Field_Vector",
            "format": "csv",
        }
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / (
                f"{code}_{cursor:%Y%m%dT%H%M%S}_{chunk_end:%Y%m%dT%H%M%S}_"
                f"{data_type}_PT1M_xyz.csv"
            )
        payload = cache_path.read_bytes() if cache_path and cache_path.exists() else None
        if payload is None:
            for attempt in range(retries + 1):
                try:
                    payload = _request(f"{HAPI_BASE}/data", params)
                    break
                except Exception:
                    if attempt == retries:
                        raise
                    time.sleep(2**attempt)
            assert payload is not None
            if cache_path is not None:
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_bytes(payload)
                temporary.replace(cache_path)

        for record in parse_hapi_vector_csv(payload):
            records_by_time[record["time"]] = record
        provenance.append(
            {
                "start": params["time.min"],
                "end": params["time.max"],
                "cache_file": str(cache_path) if cache_path else None,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
        cursor = chunk_end

    return {
        **info,
        "records": [records_by_time[key] for key in sorted(records_by_time)],
        "provenance": provenance,
    }


def minute_axis(start: datetime, end: datetime) -> list[datetime]:
    start = start.astimezone(timezone.utc).replace(second=0, microsecond=0)
    end = end.astimezone(timezone.utc).replace(second=0, microsecond=0)
    if end <= start:
        raise ValueError("end must be after start")
    count = int((end - start).total_seconds() // 60)
    return [start + timedelta(minutes=index) for index in range(count)]


def align_network(
    timestamps: list[datetime],
    stations: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(timestamps), len(stations), 3), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    time_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    for station_index, station in enumerate(stations):
        for record in station.get("records", []):
            time_index = time_to_index.get(record.get("time"))
            if time_index is None:
                continue
            for component_index, component in enumerate(COMPONENTS):
                value = record.get(component)
                if value is not None and math.isfinite(float(value)):
                    values[time_index, station_index, component_index] = float(value)
                    mask[time_index, station_index, component_index] = True
    return values, mask


def vsh_condition_summary(
    mask: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lmax: int = 2,
) -> dict[str, float | int]:
    """Summarize VSH conditioning efficiently over unique outage patterns."""

    basis, labels = real_vector_spherical_harmonic_basis(
        torch.as_tensor(latitudes, dtype=torch.float64),
        torch.as_tensor(longitudes, dtype=torch.float64),
        lmax,
    )
    design = basis.numpy().reshape(-1, len(labels))
    full_condition = (
        float(np.linalg.cond(design))
        if np.linalg.matrix_rank(design) == len(labels)
        else float("inf")
    )
    # Data XYZ maps to spherical (radial, theta, phi) as (Z, X, Y).
    spherical_mask = np.stack((mask[..., 2], mask[..., 0], mask[..., 1]), axis=-1)
    patterns, counts = np.unique(
        spherical_mask.reshape(len(mask), -1), axis=0, return_counts=True
    )
    weighted_conditions = []
    full_rank_steps = 0
    for pattern, count in zip(patterns, counts):
        observed = design[pattern]
        if (
            observed.shape[0] < len(labels)
            or np.linalg.matrix_rank(observed) != len(labels)
        ):
            continue
        condition = float(np.linalg.cond(observed))
        weighted_conditions.extend([condition] * int(count))
        full_rank_steps += int(count)
    finite = np.asarray(weighted_conditions)
    return {
        "n_coefficients": len(labels),
        "full_network": full_condition,
        "full_rank_fraction": full_rank_steps / len(mask),
        "median_observed": float(np.median(finite)) if len(finite) else float("inf"),
        "p95_observed": float(np.quantile(finite, 0.95)) if len(finite) else float("inf"),
        "max_observed": float(np.max(finite)) if len(finite) else float("inf"),
    }


def build_network_dataset(
    start: datetime,
    end: datetime,
    station_codes: Iterable[str] = DEFAULT_GLOBAL_STATIONS,
    *,
    data_type: str = "best-avail",
    chunk_days: int = 31,
    cache_dir: Path | None = None,
    workers: int = 6,
    lmax: int = 2,
) -> dict[str, Any]:
    timestamps = minute_axis(start, end)
    station_codes = tuple(code.upper() for code in station_codes)
    stations_by_code = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(station_codes))) as executor:
        futures = {
            executor.submit(
                fetch_station,
                code,
                start,
                end,
                data_type=data_type,
                chunk_days=chunk_days,
                cache_dir=cache_dir,
            ): code
            for code in station_codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                stations_by_code[code] = future.result()
            except Exception as exc:
                errors[code] = f"{type(exc).__name__}: {exc}"
                stations_by_code[code] = {"code": code, "records": []}

    stations = [stations_by_code[code] for code in station_codes]
    values, mask = align_network(timestamps, stations)
    latitudes = np.asarray(
        [station.get("latitude", np.nan) for station in stations], dtype=np.float64
    )
    longitudes = np.asarray(
        [station.get("longitude", np.nan) for station in stations], dtype=np.float64
    )
    finite_locations = np.isfinite(latitudes) & np.isfinite(longitudes)
    conditions = (
        vsh_condition_summary(mask, latitudes, longitudes, lmax=lmax)
        if finite_locations.all()
        else None
    )
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": HAPI_BASE,
        "citation": CITATION_URL,
        "data_type": data_type,
        "cadence": "PT1M",
        "orientation": "XYZF",
        "components": COMPONENTS,
        "start": timestamps[0].isoformat(),
        "end_exclusive": end.astimezone(timezone.utc).isoformat(),
        "stations": [
            {
                key: station.get(key)
                for key in (
                    "code",
                    "dataset_id",
                    "latitude",
                    "longitude",
                    "elevation_m",
                    "start",
                    "stop",
                    "description",
                    "warnings",
                    "provenance",
                )
            }
            for station in stations
        ],
        "fetch_errors": errors,
        "coverage": {
            "component_fraction": float(mask.mean()),
            "complete_vector_fraction": float(mask.all(axis=-1).mean()),
            "per_station_complete_vector_fraction": {
                code: float(mask[:, index].all(axis=-1).mean())
                for index, code in enumerate(station_codes)
            },
        },
        "vsh_lmax": lmax,
        "vsh_condition_numbers": conditions,
        "warning": (
            "Research data only. best-avail can mix definitive, quasi-definitive, "
            "and reported observations; consult INTERMAGNET conditions and each "
            "institute's license before redistribution or commercial use."
        ),
    }
    return {
        "timestamps": np.asarray([timestamp.isoformat() for timestamp in timestamps]),
        "station_codes": np.asarray(station_codes),
        "station_latitudes": latitudes,
        "station_longitudes": longitudes,
        "station_values": values,
        "station_mask": mask,
        "metadata": metadata,
    }


def save_dataset(dataset: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{key: value for key, value in dataset.items() if key != "metadata"},
        metadata_json=np.asarray(json.dumps(dataset["metadata"])),
    )


def parse_utc(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01T00:00:00Z")
    parser.add_argument("--end", default="2025-01-01T00:00:00Z")
    parser.add_argument("--stations", default=",".join(DEFAULT_GLOBAL_STATIONS))
    parser.add_argument("--data-type", default="best-avail")
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).parent.parent
        / "data"
        / "operator"
        / "cache"
        / "intermagnet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent
        / "data"
        / "operator"
        / "geomagnetic_global_2024_minute.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_network_dataset(
        parse_utc(args.start),
        parse_utc(args.end),
        (code.strip() for code in args.stations.split(",") if code.strip()),
        data_type=args.data_type,
        chunk_days=args.chunk_days,
        cache_dir=args.cache_dir,
        workers=args.workers,
        lmax=args.lmax,
    )
    save_dataset(dataset, args.output)
    print(json.dumps(dataset["metadata"], indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
