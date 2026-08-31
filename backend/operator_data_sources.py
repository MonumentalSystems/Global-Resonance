"""Shared parsers for operator-training and live space-weather feeds."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


RTSW_MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
RTSW_WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
GOES_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
USGS_GEOMAG_URL = "https://geomag.usgs.gov/ws/data/"


def parse_utc(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _active_quality_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the active, quality-zero spacecraft record at each timestamp."""

    selected: dict[datetime, dict[str, Any]] = {}
    for row in rows:
        if not row.get("active", False):
            continue
        if row.get("overall_quality") != 0:
            continue
        try:
            timestamp = parse_utc(row["time_tag"])
        except (KeyError, TypeError, ValueError):
            continue
        previous = selected.get(timestamp)
        if previous is None or int(row.get("sample_size") or 0) > int(
            previous.get("sample_size") or 0
        ):
            selected[timestamp] = row
    return [selected[key] for key in sorted(selected)]


def parse_rtsw_magnetic(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in _active_quality_rows(rows):
        values = {key: row.get(key) for key in ("bt", "by_gsm", "bz_gsm")}
        if all(value is None for value in values.values()):
            continue
        records.append(
            {
                "time": parse_utc(row["time_tag"]),
                **values,
                "source": row.get("source"),
                "quality": row.get("overall_quality"),
            }
        )
    return records


def parse_rtsw_wind(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in _active_quality_rows(rows):
        speed = row.get("proton_speed")
        density = row.get("proton_density")
        if speed is None and density is None:
            continue
        records.append(
            {
                "time": parse_utc(row["time_tag"]),
                "speed": speed,
                "density": density,
                "temperature": row.get("proton_temperature"),
                "source": row.get("source"),
                "quality": row.get("overall_quality"),
            }
        )
    return records


def parse_goes_xray(
    rows: Iterable[dict[str, Any]], energy: str = "0.1-0.8nm"
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        if row.get("energy") != energy:
            continue
        try:
            flux = float(row["flux"])
            timestamp = parse_utc(row["time_tag"])
        except (KeyError, TypeError, ValueError):
            continue
        if flux <= 0:
            continue
        records.append(
            {
                "time": timestamp,
                "flux": flux,
                "satellite": row.get("satellite"),
                "electron_contamination": bool(
                    row.get("electron_contaminaton", False)
                ),
            }
        )
    return sorted(records, key=lambda row: row["time"])


def parse_usgs_geomag(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse USGS parallel time/component arrays without filling missing data."""

    metadata = payload.get("metadata", {}).get("intermagnet", {})
    imo = metadata.get("imo", {})
    coordinates = imo.get("coordinates") or [None, None, None]
    times = []
    for value in payload.get("times", []):
        try:
            times.append(parse_utc(value))
        except (TypeError, ValueError):
            times.append(None)

    components = {}
    for entry in payload.get("values", []):
        element = entry.get("id")
        if element in {"X", "Y", "Z"}:
            values = list(entry.get("values", []))
            if len(values) < len(times):
                values.extend([None] * (len(times) - len(values)))
            components[element] = values[: len(times)]

    records = []
    for index, timestamp in enumerate(times):
        if timestamp is None:
            continue
        records.append(
            {
                "time": timestamp,
                **{
                    element: components.get(element, [None] * len(times))[index]
                    for element in ("X", "Y", "Z")
                },
            }
        )
    return {
        "code": imo.get("iaga_code"),
        "name": imo.get("name"),
        "longitude": coordinates[0],
        "latitude": coordinates[1],
        "elevation_m": coordinates[2],
        "data_type": metadata.get("data_type"),
        "sampling_period": metadata.get("sampling_period"),
        "records": records,
    }
