from datetime import datetime, timezone
import json

import numpy as np

import intermagnet_hapi as hapi


def test_parse_hapi_vector_csv_preserves_fill_mask():
    payload = (
        b"2024-05-10T00:00Z,1.0,-2.0,3.0\n"
        b"2024-05-10T00:01Z,99999.0,4.0,5.0\n"
    )

    records = hapi.parse_hapi_vector_csv(payload)

    assert records[0] == {
        "time": datetime(2024, 5, 10, tzinfo=timezone.utc),
        "X": 1.0,
        "Y": -2.0,
        "Z": 3.0,
    }
    assert records[1]["X"] is None
    assert records[1]["Y"] == 4.0


def test_fetch_station_reuses_hash_audited_chunk(tmp_path, monkeypatch):
    info = {
        "code": "TST",
        "dataset_id": "tst/best-avail/PT1M/xyzf",
        "start": "2024-01-01T00:00:00Z",
        "stop": "2025-01-01T00:00:00Z",
        "latitude": 10.0,
        "longitude": 20.0,
        "elevation_m": 30.0,
        "description": "test",
        "warnings": [],
    }
    payload = b"2024-01-01T00:00Z,1.0,2.0,3.0\n"
    monkeypatch.setattr(hapi, "fetch_station_info", lambda *_args, **_kwargs: info)
    monkeypatch.setattr(hapi, "_request", lambda *_args, **_kwargs: payload)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    first = hapi.fetch_station("TST", start, end, cache_dir=tmp_path)

    def fail_network(*_args, **_kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(hapi, "_request", fail_network)
    second = hapi.fetch_station("TST", start, end, cache_dir=tmp_path)

    assert second["records"] == first["records"]
    assert len(second["provenance"][0]["sha256"]) == 64


def test_vsh_condition_summary_tracks_outage_patterns():
    latitudes = np.asarray(
        [-80, -60, -40, -20, 0, 20, 40, 60, 80, -10, 10, 30], dtype=float
    )
    longitudes = np.asarray(
        [0, 40, 80, 120, 160, 200, 240, 280, 320, 20, 140, 260], dtype=float
    )
    mask = np.ones((3, len(latitudes), 3), dtype=bool)
    mask[1, 0] = False

    result = hapi.vsh_condition_summary(mask, latitudes, longitudes, lmax=2)

    assert result["n_coefficients"] == 25
    assert result["full_rank_fraction"] == 1.0
    assert np.isfinite(result["full_network"])
    assert result["max_observed"] >= result["median_observed"]


def test_build_network_dataset_emits_source_and_license_metadata(monkeypatch):
    def fake_station(code, *_args, **_kwargs):
        index = hapi.DEFAULT_GLOBAL_STATIONS.index(code)
        return {
            "code": code,
            "dataset_id": hapi.dataset_id(code),
            "latitude": -75.0 + index * 10.0,
            "longitude": index * 137.5 % 360,
            "elevation_m": 0.0,
            "start": "2024-01-01T00:00:00Z",
            "stop": "2025-01-01T00:00:00Z",
            "description": "test",
            "warnings": [],
            "provenance": [],
            "records": [
                {
                    "time": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "X": 1.0,
                    "Y": 2.0,
                    "Z": 3.0,
                }
            ],
        }

    monkeypatch.setattr(hapi, "fetch_station", fake_station)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc)

    dataset = hapi.build_network_dataset(start, end, workers=1)
    metadata = dataset["metadata"]

    assert dataset["station_values"].shape == (2, 16, 3)
    assert metadata["source"] == hapi.HAPI_BASE
    assert metadata["citation"] == hapi.CITATION_URL
    assert metadata["data_type"] == "best-avail"
    json.dumps(metadata)
