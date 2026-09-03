from datetime import datetime, timedelta, timezone

import numpy as np

import geomagnetic_operator_dataset as live_dataset
from historical_geomagnetic_operator import (
    align_omni_drivers,
    causal_pole_features,
    historical_ridge_controls,
    parse_omni2,
)
from geomagnetic_operator_dataset import fetch_usgs_station, forward_chaining_splits


def _omni_line(overrides=None):
    words = ["0"] * 55
    values = {
        0: "2024",
        1: "32",
        2: "5",
        4: "71",
        5: "51",
        8: "6.5",
        15: "-1.2",
        16: "-3.4",
        23: "4.5",
        24: "425",
        40: "-100",
        41: "700",
    }
    values.update(overrides or {})
    for index, value in values.items():
        words[index] = value
    return " ".join(words)


def test_omni_parser_uses_documented_upstream_words_and_fill_values():
    rows = parse_omni2(_omni_line() + "\n" + _omni_line({16: "999.9"}))

    assert rows[0]["time"] == datetime(2024, 2, 1, 5, tzinfo=timezone.utc)
    assert rows[0]["bt"] == 6.5
    assert rows[0]["by_gsm"] == -1.2
    assert rows[0]["bz_gsm"] == -3.4
    assert rows[0]["density"] == 4.5
    assert rows[0]["speed"] == 425.0
    assert "dst" not in rows[0]
    assert "ae" not in rows[0]
    assert rows[1]["bz_gsm"] is None


def test_omni_alignment_preserves_missing_driver_mask():
    hours = [
        datetime(2024, 1, 1, hour, tzinfo=timezone.utc) for hour in range(2)
    ]
    records = parse_omni2(_omni_line({1: "1", 2: "0", 24: "9999"}))
    values, mask = align_omni_drivers(hours, records)

    assert values.shape == mask.shape == (2, 5)
    assert not mask[0, 3]
    assert not mask[1].any()


def test_usgs_cache_is_resumable_and_hash_audited(tmp_path, monkeypatch):
    payload = {
        "metadata": {
            "intermagnet": {
                "imo": {
                    "iaga_code": "TST",
                    "name": "Test",
                    "coordinates": [20.0, 10.0, 30.0],
                },
                "data_type": "definitive",
                "sampling_period": 3600,
            }
        },
        "times": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
        "values": [
            {"id": "X", "values": [1.0, 2.0]},
            {"id": "Y", "values": [3.0, 4.0]},
            {"id": "Z", "values": [5.0, 6.0]},
        ],
    }
    monkeypatch.setattr(live_dataset, "fetch_json", lambda *_args, **_kwargs: payload)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    first = fetch_usgs_station(
        "TST",
        start,
        end,
        sampling_period=3600,
        data_type="definitive",
        cache_dir=tmp_path,
    )

    def fail_network(*_args, **_kwargs):
        raise AssertionError("cache miss")

    monkeypatch.setattr(live_dataset, "fetch_json", fail_network)
    second = fetch_usgs_station(
        "TST",
        start,
        end,
        sampling_period=3600,
        data_type="definitive",
        cache_dir=tmp_path,
    )

    assert second["records"] == first["records"]
    assert len(second["provenance"][0]["sha256"]) == 64


def test_fixed_pole_ridge_detects_multiscale_causal_memory():
    generator = np.random.default_rng(42)
    n_steps = 900
    half_lives = (2.0, 20.0)
    decays = np.exp(-np.log(2.0) / np.asarray(half_lives))
    drivers = generator.normal(size=(n_steps, 1)).astype(np.float32)
    driver_mask = np.ones_like(drivers, dtype=bool)
    hidden = np.zeros(2)
    coefficients = np.zeros((n_steps, 1, 1), dtype=np.float32)
    for time_index in range(n_steps - 1):
        hidden = decays * hidden + (1.0 - decays) * drivers[time_index, 0]
        coefficients[time_index + 1, 0, 0] = hidden.sum()
    coefficient_mask = np.ones_like(coefficients, dtype=bool)
    splits = forward_chaining_splits(n_steps)

    controls = historical_ridge_controls(
        coefficients,
        coefficient_mask,
        drivers,
        driver_mask,
        splits,
        half_lives=half_lives,
    )

    assert controls["markov_ridge"]["status"] == "ok"
    assert controls["parameter_matched_single_pole_ridge"]["status"] == "ok"
    assert (
        controls["parameter_matched_single_pole_ridge"]["parameters"]
        == controls["markov_ridge"]["parameters"]
    )
    assert controls["fixed_pole_ridge"]["status"] == "ok"
    assert controls["fixed_pole_improvement_fraction"] > 0.5


def test_causal_poles_reset_at_missing_driver_hours():
    drivers = np.ones((6, 1), dtype=np.float32)
    mask = np.ones_like(drivers, dtype=bool)
    mask[3] = False
    features, valid = causal_pole_features(
        drivers, mask, half_lives=(1.0,), burn_in_multiple=1.0
    )

    assert valid.tolist() == [True, True, True, False, True, True]
    assert features[3, 0] == 0.0
    assert features[4, 0] == 0.5
