import pytest

import crossyear_vector_spherical_replication as crossyear


def result(effect, validation_effect, daily_ci, weekly_ci):
    return {
        "improvement_fraction": effect,
        "validation_improvement_fraction": validation_effect,
        "paired_daily_bootstrap": {"all": {"ci95": daily_ci}},
        "paired_weekly_bootstrap": {"all": {"ci95": weekly_ci}},
    }


def test_replication_requires_direction_validation_and_both_intervals():
    discovery = result(0.1, 0.1, [0.01, 0.2], [0.01, 0.2])
    confirmed = result(0.08, 0.03, [0.01, 0.2], [0.02, 0.2])
    directional = result(0.08, 0.03, [-0.01, 0.2], [-0.02, 0.2])
    reversed_result = result(-0.08, -0.03, [-0.2, 0.01], [-0.2, 0.02])

    assert crossyear.replication_decision(discovery, confirmed)["status"] == "confirmed"
    assert (
        crossyear.replication_decision(discovery, directional)["status"]
        == "directional_not_interval_confirmed"
    )
    assert (
        crossyear.replication_decision(discovery, reversed_result)["status"]
        == "not_replicated"
    )


def test_crossyear_runner_rejects_changed_frozen_station_panel(monkeypatch, tmp_path):
    base = {
        "year": 2024,
        "station_codes": ["AAA", "BBB"],
        "station_latitudes": [10.0, -10.0],
        "station_longitudes": [20.0, 200.0],
        "lmax": 2,
        "labels": [["poloidal", 1, 0]],
        "coordinate_convention": {"vsh": ["theta"]},
        "source_cadence_seconds": 60,
        "target_cadence_seconds": 3600,
        "calendar_basis_names": ["intercept"],
    }
    changed = {**base, "year": 2023, "station_codes": ["AAA", "CCC"]}
    rows = iter((base, changed))
    monkeypatch.setattr(crossyear, "evaluate_frozen_hypothesis", lambda _: next(rows))

    with pytest.raises(ValueError, match="station_codes"):
        crossyear.run_crossyear_replication(
            tmp_path / "discovery.npz",
            tmp_path / "replication.npz",
            tmp_path / "result.npz",
        )
