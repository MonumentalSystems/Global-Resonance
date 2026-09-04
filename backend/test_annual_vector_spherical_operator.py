from datetime import datetime, timedelta, timezone

import numpy as np

import annual_vector_spherical_operator as annual


def test_hourly_median_requires_enough_real_samples():
    values = np.arange(120 * 2 * 3, dtype=np.float32).reshape(120, 2, 3)
    mask = np.ones_like(values, dtype=bool)
    mask[:31, 0, 0] = False

    hourly, hourly_mask, counts = annual.hourly_median_network(
        values, mask, minimum_samples=30
    )

    assert hourly.shape == hourly_mask.shape == counts.shape == (2, 2, 3)
    assert not hourly_mask[0, 0, 0]
    assert hourly[0, 0, 0] == 0.0
    assert counts[0, 0, 0] == 29
    assert hourly_mask[1].all()


def test_omni_indices_are_parsed_as_evaluation_labels():
    words = ["0"] * 55
    words[0:3] = ["2024", "123", "4"]
    words[38] = "87"
    words[40] = "-203"

    rows = annual.parse_omni_evaluation_indices(" ".join(words))

    assert rows[0]["time"] == datetime(2024, 5, 2, 4, tzinfo=timezone.utc)
    assert rows[0]["kp"] == 8.7
    assert rows[0]["dst"] == -203.0


def test_calendar_split_keeps_q4_untouched_with_gaps():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    hours = [start + timedelta(hours=index) for index in range(int((end - start).total_seconds() / 3600))]

    splits = annual.calendar_quarter_splits(hours, gap_hours=24)

    assert hours[splits["train"][1]] == datetime(2024, 7, 1, tzinfo=timezone.utc)
    assert hours[splits["validation"][0]] == datetime(2024, 7, 2, tzinfo=timezone.utc)
    assert hours[splits["test"][0]] == datetime(2024, 10, 2, tzinfo=timezone.utc)
    assert splits["test"][1] == len(hours)


def test_temporal_controls_are_parameter_matched_and_select_on_validation():
    rng = np.random.default_rng(42)
    n_steps = 1400
    drivers = rng.normal(size=(n_steps, 2)).astype(np.float32)
    driver_mask = np.ones_like(drivers, dtype=bool)
    fast = annual.exponential_state_features(drivers, driver_mask, 6.0)
    slow = annual.exponential_state_features(drivers, driver_mask, 24.0)
    hidden_sum = fast.sum(axis=1) + slow.sum(axis=1)
    coefficients = np.zeros((n_steps, 3), dtype=np.float32)
    coefficients[1:, 0] = hidden_sum[:-1]
    coefficients[1:, 1] = 0.5 * hidden_sum[:-1]
    coefficients[1:, 2] = -0.25 * hidden_sum[:-1]
    coefficient_mask = np.ones_like(coefficients, dtype=bool)
    labels = (
        ("radial", 0, 0),
        ("poloidal", 1, 0),
        ("toroidal", 1, 1),
    )
    splits = {"train": (0, 700), "validation": (724, 1050), "test": (1074, 1400)}
    evaluation_masks = {
        "storm": np.zeros(n_steps, dtype=bool),
        "severe": np.zeros(n_steps, dtype=bool),
        "quiet": np.ones(n_steps, dtype=bool),
    }

    result = annual.run_controls(
        coefficients,
        coefficient_mask,
        drivers,
        driver_mask,
        splits,
        labels,
        evaluation_masks,
        half_lives=(1.0, 6.0, 24.0),
        warmup_hours=24,
    )

    assert result["parameter_matched"]
    selected = result["models"]["single_pole"]
    assert selected["selection_validation_mse"] == min(
        selected["candidate_validation_mse"].values()
    )
    assert len({model["test_rows"] for model in result["models"].values()}) == 1
    uncertainty = result["paired_daily_bootstrap"]["single_pole"]["all"]
    assert uncertainty["rows"] == result["models"]["single_pole"]["test_rows"]
    assert len(uncertainty["ci95"]) == 2
    assert len(uncertainty["ci99_44"]) == 2
    assert "_test_row_mse" not in result["models"]["single_pole"]


def test_signed_cycle_recurrence_is_stable_and_resets_on_gap():
    drivers = np.ones((100, 5), dtype=np.float32)
    mask = np.ones_like(drivers, dtype=bool)
    mask[50] = False

    features = annual.orthogonal_recurrent_features(drivers, mask, 24.0)

    assert np.isfinite(features).all()
    assert np.abs(features).max() < 2.0
    assert np.all(features[50] == 0.0)


def test_control_rows_align_multi_hour_origin_and_target():
    coefficients = np.arange(10, dtype=np.float32)[:, None]
    coefficient_mask = np.ones_like(coefficients, dtype=bool)
    features = (100 + np.arange(10, dtype=np.float32))[:, None]
    ready = np.ones(10, dtype=bool)

    x, y, target_times = annual._control_rows(
        features,
        ready,
        coefficients,
        coefficient_mask,
        np.asarray([0]),
        lead_hours=3,
    )

    assert target_times.tolist() == list(range(3, 10))
    assert x[:, 0].tolist() == list(range(7))
    assert x[:, 1].tolist() == list(range(100, 107))
    assert y[:, 0].tolist() == list(range(3, 10))
