from datetime import datetime, timedelta, timezone

import numpy as np

import residualized_multihorizon_operator as residualized


def hourly_timestamps(count: int) -> list[datetime]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(hours=index) for index in range(count)]


def test_calendar_basis_has_fixed_daily_seasonal_interactions():
    design, names = residualized.calendar_harmonic_design(hourly_timestamps(48))

    assert design.shape == (48, 15)
    assert names[:7] == (
        "intercept",
        "sin_24h",
        "cos_24h",
        "sin_12h",
        "cos_12h",
        "sin_annual",
        "cos_annual",
    )
    assert len(set(names)) == len(names)
    assert np.allclose(design[:, 0], 1.0)


def test_calendar_fit_recovers_basis_signal_and_extrapolates():
    timestamps = hourly_timestamps(24 * 120)
    design, _ = residualized.calendar_harmonic_design(timestamps)
    rng = np.random.default_rng(42)
    true_weights = rng.normal(size=(design.shape[1], 3))
    coefficients = design @ true_weights
    mask = np.ones_like(coefficients, dtype=bool)

    baseline, remainder, diagnostics = residualized.fit_robust_calendar_baseline(
        coefficients, mask, design, (0, 24 * 70)
    )

    assert diagnostics["uses_validation_or_test_coefficients"] is False
    assert np.max(np.abs(baseline - coefficients)) < 2e-4
    assert np.max(np.abs(remainder)) < 2e-4


def test_calendar_fit_does_not_leak_validation_or_test_coefficients():
    timestamps = hourly_timestamps(24 * 80)
    design, _ = residualized.calendar_harmonic_design(timestamps)
    rng = np.random.default_rng(123)
    coefficients = rng.normal(size=(len(timestamps), 2))
    changed = coefficients.copy()
    changed[24 * 40 :] += 1e6
    mask = np.ones_like(coefficients, dtype=bool)

    baseline_a, _, diagnostics_a = residualized.fit_robust_calendar_baseline(
        coefficients, mask, design, (0, 24 * 40)
    )
    baseline_b, _, diagnostics_b = residualized.fit_robust_calendar_baseline(
        changed, mask, design, (0, 24 * 40)
    )

    assert np.array_equal(baseline_a, baseline_b)
    assert diagnostics_a == diagnostics_b


def test_robust_calendar_fit_downweights_training_outlier():
    timestamps = hourly_timestamps(24 * 60)
    design, _ = residualized.calendar_harmonic_design(timestamps)
    coefficients = np.column_stack((design[:, 1], design[:, 2]))
    coefficients[500] += 1e4
    mask = np.ones_like(coefficients, dtype=bool)

    _, _, diagnostics = residualized.fit_robust_calendar_baseline(
        coefficients, mask, design, (0, 24 * 40)
    )

    assert diagnostics["minimum_row_weight"] < 0.01
    assert diagnostics["fraction_downweighted"] > 0.0
