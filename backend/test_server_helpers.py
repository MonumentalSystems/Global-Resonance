from datetime import datetime, timedelta, timezone

import pytest

import server
from server import fill_vector_grid, get_solar_wind, lunar_phase, subsolar_point


def test_subsolar_point_is_near_equator_and_greenwich_at_march_equinox_noon():
    point = subsolar_point(datetime(2026, 3, 20, 12, tzinfo=timezone.utc))

    assert point["lat"] == pytest.approx(0, abs=1)
    assert point["lon"] == pytest.approx(0, abs=0.01)


@pytest.mark.parametrize(
    ("days", "expected_illumination"),
    [(0, 0), (29.53059 / 4, 50), (29.53059 / 2, 100), (3 * 29.53059 / 4, 50)],
)
def test_lunar_illumination_is_symmetric_around_full_moon(days, expected_illumination):
    reference = datetime(2000, 1, 6, tzinfo=timezone.utc)
    phase = lunar_phase(reference + timedelta(days=days))

    assert phase["illumination"] == pytest.approx(expected_illumination, abs=0.1)


def test_fill_vector_grid_preserves_samples_and_fills_supported_neighbors():
    u, v, speed = fill_vector_grid(
        [1.0, None, 3.0, None, None, None],
        [0.0, None, 4.0, None, None, None],
        n_lat=2,
        n_lon=3,
        passes=1,
    )

    assert (u[0], v[0], speed[0]) == (1.0, 0.0, 1.0)
    assert (u[2], v[2], speed[2]) == (3.0, 4.0, 5.0)
    assert u[1] == pytest.approx(2.0)
    assert v[1] == pytest.approx(2.0)
    assert speed[1] == pytest.approx(2.828)


def test_solar_wind_endpoint_uses_active_quality_records(monkeypatch):
    magnetic = [
        {
            "time_tag": "2026-08-30T00:00:00Z",
            "active": False,
            "overall_quality": 0,
            "bz_gsm": -99.0,
        },
        {
            "time_tag": "2026-08-30T00:00:00Z",
            "active": True,
            "overall_quality": 0,
            "source": "SOLAR1",
            "bt": 6.0,
            "by_gsm": 1.0,
            "bz_gsm": -3.5,
        },
    ]
    wind = [
        {
            "time_tag": "2026-08-30T00:00:00Z",
            "active": True,
            "overall_quality": 0,
            "source": "SOLAR1",
            "proton_speed": 425.0,
            "proton_density": 4.2,
        }
    ]

    def fake_cached_fetch(key, _url, ttl=server.CACHE_TTL):
        del ttl
        return magnetic if key == "sw_mag" else wind

    monkeypatch.setattr(server, "cached_fetch", fake_cached_fetch)
    result = get_solar_wind()

    assert result["current_bz"] == -3.5
    assert result["current_speed"] == 425.0
    assert result["current_density"] == 4.2
    assert result["bz"][0]["source"] == "SOLAR1"
