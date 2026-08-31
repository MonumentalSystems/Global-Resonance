from datetime import datetime, timezone

import numpy as np
import torch

from geomagnetic_operator_dataset import (
    align_drivers,
    decode_json_payload,
    fit_network_coefficients,
    forward_chaining_splits,
    hourly_reduce,
)
from operator_data_sources import (
    parse_goes_xray,
    parse_rtsw_magnetic,
    parse_rtsw_wind,
    parse_usgs_geomag,
)
from spherical_operator_experiment import real_spherical_harmonic_basis


def test_json_decoder_accepts_swpc_transport_padding():
    assert decode_json_payload(b'[{"active": true}]\x00\x00\n') == [{"active": True}]


def test_rtsw_parsers_select_active_quality_zero_spacecraft():
    rows = [
        {
            "time_tag": "2026-08-30T00:00:00",
            "active": False,
            "source": "ACE",
            "overall_quality": 0,
            "bt": 99.0,
            "bz_gsm": 99.0,
            "proton_speed": 999.0,
        },
        {
            "time_tag": "2026-08-30T00:00:00",
            "active": True,
            "source": "SOLAR1",
            "overall_quality": 0,
            "bt": 5.0,
            "by_gsm": 1.0,
            "bz_gsm": -2.0,
            "proton_speed": 450.0,
            "proton_density": 4.0,
        },
        {
            "time_tag": "2026-08-30T00:01:00",
            "active": True,
            "source": "SOLAR1",
            "overall_quality": 2,
            "bt": 88.0,
            "proton_speed": 888.0,
        },
        {
            "time_tag": "2026-08-30T00:02:00",
            "active": True,
            "source": "SOLAR1",
            "bt": 77.0,
            "proton_speed": 777.0,
        },
    ]
    magnetic = parse_rtsw_magnetic(rows)
    wind = parse_rtsw_wind(rows)
    assert len(magnetic) == len(wind) == 1
    assert magnetic[0]["bz_gsm"] == -2.0
    assert wind[0]["speed"] == 450.0


def test_goes_and_usgs_parsers_keep_quality_and_missingness():
    xray = parse_goes_xray(
        [
            {
                "time_tag": "2026-08-30T00:00:00Z",
                "energy": "0.1-0.8nm",
                "flux": 1e-6,
                "electron_contaminaton": True,
                "satellite": 18,
            },
            {
                "time_tag": "2026-08-30T00:00:00Z",
                "energy": "0.05-0.4nm",
                "flux": 1e-7,
            },
        ]
    )
    assert len(xray) == 1
    assert xray[0]["electron_contamination"] is True

    parsed = parse_usgs_geomag(
        {
            "metadata": {
                "intermagnet": {
                    "imo": {
                        "iaga_code": "TST",
                        "name": "Test",
                        "coordinates": [20.0, 10.0, 30.0],
                    },
                    "data_type": "variation",
                    "sampling_period": 60.0,
                }
            },
            "times": ["2026-08-30T00:00:00Z", "2026-08-30T00:01:00Z"],
            "values": [
                {"id": "X", "values": [1.0, None]},
                {"id": "Y", "values": [2.0, 3.0]},
                {"id": "Z", "values": [4.0, 5.0]},
            ],
        }
    )
    assert parsed["latitude"] == 10.0
    assert parsed["records"][1]["X"] is None


def test_hourly_alignment_never_zero_fills_missing_driver_values():
    records = [
        {
            "time": datetime(2026, 8, 30, 0, 5, tzinfo=timezone.utc),
            "bt": 4.0,
        },
        {
            "time": datetime(2026, 8, 30, 0, 55, tzinfo=timezone.utc),
            "bt": 6.0,
        },
    ]
    reduced = hourly_reduce(records, ("bt", "bz_gsm"))
    hours = [datetime(2026, 8, 30, 0, tzinfo=timezone.utc)]
    values, mask = align_drivers(
        hours, {"magnetic": reduced, "wind": {}, "xray": {}}
    )
    assert values[0, 0] == 5.0
    assert mask[0, 0]
    assert not mask[0, 2]


def test_masked_station_projection_recovers_bandlimited_coefficients():
    latitudes = np.array([-70, -45, -20, 0, 15, 35, 55, 72, -60, -30, 25, 65])
    longitudes = np.array([-160, -90, -20, 40, 100, 160, -130, -50, 70, 140, -170, 10])
    basis, _ = real_spherical_harmonic_basis(
        torch.tensor(latitudes), torch.tensor(longitudes), lmax=2
    )
    expected = np.linspace(-1.0, 1.0, 9)
    station_field = (basis.numpy() @ expected)[None, :, None]
    station_field = np.repeat(station_field, 3, axis=2)
    mask = np.ones_like(station_field, dtype=bool)
    recovered, recovered_mask, _ = fit_network_coefficients(
        station_field, mask, latitudes, longitudes, lmax=2, ridge=1e-10
    )
    np.testing.assert_allclose(recovered[0, 0], expected, atol=1e-7)
    assert recovered_mask.all()


def test_forward_splits_are_ordered_disjoint_and_gapped():
    splits = forward_chaining_splits(100, gap=2)
    assert splits["train"] == (0, 60)
    assert splits["train"][1] + 2 == splits["validation"][0]
    assert splits["validation"][1] + 2 == splits["test"][0]
