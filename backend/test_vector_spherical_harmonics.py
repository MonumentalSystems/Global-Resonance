import math

import numpy as np
import torch

from geomagnetic_operator_dataset import forward_chaining_splits
from vector_spherical_harmonics import (
    fit_vector_network_coefficients,
    geomagnetic_xyz_to_spherical,
    real_vector_spherical_harmonic_basis,
    sector_indices,
)
from vector_spherical_operator_pilot import _select_sector_models


def _fibonacci_points(count):
    index = torch.arange(count, dtype=torch.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    latitude = torch.rad2deg(torch.asin(z))
    longitude = torch.rad2deg(
        torch.remainder(index * math.pi * (3.0 - math.sqrt(5.0)), 2.0 * math.pi)
    ) - 180.0
    return latitude, longitude


def test_vsh_basis_is_orthonormal_on_dense_sphere():
    latitude, longitude = _fibonacci_points(2000)
    basis, labels = real_vector_spherical_harmonic_basis(latitude, longitude, 2)
    gram = torch.einsum("ncm,nck->mk", basis, basis) * (4.0 * math.pi / len(latitude))

    torch.testing.assert_close(
        gram,
        torch.eye(len(labels), dtype=torch.float64),
        atol=8e-5,
        rtol=8e-5,
    )
    sectors = sector_indices(labels)
    assert {key: len(value) for key, value in sectors.items()} == {
        "radial": 9,
        "poloidal": 8,
        "toroidal": 8,
    }


def test_vsh_basis_can_be_constructed_under_no_grad():
    latitude, longitude = _fibonacci_points(20)
    with torch.no_grad():
        basis, labels = real_vector_spherical_harmonic_basis(
            latitude, longitude, lmax=2
        )
    assert basis.shape == (20, 3, len(labels))


def test_joint_vector_projection_recovers_bandlimited_field():
    latitude_t, longitude_t = _fibonacci_points(20)
    latitude = latitude_t.numpy()
    longitude = longitude_t.numpy()
    basis, labels = real_vector_spherical_harmonic_basis(
        latitude_t, longitude_t, lmax=2
    )
    expected = np.linspace(-1.0, 1.0, len(labels))
    field = np.einsum("scm,m->sc", basis.numpy(), expected)[None]
    mask = np.ones_like(field, dtype=bool)

    recovered, recovered_mask, recovered_labels, condition = (
        fit_vector_network_coefficients(
            field,
            mask,
            latitude,
            longitude,
            lmax=2,
            ridge=1e-10,
        )
    )

    np.testing.assert_allclose(recovered[0], expected, atol=2e-7)
    assert recovered_mask.all()
    assert recovered_labels == labels
    assert condition["full_network"] < 3.0
    assert condition["max_fitted"] < 3.0


def test_geomagnetic_xyz_conversion_uses_north_east_down_convention():
    values = np.asarray([[[1.0, 2.0, 3.0]]])
    mask = np.asarray([[[True, False, True]]])
    spherical, spherical_mask = geomagnetic_xyz_to_spherical(values, mask)

    np.testing.assert_array_equal(spherical, [[[-3.0, -1.0, 0.0]]])
    np.testing.assert_array_equal(spherical_mask, [[[True, True, False]]])


def test_sector_specific_poles_are_parameter_matched_to_sector_markov():
    generator = np.random.default_rng(7)
    n_steps = 360
    drivers = generator.normal(size=(n_steps, 2)).astype(np.float32)
    driver_mask = np.ones_like(drivers, dtype=bool)
    coefficients = generator.normal(size=(n_steps, 6)).astype(np.float32)
    coefficient_mask = np.ones_like(coefficients, dtype=bool)
    sectors = {
        "radial": np.asarray([0, 1]),
        "poloidal": np.asarray([2, 3]),
        "toroidal": np.asarray([4, 5]),
    }

    result = _select_sector_models(
        coefficients,
        coefficient_mask,
        drivers,
        driver_mask,
        forward_chaining_splits(n_steps),
        sectors,
        half_lives=(1.0, 3.0),
    )

    assert result["sector_specific_markov"]["status"] == "ok"
    assert result["sector_specific_single_poles"]["status"] == "ok"
    assert result["parameter_matched"] is True
    assert result["sector_specific_gated_poles"]["status"] == "ok"
    assert result["gated_parameter_matched"] is True
