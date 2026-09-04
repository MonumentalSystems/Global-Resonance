"""Low-degree real vector spherical harmonics for irregular station networks."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

try:
    from .spherical_operator_experiment import real_spherical_harmonic_basis
except ImportError:
    from spherical_operator_experiment import real_spherical_harmonic_basis


SECTORS = ("radial", "poloidal", "toroidal")


def real_vector_spherical_harmonic_basis(
    latitudes_deg: Tensor,
    longitudes_deg: Tensor,
    lmax: int,
) -> tuple[Tensor, tuple[tuple[str, int, int], ...]]:
    """Return orthonormal radial/poloidal/toroidal basis vectors.

    Components are ordered ``(outward radial, southward theta, eastward phi)``.
    Poloidal modes use the normalized surface gradient of ``Y_lm``. Toroidal
    modes use ``e_r x grad_s(Y_lm)``. Tangential sectors begin at ``l=1``.
    """

    latitudes_deg, longitudes_deg = torch.broadcast_tensors(
        latitudes_deg, longitudes_deg
    )
    with torch.enable_grad():
        theta = torch.deg2rad(90.0 - latitudes_deg).detach().clone()
        phi = torch.deg2rad(longitudes_deg).detach().clone()
        theta.requires_grad_(True)
        phi.requires_grad_(True)
        scalar_basis, scalar_modes = real_spherical_harmonic_basis(
            90.0 - torch.rad2deg(theta),
            torch.rad2deg(phi),
            lmax,
        )

        derivatives = []
        for mode_index in range(scalar_basis.shape[-1]):
            harmonic = scalar_basis[..., mode_index]
            d_theta, d_phi = torch.autograd.grad(
                harmonic.sum(),
                (theta, phi),
                retain_graph=True,
                allow_unused=True,
            )
            if d_theta is None:
                d_theta = torch.zeros_like(theta)
            if d_phi is None:
                d_phi = torch.zeros_like(phi)
            derivatives.append((d_theta, d_phi))

    zero = torch.zeros_like(theta)
    sin_theta = torch.sin(theta).clamp_min(torch.finfo(theta.dtype).eps)
    columns = []
    labels: list[tuple[str, int, int]] = []

    for mode_index, (degree, order) in enumerate(scalar_modes):
        columns.append(torch.stack((scalar_basis[..., mode_index], zero, zero), dim=-1))
        labels.append(("radial", degree, order))

    for sector in ("poloidal", "toroidal"):
        for mode_index, (degree, order) in enumerate(scalar_modes):
            if degree == 0:
                continue
            normalization = math.sqrt(degree * (degree + 1))
            d_theta, d_phi = derivatives[mode_index]
            theta_component = d_theta / normalization
            phi_component = d_phi / (sin_theta * normalization)
            if sector == "poloidal":
                vector = torch.stack((zero, theta_component, phi_component), dim=-1)
            else:
                vector = torch.stack((zero, -phi_component, theta_component), dim=-1)
            columns.append(vector)
            labels.append((sector, degree, order))

    return torch.stack(columns, dim=-1).detach(), tuple(labels)


def geomagnetic_xyz_to_spherical(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert X=north, Y=east, Z=down to outward-radial/theta/phi."""

    if values.shape != mask.shape or values.shape[-1] != 3:
        raise ValueError("expected matching (..., 3) values and masks")
    spherical = np.stack((-values[..., 2], -values[..., 0], values[..., 1]), axis=-1)
    spherical_mask = np.stack((mask[..., 2], mask[..., 0], mask[..., 1]), axis=-1)
    spherical[~spherical_mask] = 0.0
    return spherical, spherical_mask


def fit_vector_network_coefficients(
    spherical_values: np.ndarray,
    spherical_mask: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lmax: int = 2,
    ridge: float = 1e-4,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[tuple[str, int, int], ...],
    dict[str, float],
]:
    """Jointly fit all available vector components to a real VSH basis."""

    basis_t, labels = real_vector_spherical_harmonic_basis(
        torch.as_tensor(latitudes, dtype=torch.float64),
        torch.as_tensor(longitudes, dtype=torch.float64),
        lmax,
    )
    basis = basis_t.cpu().numpy().reshape(-1, basis_t.shape[-1])
    n_modes = basis.shape[1]
    coefficients = np.zeros((len(spherical_values), n_modes), dtype=np.float32)
    coefficient_mask = np.zeros_like(coefficients, dtype=bool)
    identity = np.eye(n_modes)
    finite_stations = np.isfinite(latitudes) & np.isfinite(longitudes)
    finite_rows = np.repeat(finite_stations, 3)
    complete_design = basis[finite_rows]
    full_network_condition = (
        float(np.linalg.cond(complete_design))
        if np.linalg.matrix_rank(complete_design) == n_modes
        else float("inf")
    )

    flat_values = spherical_values.reshape(len(spherical_values), -1)
    flat_mask = spherical_mask.reshape(len(spherical_mask), -1)
    fitted_conditions = []
    for time_index in range(len(spherical_values)):
        valid = flat_mask[time_index] & finite_rows
        design = basis[valid]
        if valid.sum() < n_modes or np.linalg.matrix_rank(design) < n_modes:
            continue
        fitted_conditions.append(float(np.linalg.cond(design)))
        target = flat_values[time_index, valid]
        gram = design.T @ design
        scale = np.trace(gram) / n_modes
        solution = np.linalg.solve(
            gram + ridge * max(scale, 1e-12) * identity,
            design.T @ target,
        )
        coefficients[time_index] = solution
        coefficient_mask[time_index] = True
    condition_summary = {"full_network": full_network_condition}
    if fitted_conditions:
        condition_summary.update(
            {
                "median_fitted": float(np.median(fitted_conditions)),
                "p95_fitted": float(np.quantile(fitted_conditions, 0.95)),
                "max_fitted": float(np.max(fitted_conditions)),
            }
        )
    else:
        condition_summary.update(
            {
                "median_fitted": float("inf"),
                "p95_fitted": float("inf"),
                "max_fitted": float("inf"),
            }
        )
    return coefficients, coefficient_mask, labels, condition_summary


def sector_indices(
    labels: tuple[tuple[str, int, int], ...],
) -> dict[str, np.ndarray]:
    return {
        sector: np.asarray(
            [index for index, label in enumerate(labels) if label[0] == sector],
            dtype=np.int64,
        )
        for sector in SECTORS
    }
