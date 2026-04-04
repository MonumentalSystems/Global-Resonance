#!/usr/bin/env python3
"""
Jelly Ball as Ringing Bell — Spherical Harmonic Modal Decomposition

The Paper XXV zones are not arbitrary angular bins — they correspond
to the nodes and antinodes of Legendre polynomial cavity modes.

Key result: The l=2 (quadrupole) mode coefficient FLIPS SIGN between
compression and relaxation phases of geomagnetic storms. This explains
the far-suppress zone inversion observed in the backtest (p=0.0017).

Model: R(theta, t) = 1 + sum_l[ a_l(t) * P_l(cos theta) ]

where: a_l(t) = A_l * cos(omega_l * t + phi_l) * exp(-gamma_l * t)

Same Legendre polynomial mode shapes as Schumann resonances,
but at lithospheric timescales (~1 cycle/day vs 7.83 Hz).
"""
import numpy as np
from scipy.special import legendre
from scipy.optimize import minimize
import json
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Paper XXV 10-zone model
ZONES = [
    ("eye",            7.5,  0.85),
    ("inner",         22.5,  0.92),
    ("transition",    45.0,  0.98),
    ("wavefront",     67.5,  1.36),
    ("wavefront-tail",87.5,  1.09),
    ("neutral",      110.0,  0.95),
    ("far-suppress", 127.5,  0.82),
    ("far-neutral",  145.0,  0.90),
    ("pre-antipodal",160.0,  1.00),
    ("antipodal",    172.5,  1.16),
]


def fit_legendre(targets, theta_deg, n_modes=6):
    """Fit Legendre polynomial coefficients to observed zone ratios."""
    cos_t = np.cos(np.radians(theta_deg))

    def model(a):
        result = np.ones_like(cos_t)
        for l, coeff in enumerate(a, start=1):
            result += coeff * legendre(l)(cos_t)
        return result

    def cost(a):
        return np.sum((model(a) - targets) ** 2)

    res = minimize(cost, np.zeros(n_modes), method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-8})
    return res.x, model(res.x)


def main():
    zone_names = [z[0] for z in ZONES]
    theta_centers = np.array([z[1] for z in ZONES])
    expected = np.array([z[2] for z in ZONES])

    # Decompose Paper XXV static pattern
    a_static, recon_static = fit_legendre(expected, theta_centers)

    print("=" * 70)
    print("JELLY BALL MODAL DECOMPOSITION")
    print("=" * 70)
    print("\nPaper XXV static pattern -> Legendre coefficients:")
    for l in range(len(a_static)):
        f_schumann = (3e8 / (2 * np.pi * 6.371e6)) * np.sqrt((l+1) * (l+2))
        print(f"  l={l+1}  a={a_static[l]:+.4f}  (Schumann f_{l+1} = {f_schumann:.1f} Hz)")

    # Load backtest phase ratios if available
    ratios_file = OUT / "jellyball_phase_ratios.json"
    if ratios_file.exists():
        with open(ratios_file) as f:
            ratios = json.load(f)

        print("\nPhase-resolved modal coefficients:")
        print(f"  {'Phase':25s} {'l=1':>8s} {'l=2':>8s} {'l=3':>8s} {'l=4':>8s} {'l=5':>8s} {'l=6':>8s}")
        print("  " + "-" * 73)

        for phase in ['compression', 'peak', 'relaxation_early', 'relaxation_late']:
            obs = np.array([ratios[phase].get(z, 1.0) for z in zone_names])
            a_phase, _ = fit_legendre(obs, theta_centers)
            coeffs = "  ".join([f"{a:+.3f}" for a in a_phase])
            print(f"  {phase:25s} {coeffs}")

        # l=2 sign flip analysis
        comp_obs = np.array([ratios['compression'].get(z, 1.0) for z in zone_names])
        relax_obs = np.array([ratios['relaxation_late'].get(z, 1.0) for z in zone_names])
        a_comp, _ = fit_legendre(comp_obs, theta_centers)
        a_relax, _ = fit_legendre(relax_obs, theta_centers)

        print(f"\n  l=2 sign flip: compression={a_comp[1]:+.3f} -> relaxation={a_relax[1]:+.3f}")
        if a_comp[1] * a_relax[1] < 0:
            print("  ** CONFIRMED: l=2 inverts between compression and relaxation **")

    # Legendre node analysis
    print("\n" + "=" * 70)
    print("LEGENDRE NODES vs ZONE BOUNDARIES")
    print("=" * 70)
    theta = np.linspace(0, 180, 10000)
    for l in range(1, 7):
        P = legendre(l)
        vals = P(np.cos(np.radians(theta)))
        zeros = []
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                t = theta[i] + (theta[i+1] - theta[i]) * abs(vals[i]) / (abs(vals[i]) + abs(vals[i+1]))
                zeros.append(t)
        zones_str = ", ".join([f"{z:.0f}°" for z in zeros])
        print(f"  l={l}: nodes at {zones_str}")

    # Generate continuous model prediction
    theta_fine = np.linspace(0, 180, 500)
    prediction = np.ones_like(theta_fine)
    for l, a in enumerate(a_static, start=1):
        prediction += a * legendre(l)(np.cos(np.radians(theta_fine)))

    # Save for potential frontend use
    model_data = {
        "static_coefficients": {f"l{l+1}": float(a_static[l]) for l in range(len(a_static))},
        "zone_reconstruction": {zone_names[i]: float(recon_static[i]) for i in range(len(zone_names))},
        "continuous_prediction": {
            "theta_deg": theta_fine.tolist(),
            "ratio": prediction.tolist(),
        },
    }
    with open(OUT / "jellyball_modal_model.json", "w") as f:
        json.dump(model_data, f)
    print(f"\nSaved: {OUT / 'jellyball_modal_model.json'}")


if __name__ == "__main__":
    main()
