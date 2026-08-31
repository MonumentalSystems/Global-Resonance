#!/usr/bin/env python3
"""Spherical neural-operator benchmark with harmonic fixed-pole closure.

This is deliberately a small architecture experiment, not an operational
forecast.  It asks one question: after putting a field in the correct S^2
basis, does a causal multiscale state improve forecasts when several global
drivers act at different delays?

The synthetic generator is an exactly known sum of damped spherical modes.
The two learned models have nearly the same parameterization:

* SphericalMarkovOperator mixes all pole branches instantaneously.
* SphericalPoleOperator retains those branches as causal recurrent state.

Both preserve the l=0 mode, share transition parameters across every m at a
given degree l, and avoid pointwise MLPs, normalization, dropout, auxiliary
losses, and gradient clipping.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_HALF_LIVES = (1.5, 6.0, 24.0, 96.0)


def _associated_legendre(degree: int, m: int, x: Tensor) -> Tensor:
    """Associated Legendre P_l^m(x), including the Condon-Shortley phase."""

    if not 0 <= m <= degree:
        raise ValueError(f"expected 0 <= m <= l, got l={degree}, m={m}")
    p_mm = torch.ones_like(x)
    if m:
        root = torch.sqrt(torch.clamp(1.0 - x.square(), min=0.0))
        factor = 1.0
        for _ in range(m):
            p_mm = -factor * root * p_mm
            factor += 2.0
    if degree == m:
        return p_mm
    p_m1m = (2 * m + 1) * x * p_mm
    if degree == m + 1:
        return p_m1m
    p_lm2, p_lm1 = p_mm, p_m1m
    for current_degree in range(m + 2, degree + 1):
        p_lm = (
            (2 * current_degree - 1) * x * p_lm1
            - (current_degree + m - 1) * p_lm2
        ) / (current_degree - m)
        p_lm2, p_lm1 = p_lm1, p_lm
    return p_lm1


class RealSphericalHarmonicTransform(nn.Module):
    """Small differentiable real SHT for band-limited benchmark fields.

    A weighted least-squares dual corrects the finite equiangular quadrature,
    so analysis(synthesis(coefficients)) is exact to numerical precision for
    the retained modes.  This implementation is intentionally small; a real
    high-resolution deployment should use torch-harmonics/makani kernels.
    """

    def __init__(self, nlat: int = 12, nlon: int = 24, lmax: int = 5):
        super().__init__()
        if nlat < lmax + 1 or nlon < 2 * lmax + 1:
            raise ValueError("grid is too small for requested spherical modes")
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.lmax = int(lmax)

        theta = (torch.arange(nlat, dtype=torch.float64) + 0.5) * math.pi / nlat
        phi = (torch.arange(nlon, dtype=torch.float64) + 0.5) * 2.0 * math.pi / nlon
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
        x = torch.cos(theta_grid)

        basis = []
        modes: list[tuple[int, int]] = []
        for degree in range(lmax + 1):
            for m_signed in range(-degree, degree + 1):
                m = abs(m_signed)
                log_norm = 0.5 * (
                    math.log((2 * degree + 1) / (4 * math.pi))
                    + math.lgamma(degree - m + 1)
                    - math.lgamma(degree + m + 1)
                )
                y = math.exp(log_norm) * _associated_legendre(degree, m, x)
                if m_signed > 0:
                    y = math.sqrt(2.0) * y * torch.cos(m * phi_grid)
                elif m_signed < 0:
                    y = math.sqrt(2.0) * y * torch.sin(m * phi_grid)
                basis.append(y.reshape(-1))
                modes.append((degree, m_signed))

        synthesis = torch.stack(basis)
        weights = (
            torch.sin(theta_grid) * (math.pi / nlat) * (2.0 * math.pi / nlon)
        ).reshape(-1)
        weighted_basis = synthesis * weights.unsqueeze(0)
        gram = weighted_basis @ synthesis.T
        analysis = torch.linalg.solve(gram, weighted_basis)

        self.modes = tuple(modes)
        self.register_buffer("synthesis_basis", synthesis.float())
        self.register_buffer("analysis_basis", analysis.float())
        self.register_buffer(
            "mode_degrees",
            torch.tensor([degree for degree, _ in modes], dtype=torch.long),
        )

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def analysis(self, field: Tensor) -> Tensor:
        if field.shape[-2:] != (self.nlat, self.nlon):
            raise ValueError(
                f"expected (..., {self.nlat}, {self.nlon}), got {tuple(field.shape)}"
            )
        flat = field.reshape(*field.shape[:-2], -1)
        return torch.einsum("...p,mp->...m", flat, self.analysis_basis)

    def synthesis(self, coefficients: Tensor) -> Tensor:
        if coefficients.shape[-1] != self.n_modes:
            raise ValueError(f"expected {self.n_modes} modes")
        flat = torch.einsum("...m,mp->...p", coefficients, self.synthesis_basis)
        return flat.reshape(*coefficients.shape[:-1], self.nlat, self.nlon)


@dataclass(frozen=True)
class BenchmarkConfig:
    nlat: int = 12
    nlon: int = 24
    lmax: int = 5
    n_drivers: int = 8
    n_sequences: int = 64
    train_steps: int = 96
    validation_steps: int = 32
    test_steps: int = 64
    burn_in_steps: int = 128
    epochs: int = 240
    learning_rate: float = 3e-3
    half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES


@dataclass
class SyntheticBatch:
    coefficients: Tensor
    drivers: Tensor
    mode_degrees: Tensor


def _pole_decays(half_lives: Iterable[float], *, dtype=torch.float32) -> Tensor:
    half_lives_t = torch.tensor(tuple(half_lives), dtype=dtype)
    return torch.exp(-math.log(2.0) / half_lives_t)


def generate_multistream_sphere(config: BenchmarkConfig, seed: int) -> SyntheticBatch:
    """Generate a partially observed multirate field on S^2.

    Each driver has its own autocorrelation time.  Each spherical mode receives
    a different mixture of four hidden causal cavities.  Only their sum is
    observed, so the current field is not a sufficient Markov state.
    """

    transform = RealSphericalHarmonicTransform(config.nlat, config.nlon, config.lmax)
    generator = torch.Generator().manual_seed(seed + 10_000)
    n_total = (
        config.burn_in_steps
        + config.train_steps
        + config.validation_steps
        + config.test_steps
    )
    noise = torch.randn(
        config.n_sequences, n_total, config.n_drivers, generator=generator
    )
    driver_decay = torch.linspace(0.08, 0.94, config.n_drivers)
    driver_scale = torch.sqrt(1.0 - driver_decay.square())
    drivers = torch.zeros_like(noise)
    for t in range(n_total):
        previous = drivers[:, t - 1] if t else torch.zeros_like(drivers[:, 0])
        drivers[:, t] = driver_decay * previous + driver_scale * noise[:, t]

    true_half_lives = torch.tensor((2.0, 8.0, 32.0, 128.0))
    true_decays = torch.exp(-math.log(2.0) / true_half_lives)
    n_poles = len(true_half_lives)
    n_modes = transform.n_modes
    degree_scale = (1.0 + transform.mode_degrees.float()).pow(-1.35)
    injection = torch.randn(
        n_poles, config.n_drivers, n_modes, generator=generator
    )
    injection *= degree_scale.view(1, 1, -1) / math.sqrt(config.n_drivers)
    injection[:, :, 0] = 0.0  # conserve the global mean (l=0)

    hidden = torch.zeros(config.n_sequences, n_poles, n_modes)
    recorded_drivers = []
    recorded_coefficients = []
    global_mean = 0.15 * torch.randn(config.n_sequences, generator=generator)

    for t in range(n_total):
        drive = torch.einsum("bd,pdm->bpm", drivers[:, t], injection)
        hidden = true_decays.view(1, -1, 1) * hidden + drive
        if t == config.burn_in_steps - 1:
            coeff = hidden.sum(dim=1) / math.sqrt(n_poles)
            coeff[:, 0] = global_mean
            recorded_coefficients.append(coeff)
        elif t >= config.burn_in_steps:
            recorded_drivers.append(drivers[:, t])
            coeff = hidden.sum(dim=1) / math.sqrt(n_poles)
            coeff[:, 0] = global_mean
            recorded_coefficients.append(coeff)

    return SyntheticBatch(
        coefficients=torch.stack(recorded_coefficients, dim=1),
        drivers=torch.stack(recorded_drivers, dim=1),
        mode_degrees=transform.mode_degrees.clone(),
    )


class _SphericalOperatorBase(nn.Module):
    def __init__(self, n_drivers: int, mode_degrees: Tensor, half_lives: Iterable[float]):
        super().__init__()
        self.n_drivers = int(n_drivers)
        self.n_modes = int(mode_degrees.numel())
        self.n_degrees = int(mode_degrees.max().item()) + 1
        self.n_poles = len(tuple(half_lives))
        self.register_buffer("mode_degrees", mode_degrees.clone().long())
        self.register_buffer("nonconstant_mask", (mode_degrees != 0).float())
        self.driver_to_poles = nn.Linear(
            self.n_drivers, self.n_poles * self.n_modes, bias=False
        )
        # RotationalAdamW deliberately preserves each projection row's norm.
        # A separate dynamical scale therefore carries the physically meaningful
        # degree-dependent amplitude without applying L2 decay to it.
        degree_scale = (1.0 + torch.arange(self.n_degrees)).pow(-1.35)
        initial_scale = degree_scale / math.sqrt(self.n_poles)
        self.log_input_scale = nn.Parameter(
            initial_scale.log().repeat(self.n_poles, 1)
        )

    def _driver_branches(self, drivers: Tensor) -> Tensor:
        branches = self.driver_to_poles(drivers).reshape(
            drivers.shape[0], self.n_poles, self.n_modes
        )
        scale = torch.exp(self.log_input_scale.clamp(-8.0, 4.0))[:, self.mode_degrees]
        return branches * scale.unsqueeze(0) * self.nonconstant_mask.view(1, 1, -1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SphericalMarkovOperator(_SphericalOperatorBase):
    """Parameter-matched instantaneous spherical spectral operator."""

    def __init__(self, n_drivers: int, mode_degrees: Tensor, half_lives=DEFAULT_HALF_LIVES):
        super().__init__(n_drivers, mode_degrees, half_lives)
        self.raw_carry = nn.Parameter(torch.full((self.n_degrees,), 1.4))
        self.branch_coeffs = nn.Parameter(
            torch.ones(self.n_poles, self.n_degrees)
        )

    def step(self, previous: Tensor, drivers: Tensor, state=None) -> tuple[Tensor, None]:
        carry = torch.sigmoid(self.raw_carry)[self.mode_degrees]
        branch_weights = self.branch_coeffs[:, self.mode_degrees]
        forcing = (self._driver_branches(drivers) * branch_weights.unsqueeze(0)).sum(dim=1)
        predicted = carry.unsqueeze(0) * previous + forcing
        predicted = torch.cat((previous[:, :1], predicted[:, 1:]), dim=-1)
        return predicted, None

    def predict_sequence(
        self, initial: Tensor, drivers: Tensor, observations: Tensor | None = None
    ) -> Tensor:
        previous = initial
        outputs = []
        for t in range(drivers.shape[1]):
            if observations is not None:
                previous = observations[:, t]
            previous, _ = self.step(previous, drivers[:, t])
            outputs.append(previous)
        return torch.stack(outputs, dim=1)


class SphericalPoleOperator(_SphericalOperatorBase):
    """Causal fixed-pole closure in the spherical harmonic domain."""

    def __init__(self, n_drivers: int, mode_degrees: Tensor, half_lives=DEFAULT_HALF_LIVES):
        super().__init__(n_drivers, mode_degrees, half_lives)
        self.register_buffer("pole_decay", _pole_decays(half_lives))
        self.pole_gate_logits = nn.Parameter(torch.zeros(self.n_poles, self.n_degrees))

    def initialize(self, coefficients: Tensor) -> Tensor:
        allocation = torch.softmax(self.pole_gate_logits, dim=0)[:, self.mode_degrees]
        return coefficients.unsqueeze(1) * allocation.unsqueeze(0)

    def step(self, previous: Tensor, drivers: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        allocation = torch.softmax(self.pole_gate_logits, dim=0)[:, self.mode_degrees]
        correction = previous - state.sum(dim=1)
        state = state + correction.unsqueeze(1) * allocation.unsqueeze(0)
        state = self.pole_decay.view(1, -1, 1) * state + self._driver_branches(drivers)
        predicted = state.sum(dim=1)
        predicted = torch.cat((previous[:, :1], predicted[:, 1:]), dim=-1)
        state = torch.cat((state[:, :, :1] * 0.0, state[:, :, 1:]), dim=-1)
        state[:, 0, 0] = previous[:, 0]
        return predicted, state

    def predict_sequence(
        self, initial: Tensor, drivers: Tensor, observations: Tensor | None = None
    ) -> Tensor:
        previous = initial
        state = self.initialize(initial)
        outputs = []
        for t in range(drivers.shape[1]):
            if observations is not None:
                previous = observations[:, t]
            previous, state = self.step(previous, drivers[:, t], state)
            outputs.append(previous)
        return torch.stack(outputs, dim=1)


def _degree_mse(prediction: Tensor, target: Tensor, degrees: Tensor) -> dict[str, float]:
    result = {}
    for degree in range(int(degrees.max().item()) + 1):
        mask = degrees == degree
        result[str(degree)] = F.mse_loss(prediction[..., mask], target[..., mask]).item()
    return result


@torch.no_grad()
def evaluate_operator(
    model: _SphericalOperatorBase,
    coefficients: Tensor,
    drivers: Tensor,
    start: int,
    stop: int,
) -> dict:
    initial = coefficients[:, start]
    inputs = drivers[:, start:stop]
    target = coefficients[:, start + 1 : stop + 1]
    observations = coefficients[:, start:stop]
    one_step = model.predict_sequence(initial, inputs, observations=observations)
    rollout = model.predict_sequence(initial, inputs, observations=None)
    persistence = initial.unsqueeze(1).expand_as(target)
    variance = target[..., 1:].var().item() + 1e-12
    target_rms = target[..., 1:].square().mean(dim=-1).sqrt()
    rollout_rms = rollout[..., 1:].square().mean(dim=-1).sqrt()
    return {
        "one_step_mse": F.mse_loss(one_step[..., 1:], target[..., 1:]).item(),
        "rollout_mse": F.mse_loss(rollout[..., 1:], target[..., 1:]).item(),
        "rollout_nmse": F.mse_loss(rollout[..., 1:], target[..., 1:]).item() / variance,
        "persistence_rollout_mse": F.mse_loss(
            persistence[..., 1:], target[..., 1:]
        ).item(),
        "final_step_mse": F.mse_loss(rollout[:, -1, 1:], target[:, -1, 1:]).item(),
        "max_rms_ratio": (rollout_rms.max() / (target_rms.max() + 1e-12)).item(),
        "mean_mode_max_abs_drift": (
            rollout[..., 0] - initial[:, 0].unsqueeze(1)
        ).abs().max().item(),
        "rollout_mse_by_degree": _degree_mse(rollout, target, model.mode_degrees),
    }


def _make_optimizer(model: nn.Module, lr: float):
    try:
        from harmonic_gpt.utils.training import make_rotational_optimizer

        return make_rotational_optimizer(model, lr=lr, weight_decay=0.01), "RotationalAdamW"
    except ImportError as exc:  # pragma: no cover - deployment convenience only
        raise RuntimeError(
            "Training this research experiment requires the harmonic-gpt package "
            "and its RotationalAdamW optimizer."
        ) from exc


def _preflight() -> dict:
    try:
        from harmonic_gpt.utils import preflight
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install harmonic-gpt before running training") from exc
    return preflight(require_wandb=True)


def train_condition(
    model_name: str,
    batch: SyntheticBatch,
    config: BenchmarkConfig,
    seed: int,
    output_dir: Path,
) -> tuple[dict, str]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = {
        "markov": SphericalMarkovOperator,
        "fixed_pole": SphericalPoleOperator,
    }[model_name]
    model = model_cls(config.n_drivers, batch.mode_degrees, config.half_lives).to(device)
    coefficients = batch.coefficients.to(device)
    drivers = batch.drivers.to(device)
    optimizer, optimizer_name = _make_optimizer(model, config.learning_rate)

    import wandb

    run = wandb.init(
        project="symbiogenesis",
        tags=["harmonic-gpt", "global-resonance", "neural-operator", "spherical", model_name],
        name=f"global-s2-closure-{model_name}-seed{seed}",
        config={**asdict(config), "model": model_name, "seed": seed},
        reinit="finish_previous",
    )
    best_loss = float("inf")
    best_state = None
    observations = coefficients[:, : config.train_steps]
    target = coefficients[:, 1 : config.train_steps + 1]
    train_drivers = drivers[:, : config.train_steps]
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        prediction = model.predict_sequence(
            coefficients[:, 0], train_drivers, observations=observations
        )
        loss = F.mse_loss(prediction[..., 1:], target[..., 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        metrics = evaluate_operator(
            model,
            coefficients,
            drivers,
            config.train_steps,
            config.train_steps + config.validation_steps,
        )
        if metrics["rollout_mse"] < best_loss:
            best_loss = metrics["rollout_mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            wandb.log({"epoch": epoch, "train/mse": loss.item(), **{f"val/{k}": v for k, v in metrics.items() if isinstance(v, float)}})

    assert best_state is not None
    model.load_state_dict(best_state)
    test_start = config.train_steps + config.validation_steps
    final_metrics = evaluate_operator(
        model,
        coefficients,
        drivers,
        test_start,
        test_start + config.test_steps,
    )
    final_metrics.update(
        {
            "model": model_name,
            "seed": seed,
            "parameters": model.count_params(),
            "optimizer": optimizer_name,
            "selection_validation_rollout_mse": best_loss,
            "wall_clock_seconds": time.perf_counter() - started,
        }
    )
    checkpoint = output_dir / f"{model_name}_seed{seed}.pt"
    torch.save(
        {"model_state": best_state, "config": asdict(config), "metrics": final_metrics},
        checkpoint,
    )
    wandb.log({f"final/{k}": v for k, v in final_metrics.items() if isinstance(v, float)})
    run_url = run.url
    run.finish()
    return final_metrics, run_url


def _aggregate(results: list[dict]) -> dict:
    aggregate = {}
    for model_name in ("markov", "fixed_pole"):
        rows = [row for row in results if row["model"] == model_name]
        aggregate[model_name] = {}
        for metric in ("one_step_mse", "rollout_mse", "rollout_nmse", "final_step_mse", "max_rms_ratio"):
            values = np.array([row[metric] for row in rows], dtype=np.float64)
            aggregate[model_name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    markov = aggregate["markov"]["rollout_mse"]["mean"]
    fixed = aggregate["fixed_pole"]["rollout_mse"]["mean"]
    aggregate["fixed_pole_rollout_improvement_fraction"] = 1.0 - fixed / markov
    return aggregate


def run_experiment(config: BenchmarkConfig, output_dir: Path, seeds=DEFAULT_SEEDS) -> dict:
    _preflight()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    run_urls = []
    for seed in seeds:
        batch = generate_multistream_sphere(config, seed)
        for model_name in ("markov", "fixed_pole"):
            metrics, run_url = train_condition(
                model_name, batch, config, seed, output_dir
            )
            results.append(metrics)
            run_urls.append(run_url)
            print(
                f"{model_name:10s} seed={seed} "
                f"one_step={metrics['one_step_mse']:.6g} "
                f"rollout={metrics['rollout_mse']:.6g}"
            )
    report = {
        "experiment_id": "global_resonance_s2_fixed_pole_closure_20260830",
        "status": "complete",
        "config": asdict(config),
        "seeds": list(seeds),
        "results": results,
        "aggregate": _aggregate(results),
        "wandb_runs": run_urls,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=BenchmarkConfig.epochs)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output" / "spherical_operator_experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(epochs=args.epochs)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    run_experiment(config, args.output_dir, seeds)


if __name__ == "__main__":
    main()
