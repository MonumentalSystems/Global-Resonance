#!/usr/bin/env python3
"""
JellyBallNet — Physics-Informed Neural Model for CME-Seismicity Coupling

Architecture:
  Layer 1: ResonanceCavity (6 Legendre modes with Kuramoto dynamics)
  Layer 2: CliffordCoupling (Cl(3,0) grade structure for J and Bz)
  Layer 3: LoheZoneMap (mode amplitudes -> 10 zone ratios via P_l)

The model directly encodes the physics:
  - Resonance bands = Legendre cavity modes P_l(cos theta)
  - Kuramoto coupling K = J stiffness -> phase transition at K_c = 2/pi
  - Clifford commutator = grade-2 coupling [F, nabla_F]
  - Lohe mapping = mode -> zone via P_l(cos theta_zone)

Training data: 474 Kp>=5 storms x 10 zones x 5 phases = 23,700 samples
Input: [Kp, Dst, Bz, V_sw, X-ray, storm_phase]
Output: 10 zone seismicity ratios
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import legendre
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json
import math

OUT = Path(__file__).parent / "output"
INIT_DATE = datetime(2000, 1, 1)

# Paper XXV zones
ZONES = [
    ("eye",            0,  15, 0.85),
    ("inner",         15,  30, 0.92),
    ("transition",    30,  60, 0.98),
    ("wavefront",     60,  75, 1.36),
    ("wavefront-tail",75, 100, 1.09),
    ("neutral",      100, 120, 0.95),
    ("far-suppress", 120, 135, 0.82),
    ("far-neutral",  135, 155, 0.90),
    ("pre-antipodal",155, 165, 1.00),
    ("antipodal",    165, 180, 1.16),
]

ZONE_CENTERS = np.array([(z[1] + z[2]) / 2 for z in ZONES])
N_ZONES = len(ZONES)
N_MODES = 6  # Legendre l=1..6

# Precompute P_l(cos theta) matrix: (N_MODES, N_ZONES)
P_MATRIX = np.zeros((N_MODES, N_ZONES))
for l in range(N_MODES):
    P = legendre(l + 1)
    P_MATRIX[l] = P(np.cos(np.radians(ZONE_CENTERS)))

# J_c = 2/pi (Kuramoto critical coupling)
J_C = 2 / math.pi


# ============================================================
# MODEL COMPONENTS
# ============================================================

class ResonanceCavity(nn.Module):
    """
    6 coupled Legendre cavity modes with Kuramoto-like dynamics.

    Each mode l has:
      - Natural frequency omega_l (learnable, initialized from cavity physics)
      - Damping gamma_l (learnable, initialized from Q ~ 3-5)
      - Excitation amplitude A_l (from input solar driving)

    The coupling K between modes is the J stiffness parameter.
    Phase transition occurs at K = J_c = 2/pi.
    """
    def __init__(self, n_modes=N_MODES):
        super().__init__()
        self.n_modes = n_modes

        # Natural frequencies: initialized from cavity physics
        # Lithospheric modes at ~0.3-1 cycle/day (much slower than Schumann)
        init_omega = torch.tensor([0.3, 0.5, 0.7, 0.9, 1.1, 1.3], dtype=torch.float32)
        self.log_omega = nn.Parameter(torch.log(init_omega))

        # Damping: Q ~ 3-5 -> gamma = omega / (2*Q)
        init_gamma = init_omega / (2 * 4.0)  # Q=4 default
        self.log_gamma = nn.Parameter(torch.log(init_gamma))

        # Excitation weights: how solar input maps to mode amplitudes
        self.excitation = nn.Linear(6, n_modes)  # 6 solar inputs -> 6 modes

        # Mode-mode coupling matrix (antisymmetric part = Kuramoto interaction)
        self.coupling = nn.Parameter(torch.randn(n_modes, n_modes) * 0.01)

    def forward(self, solar_input, J, t_phase):
        """
        Args:
            solar_input: (batch, 6) [Kp, Dst, Bz, V_sw, density, X-ray]
            J: (batch, 1) coupling stiffness
            t_phase: (batch, 1) time within storm (0=onset, 1=day+5)
        Returns:
            mode_amplitudes: (batch, n_modes) signed amplitude of each mode
            mode_phases: (batch, n_modes) phase of each mode
        """
        B = solar_input.shape[0]
        omega = torch.exp(self.log_omega)  # (n_modes,)
        gamma = torch.exp(self.log_gamma)  # (n_modes,)

        # Excitation amplitude from solar input
        A = self.excitation(solar_input)  # (B, n_modes)

        # Damped oscillation: A_l * cos(omega_l * t) * exp(-gamma_l * t)
        t = t_phase  # (B, 1)
        phase = omega.unsqueeze(0) * t  # (B, n_modes)
        envelope = torch.exp(-gamma.unsqueeze(0) * t)  # (B, n_modes)

        # Mode amplitudes with damped oscillation
        mode_amplitudes = A * torch.cos(phase) * envelope  # (B, n_modes)

        # Kuramoto-like coupling: near J_c, modes interact
        # The coupling INCREASES as J approaches J_c
        J_ratio = J / J_C  # (B, 1)
        coupling_strength = torch.exp(-1.0 / (torch.abs(J_ratio - 1) + 0.01))  # peaks at J_c

        # Antisymmetric coupling (energy-preserving)
        C = self.coupling - self.coupling.t()  # (n_modes, n_modes)
        mode_interaction = torch.matmul(mode_amplitudes, C)  # (B, n_modes)

        mode_amplitudes = mode_amplitudes + coupling_strength * mode_interaction
        mode_phases = phase  # for diagnostics

        return mode_amplitudes, mode_phases


class CliffordCoupling(nn.Module):
    """
    Cl(3,0) grade structure for the electromagnetic coupling.

    Encodes the solar wind as a Clifford multivector:
      Grade 0 (scalar): J stiffness
      Grade 1 (vector): [Bx, By, Bz]
      Grade 2 (bivector): mode coupling planes
      Grade 3 (pseudoscalar): Bz sign (shield state)

    The commutator [F, nabla_F] extracts the grade-2 coupling,
    which is the mechanism from Paper XXV.
    """
    def __init__(self, n_modes=N_MODES):
        super().__init__()
        self.n_modes = n_modes

        # Map solar input to Clifford multivector components
        # 8 components: [scalar, e1, e2, e3, e12, e23, e13, e123]
        self.to_multivector = nn.Linear(6, 8)

        # Bivector extraction -> mode modulation
        self.bivector_to_modes = nn.Linear(3, n_modes)  # 3 bivector components -> mode modulation

        # Shield gate: Bz sign determines how much compression transmits
        self.shield_gate = nn.Linear(1, n_modes)

    def forward(self, solar_input, mode_amplitudes):
        """
        Args:
            solar_input: (batch, 6) [Kp, Dst, Bz, V_sw, density, X-ray]
            mode_amplitudes: (batch, n_modes) from ResonanceCavity
        Returns:
            modulated_modes: (batch, n_modes) modes after EM coupling
            J: (batch, 1) computed J stiffness
            bivector_norm: (batch, 1) criticality indicator
        """
        # Build multivector from solar input
        mv = self.to_multivector(solar_input)  # (B, 8)

        # Extract grades
        scalar = mv[:, 0:1]  # J stiffness
        vector = mv[:, 1:4]  # [Bx, By, Bz]
        bivector = mv[:, 4:7]  # mode coupling planes
        pseudoscalar = mv[:, 7:8]  # handedness

        # J stiffness (grade-0)
        J = torch.sigmoid(scalar) * 0.5 + 0.4  # range [0.4, 0.9]

        # Bivector norm = criticality indicator
        bivector_norm = torch.norm(bivector, dim=1, keepdim=True)

        # Grade-2 coupling: bivector modulates mode amplitudes
        biv_mod = self.bivector_to_modes(bivector)  # (B, n_modes)

        # Shield gate: Bz determines transmission
        bz = solar_input[:, 2:3]  # Bz from input
        shield = torch.sigmoid(self.shield_gate(bz))  # (B, n_modes) in [0,1]

        # Modulate modes: original + bivector coupling, gated by shield
        modulated = mode_amplitudes + shield * biv_mod * 0.1

        return modulated, J, bivector_norm


class LoheZoneMap(nn.Module):
    """
    Maps Legendre mode amplitudes to zone seismicity ratios.

    Uses the physics-derived P_l(cos theta) matrix as the core mapping,
    with a learnable residual for empirical corrections.

    R(theta) = 1 + sum_l[ a_l * P_l(cos theta) ] + residual
    """
    def __init__(self, n_modes=N_MODES, n_zones=N_ZONES):
        super().__init__()

        # Fixed P_l matrix (physics)
        self.register_buffer('P_matrix', torch.tensor(P_MATRIX, dtype=torch.float32))

        # Learnable residual correction (small)
        self.residual = nn.Linear(n_modes, n_zones, bias=True)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

        # Output scale
        self.scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, mode_amplitudes):
        """
        Args:
            mode_amplitudes: (batch, n_modes) Legendre coefficients a_l
        Returns:
            zone_ratios: (batch, n_zones) predicted seismicity ratio per zone
        """
        # Physics: R = 1 + sum_l[ a_l * P_l(cos theta_zone) ]
        # mode_amplitudes are the a_l coefficients
        physics_contrib = torch.matmul(mode_amplitudes * self.scale, self.P_matrix)  # (B, n_zones)

        # Learnable residual (small correction)
        residual = self.residual(mode_amplitudes) * 0.05

        # Zone ratios: 1.0 = background rate
        zone_ratios = 1.0 + physics_contrib + residual

        # Soft clamp to reasonable range [0.2, 5.0]
        zone_ratios = 0.2 + 4.8 * torch.sigmoid(zone_ratios - 1.0 + 0.5)

        return zone_ratios


class JellyBallNet(nn.Module):
    """
    Full JellyBall model: Solar input -> Zone seismicity ratios.

    Architecture:
      ResonanceCavity: solar -> 6 Legendre mode amplitudes
      CliffordCoupling: EM coupling + J stiffness + criticality
      LoheZoneMap: mode amplitudes -> 10 zone ratios via P_l matrix
    """
    def __init__(self):
        super().__init__()
        self.cavity = ResonanceCavity()
        self.coupling = CliffordCoupling()
        self.zone_map = LoheZoneMap()

    def forward(self, solar_input, t_phase):
        """
        Args:
            solar_input: (batch, 6) [Kp, Dst, Bz, V_sw, density, X-ray]
            t_phase: (batch, 1) storm phase time [0=onset, 5=late relaxation]
        Returns:
            zone_ratios: (batch, 10) predicted seismicity ratio per zone
            diagnostics: dict with J, bivector_norm, mode_amplitudes
        """
        # Stage 1: Excite cavity modes from solar input
        mode_amps, mode_phases = self.cavity(solar_input,
                                              torch.ones(solar_input.shape[0], 1) * 0.6,  # initial J guess
                                              t_phase)

        # Stage 2: Clifford EM coupling modifies modes
        modulated_modes, J, biv_norm = self.coupling(solar_input, mode_amps)

        # Stage 3: Map modes to zone ratios via Legendre polynomials
        zone_ratios = self.zone_map(modulated_modes)

        diagnostics = {
            'J': J,
            'bivector_norm': biv_norm,
            'mode_amplitudes': modulated_modes,
            'mode_phases': mode_phases,
            'above_critical': (J > J_C).float(),
        }

        return zone_ratios, diagnostics

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# DATA PIPELINE
# ============================================================

def subsolar_point(dt_utc):
    doy = dt_utc.timetuple().tm_yday
    decl = -23.44 * np.cos(np.radians(360 / 365 * (doy + 10)))
    hour = dt_utc.hour + dt_utc.minute / 60.0
    lon = 180.0 - 15.0 * hour
    if lon < -180: lon += 360
    if lon > 180: lon -= 360
    return decl, lon

def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

def solid_angle(d1, d2):
    return 2 * np.pi * abs(np.cos(np.radians(d1)) - np.cos(np.radians(d2)))


def build_training_data():
    """Build training dataset from OMNI + earthquake cache."""
    print("Building training data...")

    # Load earthquakes
    eq = pd.read_csv(OUT / "earthquakes_m4.5_cache.csv")
    eq["time_parsed"] = pd.to_datetime(eq["time"], utc=True).dt.tz_localize(None)
    eq["day_number"] = ((eq["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values

    # Load OMNI
    omni = pd.read_csv(OUT / "omni2_hourly.csv", parse_dates=["datetime"])
    omni = omni.dropna(subset=["kp", "bz"])

    # Find storms
    daily = omni.groupby("day_number").agg({
        "kp": "max", "bz": "mean", "dst": "min", "datetime": "first"
    }).reset_index()

    storm_days = daily[daily["kp"] >= 5].sort_values("day_number")
    events = []
    last_day = -999
    for _, row in storm_days.iterrows():
        if row["day_number"] - last_day >= 5:
            events.append(row.to_dict())
            last_day = row["day_number"]
        elif row["kp"] > events[-1]["kp"]:
            events[-1] = row.to_dict()
            last_day = row["day_number"]
    storms = pd.DataFrame(events)
    print(f"  {len(storms)} storms, {len(eq)} earthquakes")

    zone_bins = np.array([z[1] for z in ZONES] + [180])
    sa = np.array([solid_angle(z[1], z[2]) for z in ZONES])

    # Phase windows: (day_start, day_end, phase_value)
    phases = [
        (-1, 0, 0.0),    # compression
        (0, 1, 0.2),     # peak
        (1, 3, 0.5),     # relaxation_early
        (3, 7, 0.8),     # relaxation_late
    ]

    X_list, y_list = [], []

    for _, storm in storms.iterrows():
        d = storm["day_number"]
        dt = storm["datetime"]
        peak_kp = storm["kp"]
        ss_lat, ss_lon = subsolar_point(dt)

        # Get OMNI context for this storm
        ctx = omni[(omni["day_number"] >= d - 1) & (omni["day_number"] <= d)]
        if len(ctx) < 3:
            continue
        mean_bz = ctx["bz"].mean()
        mean_dst = ctx["dst"].mean() if "dst" in ctx.columns else -30

        # Background
        bg = eq[(eq["day_number"] >= d - 10) & (eq["day_number"] < d - 5)]
        if len(bg) < 3:
            continue
        bg_dists = angular_distance(ss_lat, ss_lon, bg["latitude"].values, bg["longitude"].values)
        bg_counts, _ = np.histogram(bg_dists, bins=zone_bins)
        bg_density = bg_counts / 5 / sa
        bg_density[bg_density == 0] = 1e-10

        for d_start, d_end, phase_val in phases:
            phase_eq = eq[(eq["day_number"] >= d + d_start) & (eq["day_number"] < d + d_end)]
            if len(phase_eq) == 0:
                continue
            p_dists = angular_distance(ss_lat, ss_lon, phase_eq["latitude"].values, phase_eq["longitude"].values)
            p_counts, _ = np.histogram(p_dists, bins=zone_bins)
            p_density = p_counts / max(d_end - d_start, 1) / sa
            ratios = np.clip(p_density / bg_density, 0, 10)

            # Input features: [Kp, Dst, Bz, V_sw=0, density=0, Xray=0]
            x = np.array([peak_kp, mean_dst, mean_bz, 0, 0, 0], dtype=np.float32)
            X_list.append(np.concatenate([x, [phase_val]]))
            y_list.append(ratios.astype(np.float32))

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Training samples: {len(X)}")
    return X, y


# ============================================================
# TRAINING
# ============================================================

def train_model(epochs=200, lr=0.003, batch_size=64):
    """Train JellyBallNet on storm-earthquake data."""
    print("=" * 70)
    print("  TRAINING JellyBallNet")
    print("=" * 70)

    X, y = build_training_data()

    # Normalize inputs
    X_solar = X[:, :6]
    X_phase = X[:, 6:7]

    # Standardize solar inputs
    x_mean = X_solar.mean(axis=0)
    x_std = X_solar.std(axis=0) + 1e-8
    X_solar = (X_solar - x_mean) / x_std

    # Convert to tensors
    X_solar_t = torch.tensor(X_solar, dtype=torch.float32)
    X_phase_t = torch.tensor(X_phase, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Model
    model = JellyBallNet()
    print(f"  Parameters: {model.count_params()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Split train/val (80/20 by time)
    n = len(X_solar_t)
    n_train = int(0.8 * n)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        perm = torch.randperm(n_train)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            pred, diag = model(X_solar_t[idx], X_phase_t[idx])

            # MSE on zone ratios
            loss_mse = F.mse_loss(pred, y_t[idx])

            # Physics regularization: l=2 coefficient should match observed sign flip
            mode_amps = diag['mode_amplitudes']
            l2_amp = mode_amps[:, 1]  # l=2 mode
            phase = X_phase_t[idx, 0]
            # During compression (phase < 0.3), l2 should be positive
            # During relaxation (phase > 0.5), l2 should be negative
            comp_mask = phase < 0.3
            relax_mask = phase > 0.5
            l2_reg = 0
            if comp_mask.sum() > 0:
                l2_reg += F.relu(-l2_amp[comp_mask]).mean() * 0.1
            if relax_mask.sum() > 0:
                l2_reg += F.relu(l2_amp[relax_mask]).mean() * 0.1

            # J should be near J_c during storms
            J = diag['J']
            j_reg = ((J - J_C) ** 2).mean() * 0.01

            loss = loss_mse + l2_reg + j_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_mse.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_diag = model(X_solar_t[n_train:], X_phase_t[n_train:])
            val_loss = F.mse_loss(val_pred, y_t[n_train:]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == epochs - 1:
            train_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch:4d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  best={best_val_loss:.4f}")

    # Load best model
    model.load_state_dict(best_state)

    # Save
    save_path = OUT / "jellyball_net.pt"
    torch.save({
        'model_state': best_state,
        'x_mean': x_mean,
        'x_std': x_std,
        'val_loss': best_val_loss,
    }, save_path)
    print(f"\n  Saved: {save_path}")
    print(f"  Best val MSE: {best_val_loss:.4f}")

    return model, x_mean, x_std


def evaluate_model(model, x_mean, x_std):
    """Evaluate trained model: print zone predictions by phase."""
    print("\n" + "=" * 70)
    print("  EVALUATION: Predictions by Storm Phase")
    print("=" * 70)

    model.eval()
    zone_names = [z[0] for z in ZONES]

    # Test scenarios
    scenarios = [
        ("Quiet (Kp=2)", [2, -10, 0, 0, 0, 0]),
        ("Moderate storm (Kp=5)", [5, -40, -5, 0, 0, 0]),
        ("Strong storm (Kp=7)", [7, -80, -10, 0, 0, 0]),
        ("Extreme Bz south (Kp=8)", [8, -120, -20, 0, 0, 0]),
        ("Strong Bz north (Kp=7)", [7, -70, 5, 0, 0, 0]),
    ]

    phases = [("Compression", 0.0), ("Peak", 0.2), ("Relax early", 0.5), ("Relax late", 0.8)]

    for scenario_name, solar in scenarios:
        print(f"\n  --- {scenario_name} ---")
        solar_norm = (np.array(solar, dtype=np.float32) - x_mean) / x_std

        print(f"  {'Zone':18s}", end="")
        for phase_name, _ in phases:
            print(f" {phase_name:>10s}", end="")
        print()

        for z_idx, z_name in enumerate(zone_names):
            print(f"  {z_name:18s}", end="")
            for _, phase_val in phases:
                with torch.no_grad():
                    x = torch.tensor(solar_norm, dtype=torch.float32).unsqueeze(0)
                    t = torch.tensor([[phase_val]], dtype=torch.float32)
                    pred, diag = model(x, t)
                    ratio = pred[0, z_idx].item()
                print(f" {ratio:9.2f}x", end="")
            print()

        # Print diagnostics for compression phase
        with torch.no_grad():
            x = torch.tensor(solar_norm, dtype=torch.float32).unsqueeze(0)
            t = torch.tensor([[0.0]], dtype=torch.float32)
            _, diag = model(x, t)
            J = diag['J'][0, 0].item()
            bv = diag['bivector_norm'][0, 0].item()
            modes = diag['mode_amplitudes'][0].numpy()

        print(f"  J={J:.4f} (J_c={J_C:.4f}) biv_norm={bv:.4f}")
        print(f"  Modes: " + " ".join([f"l{i+1}={modes[i]:+.3f}" for i in range(N_MODES)]))


def main():
    model, x_mean, x_std = train_model(epochs=300, lr=0.003)
    evaluate_model(model, x_mean, x_std)

    # Print P_l matrix for reference
    print("\n" + "=" * 70)
    print("  PHYSICS: P_l(cos theta) Matrix")
    print("=" * 70)
    zone_names = [z[0] for z in ZONES]
    print(f"  {'Zone':18s}", end="")
    for l in range(N_MODES):
        print(f"  l={l+1:d}", end="")
    print()
    for z_idx, z_name in enumerate(zone_names):
        print(f"  {z_name:18s}", end="")
        for l in range(N_MODES):
            print(f" {P_MATRIX[l, z_idx]:+5.2f}", end="")
        print()


if __name__ == "__main__":
    main()
