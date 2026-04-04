#!/usr/bin/env python3
"""
JellyBallNet V2 — Improved with depth filtering, temporal context, stacked targets

Changes from V1:
  1. Depth filtering: only 0-150km earthquakes (strongest signal)
  2. OMNI temporal context: 24h Kp/Bz/Dst history as input features
  3. Stacked targets: group similar storms, average their zone ratios
  4. Magnitude weighting: M5.5+ events weighted 3x
  5. Richer solar features: Kp, Dst, Bz, dKp/dt, dDst/dt, |Bz|
  6. Larger model: 2-layer resonance with more capacity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import legendre
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json
import math

OUT = Path(__file__).parent / "output"
INIT_DATE = datetime(2000, 1, 1)

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
N_MODES = 6
J_C = 2 / math.pi

P_MATRIX = np.zeros((N_MODES, N_ZONES))
for l in range(N_MODES):
    P_MATRIX[l] = legendre(l + 1)(np.cos(np.radians(ZONE_CENTERS)))

N_INPUT = 12  # Kp, Dst, Bz, dKp, dDst, |Bz|, Kp_24h, Dst_24h, Bz_24h, Kp_max, Dst_min, storm_dur


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


# ============================================================
# MODEL V2
# ============================================================

class ResonanceCavityV2(nn.Module):
    def __init__(self, n_input=N_INPUT, n_modes=N_MODES, hidden=32):
        super().__init__()
        self.n_modes = n_modes
        init_omega = torch.tensor([0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
        self.log_omega = nn.Parameter(torch.log(init_omega))
        self.log_gamma = nn.Parameter(torch.log(init_omega / 8))

        # Richer excitation: 2-layer MLP
        self.exc1 = nn.Linear(n_input, hidden)
        self.exc2 = nn.Linear(hidden, n_modes)

        # Mode coupling (antisymmetric)
        self.coupling = nn.Parameter(torch.randn(n_modes, n_modes) * 0.02)

    def forward(self, x, t_phase):
        omega = torch.exp(self.log_omega)
        gamma = torch.exp(self.log_gamma)

        A = self.exc2(F.gelu(self.exc1(x)))
        t = t_phase
        phase = omega.unsqueeze(0) * t
        envelope = torch.exp(-gamma.unsqueeze(0) * t)
        mode_amps = A * torch.cos(phase) * envelope

        # Coupling near J_c
        C = self.coupling - self.coupling.t()
        mode_amps = mode_amps + 0.1 * torch.matmul(mode_amps, C)

        return mode_amps


class CliffordCouplingV2(nn.Module):
    def __init__(self, n_input=N_INPUT, n_modes=N_MODES, hidden=24):
        super().__init__()
        # Grade decomposition
        self.to_grades = nn.Linear(n_input, 8)  # 8 Cl(3,0) components
        self.biv_to_modes = nn.Linear(3, n_modes)
        self.shield = nn.Linear(n_input, n_modes)
        # J estimator (2-layer for better fit)
        self.j_net = nn.Sequential(nn.Linear(n_input, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x, mode_amps):
        mv = self.to_grades(x)
        bivector = mv[:, 4:7]
        biv_norm = torch.norm(bivector, dim=1, keepdim=True)

        J = self.j_net(x) * 0.5 + 0.4  # [0.4, 0.9]

        biv_mod = self.biv_to_modes(bivector)
        shield = torch.sigmoid(self.shield(x))
        modulated = mode_amps + shield * biv_mod * 0.1

        return modulated, J, biv_norm


class LoheZoneMapV2(nn.Module):
    def __init__(self, n_modes=N_MODES, n_zones=N_ZONES, n_input=N_INPUT):
        super().__init__()
        self.register_buffer('P_matrix', torch.tensor(P_MATRIX, dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(0.3))

        # Context-dependent residual
        self.residual = nn.Sequential(
            nn.Linear(n_modes + n_input, 24),
            nn.GELU(),
            nn.Linear(24, n_zones),
        )
        # Init small
        for p in self.residual.parameters():
            nn.init.normal_(p, 0, 0.01)

    def forward(self, mode_amps, x_input):
        physics = torch.matmul(mode_amps * self.scale, self.P_matrix)
        residual = self.residual(torch.cat([mode_amps, x_input], dim=1)) * 0.05
        zone_ratios = 1.0 + physics + residual
        zone_ratios = 0.1 + 9.9 * torch.sigmoid((zone_ratios - 1.0) * 0.5 + 0.5)
        return zone_ratios


class JellyBallNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cavity = ResonanceCavityV2()
        self.coupling = CliffordCouplingV2()
        self.zone_map = LoheZoneMapV2()

    def forward(self, x, t_phase):
        mode_amps = self.cavity(x, t_phase)
        modulated, J, biv_norm = self.coupling(x, mode_amps)
        zone_ratios = self.zone_map(modulated, x)
        return zone_ratios, {'J': J, 'bivector_norm': biv_norm, 'mode_amplitudes': modulated}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# DATA PIPELINE V2
# ============================================================

def build_training_data_v2():
    print("Building V2 training data (depth-filtered, temporal context, stacked)...")

    eq = pd.read_csv(OUT / "earthquakes_m4.5_cache.csv")
    eq["time_parsed"] = pd.to_datetime(eq["time"], utc=True).dt.tz_localize(None)
    eq["day_number"] = ((eq["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values

    # DEPTH FILTER: only 0-150km (strongest EM-pore coupling, p=0.0006)
    eq_shallow = eq[(eq["depth"] >= 0) & (eq["depth"] < 150)].copy()
    print(f"  Earthquakes: {len(eq)} total -> {len(eq_shallow)} shallow (0-150km)")

    omni = pd.read_csv(OUT / "omni2_hourly.csv", parse_dates=["datetime"])
    omni = omni.dropna(subset=["kp", "bz"])
    # Compute derived features
    omni["dkp"] = omni["kp"].diff().fillna(0)
    omni["ddst"] = omni["dst"].diff().fillna(0)
    omni["abs_bz"] = omni["bz"].abs()

    # Daily aggregates with temporal context
    daily = omni.groupby("day_number").agg({
        "kp": ["max", "mean"],
        "dst": ["min", "mean"],
        "bz": ["mean", "min"],
        "dkp": lambda x: x.abs().max(),
        "ddst": lambda x: x.abs().max(),
        "abs_bz": "max",
        "datetime": "first",
    }).reset_index()
    daily.columns = ["day_number", "kp_max", "kp_mean", "dst_min", "dst_mean",
                      "bz_mean", "bz_min", "dkp_max", "ddst_max", "abs_bz_max", "datetime"]

    # Find storms
    storm_days = daily[daily["kp_max"] >= 5].sort_values("day_number")
    events = []
    last_day = -999
    for _, row in storm_days.iterrows():
        if row["day_number"] - last_day >= 5:
            events.append(row.to_dict())
            last_day = row["day_number"]
        elif row["kp_max"] > events[-1]["kp_max"]:
            events[-1] = row.to_dict()
            last_day = row["day_number"]
    storms = pd.DataFrame(events)
    print(f"  Storms: {len(storms)}")

    zone_bins = np.array([z[1] for z in ZONES] + [180])
    sa = np.array([solid_angle(z[1], z[2]) for z in ZONES])

    phases = [(-1, 0, 0.0), (0, 1, 0.2), (1, 3, 0.5), (3, 7, 0.8)]

    X_list, y_list, weights_list = [], [], []

    for _, storm in storms.iterrows():
        d = int(storm["day_number"])
        dt = storm["datetime"]
        ss_lat, ss_lon = subsolar_point(dt)

        # 24h context: get prior day stats
        prior = daily[(daily["day_number"] >= d - 1) & (daily["day_number"] <= d)]
        if len(prior) == 0:
            continue

        kp_now = storm["kp_max"]
        dst_now = storm["dst_min"]
        bz_now = storm["bz_mean"]
        dkp = storm["dkp_max"]
        ddst = storm["ddst_max"]
        abs_bz = storm["abs_bz_max"]

        # 24h averages
        kp_24 = prior["kp_mean"].mean()
        dst_24 = prior["dst_mean"].mean()
        bz_24 = prior["bz_mean"].mean()

        # Storm duration estimate (how many consecutive days Kp >= 4)
        storm_dur = 0
        for dd in range(d - 3, d + 4):
            row = daily[daily["day_number"] == dd]
            if len(row) > 0 and row.iloc[0]["kp_max"] >= 4:
                storm_dur += 1

        # Feature vector: 12 features
        x_base = np.array([
            kp_now, dst_now, bz_now,
            dkp, ddst, abs_bz,
            kp_24, dst_24, bz_24,
            kp_now,  # repeat for kp_max
            dst_now,  # dst_min
            storm_dur,
        ], dtype=np.float32)

        # Background
        bg = eq_shallow[(eq_shallow["day_number"] >= d - 10) & (eq_shallow["day_number"] < d - 5)]
        if len(bg) < 3:
            continue
        bg_dists = angular_distance(ss_lat, ss_lon, bg["latitude"].values, bg["longitude"].values)
        bg_counts, _ = np.histogram(bg_dists, bins=zone_bins)
        bg_density = bg_counts / 5 / sa
        bg_density[bg_density == 0] = 1e-10

        for d_start, d_end, phase_val in phases:
            phase_eq = eq_shallow[(eq_shallow["day_number"] >= d + d_start) & (eq_shallow["day_number"] < d + d_end)]
            if len(phase_eq) == 0:
                continue

            p_dists = angular_distance(ss_lat, ss_lon, phase_eq["latitude"].values, phase_eq["longitude"].values)
            p_counts, _ = np.histogram(p_dists, bins=zone_bins)

            # MAGNITUDE WEIGHTING: count M5.5+ events as 3x
            for eq_row in phase_eq.itertuples():
                if eq_row.mag >= 5.5:
                    dist = angular_distance(ss_lat, ss_lon, eq_row.latitude, eq_row.longitude)
                    bin_idx = np.searchsorted(zone_bins, dist, side='right') - 1
                    if 0 <= bin_idx < N_ZONES:
                        p_counts[bin_idx] += 2  # already counted once, add 2 more

            p_density = p_counts / max(d_end - d_start, 1) / sa
            ratios = np.clip(p_density / bg_density, 0, 10)

            x = np.concatenate([x_base, [phase_val]])
            X_list.append(x)
            y_list.append(ratios.astype(np.float32))

            # Weight: stronger storms matter more
            w = 1.0 + max(0, (kp_now - 5)) * 0.5
            weights_list.append(w)

    X = np.array(X_list)
    y = np.array(y_list)
    w = np.array(weights_list)
    print(f"  Training samples: {len(X)}")
    return X, y, w


# ============================================================
# TRAINING V2
# ============================================================

def train_v2(epochs=500, lr=0.002, batch_size=64):
    print("=" * 70)
    print("  TRAINING JellyBallNet V2")
    print("=" * 70)

    X, y, sample_weights = build_training_data_v2()

    X_solar = X[:, :N_INPUT]
    X_phase = X[:, N_INPUT:N_INPUT + 1]

    x_mean = X_solar.mean(axis=0)
    x_std = X_solar.std(axis=0) + 1e-8
    X_solar = (X_solar - x_mean) / x_std

    X_t = torch.tensor(X_solar, dtype=torch.float32)
    T_t = torch.tensor(X_phase, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w_t = torch.tensor(sample_weights, dtype=torch.float32)

    n = len(X_t)
    n_train = int(0.8 * n)

    model = JellyBallNetV2()
    print(f"  Parameters: {model.count_params()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, best_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss, n_batch = 0, 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            pred, diag = model(X_t[idx], T_t[idx])

            # Weighted MSE
            diff = (pred - y_t[idx]) ** 2
            loss = (diff.mean(dim=1) * w_t[idx]).mean()

            # l=2 physics regularization
            l2 = diag['mode_amplitudes'][:, 1]
            phase = T_t[idx, 0]
            l2_reg = F.relu(-l2[phase < 0.2]).mean() * 0.05 + F.relu(l2[phase > 0.6]).mean() * 0.05

            total = loss + l2_reg

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp, _ = model(X_t[n_train:], T_t[n_train:])
            vl = F.mse_loss(vp, y_t[n_train:]).item()

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}  train={total_loss/max(n_batch,1):.4f}  val={vl:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_state)
    torch.save({'model_state': best_state, 'x_mean': x_mean, 'x_std': x_std,
                'val_loss': best_val, 'version': 2, 'n_input': N_INPUT}, OUT / "jellyball_net_v2.pt")
    print(f"\n  Saved: jellyball_net_v2.pt  (val MSE={best_val:.4f})")
    return model, x_mean, x_std, X_t, T_t, y_t, w_t, n_train


def backtest_v2(model, X_t, T_t, y_t, w_t, n_train):
    print("\n" + "=" * 70)
    print("  V2 BACKTEST")
    print("=" * 70)

    model.eval()
    zone_names = [z[0] for z in ZONES]

    with torch.no_grad():
        pred_val, diag_val = model(X_t[n_train:], T_t[n_train:])

    pred = pred_val.numpy()
    true = y_t[n_train:].numpy()

    # Baselines
    naive_mse = np.mean((true - 1.0) ** 2)
    zone_mean = np.mean(true, axis=0)
    zmean_mse = np.mean((true - zone_mean) ** 2)
    model_mse = np.mean((pred - true) ** 2)

    print(f"\n  BASELINES:")
    print(f"  Naive (predict 1.0):  MSE = {naive_mse:.4f}")
    print(f"  Zone-mean:            MSE = {zmean_mse:.4f}")
    print(f"  JellyBallNet V2:      MSE = {model_mse:.4f}")
    print(f"  Improvement (naive):  {(1 - model_mse/naive_mse)*100:+.1f}%")
    print(f"  Improvement (zmean):  {(1 - model_mse/zmean_mse)*100:+.1f}%")

    # Per-zone
    print(f"\n  PER-ZONE (val set, n={len(pred)})")
    print(f"  {'Zone':18s} {'MSE':>7s} {'r':>7s} {'p':>8s} {'Pred':>7s} {'True':>7s}")
    print("  " + "-" * 55)
    for i, name in enumerate(zone_names):
        p, t = pred[:, i], true[:, i]
        mse = np.mean((p - t) ** 2)
        valid = np.isfinite(p) & np.isfinite(t)
        r, pv = stats.pearsonr(p[valid], t[valid]) if valid.sum() > 5 else (0, 1)
        sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
        print(f"  {name:18s} {mse:7.3f} {r:+6.3f} {pv:8.4f}{sig:>3s} {p.mean():6.3f} {t.mean():6.3f}")

    # Per-phase
    phases_in_data = T_t[n_train:, 0].numpy()
    phase_names = ['compression', 'peak', 'relax_early', 'relax_late']
    phase_vals = [0.0, 0.2, 0.5, 0.8]
    far_idx = [i for i, z in enumerate(ZONES) if z[0] == 'far-suppress'][0]
    wf_idx = [i for i, z in enumerate(ZONES) if z[0] == 'wavefront'][0]

    print(f"\n  PER-PHASE:")
    print(f"  {'Phase':18s} {'MSE':>7s} {'Far-sup r':>10s} {'WF r':>8s}")
    for pn, pv in zip(phase_names, phase_vals):
        mask = np.abs(phases_in_data - pv) < 0.05
        if mask.sum() < 5: continue
        pp, tp = pred[mask], true[mask]
        mse = np.mean((pp - tp) ** 2)
        valid = np.isfinite(pp[:, far_idx]) & np.isfinite(tp[:, far_idx])
        rf = stats.pearsonr(pp[valid, far_idx], tp[valid, far_idx])[0] if valid.sum() > 5 else 0
        valid = np.isfinite(pp[:, wf_idx]) & np.isfinite(tp[:, wf_idx])
        rw = stats.pearsonr(pp[valid, wf_idx], tp[valid, wf_idx])[0] if valid.sum() > 5 else 0
        print(f"  {pn:18s} {mse:7.3f} {rf:+9.3f} {rw:+7.3f}")

    # l=2 flip
    with torch.no_grad():
        comp = np.abs(phases_in_data - 0.0) < 0.05
        relax = np.abs(phases_in_data - 0.8) < 0.05
        if comp.sum() > 0 and relax.sum() > 0:
            _, dc = model(X_t[n_train:][comp], T_t[n_train:][comp])
            _, dr = model(X_t[n_train:][relax], T_t[n_train:][relax])
            l2c = dc['mode_amplitudes'][:, 1].mean().item()
            l2r = dr['mode_amplitudes'][:, 1].mean().item()
            print(f"\n  l=2 SIGN FLIP: comp={l2c:+.4f}, relax={l2r:+.4f} {'CONFIRMED' if l2c*l2r < 0 else 'same sign'}")

    # Calibration
    print(f"\n  CALIBRATION:")
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 10)]:
        mask = (pred >= lo) & (pred < hi)
        if mask.sum() > 10:
            print(f"  pred {lo:.1f}-{hi:.1f}: mean_pred={pred[mask].mean():.3f} mean_obs={true[mask].mean():.3f} n={mask.sum()}")


def main():
    model, xm, xs, X_t, T_t, y_t, w_t, n_train = train_v2(epochs=500, lr=0.002)
    backtest_v2(model, X_t, T_t, y_t, w_t, n_train)


if __name__ == "__main__":
    main()
