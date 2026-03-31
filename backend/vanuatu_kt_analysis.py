#!/usr/bin/env python3
"""
Vanuatu KT Earthquake Analysis
================================
Adapted from Saldanha et al. (2025) "The role of solar heat in earthquake activity"
Extended with Kuramoto-KT framework predictions:
  - Fault lines as KT phase boundaries (J crossing J_c = 2/π)
  - Solar flare / CME coupling via ionospheric capacitor mechanism
  - Temperature modulation of crustal stiffness J(T)

Tests whether the solar-seismic coupling observed in Japan holds for
the Vanuatu subduction zone (Australian-Pacific plate boundary).

Data sources:
  - USGS ComCat earthquake catalog
  - SILSO daily sunspot numbers
  - GFZ Kp geomagnetic index
  - NOAA OISST sea surface temperature
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy import stats
from io import StringIO
from pathlib import Path
import datetime as dt
import requests
import sys
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── Configuration ───────────────────────────────────────────────────────────

# Vanuatu bounding box (expanded to capture the full subduction zone)
LAT_MIN, LAT_MAX = -22.0, -12.0
LON_MIN, LON_MAX = 164.0, 174.0

# Time range: 2000-01-01 to present
START_DATE = "2000-01-01"
END_DATE = "2026-03-31"

# Minimum magnitude for the catalog
# Use 4.5 to keep distance matrix tractable (N^2 computation)
MIN_MAG = 4.5

# Window and prediction sizes (days)
W = 7          # lookback window
PRED_WINDOW = 1  # prediction horizon

# Train/test split
TRAIN_FRACTION = 0.5

# Reference date for day numbering
INIT_DATE = dt.datetime(2000, 1, 1)

# Output directory
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ─── Data Download ───────────────────────────────────────────────────────────

def download_earthquakes():
    """Download Vanuatu region earthquakes from USGS ComCat."""
    print("Downloading earthquake catalog from USGS...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "csv",
        "starttime": START_DATE,
        "endtime": END_DATE,
        "minlatitude": LAT_MIN,
        "maxlatitude": LAT_MAX,
        "minlongitude": LON_MIN,
        "maxlongitude": LON_MAX,
        "minmagnitude": MIN_MAG,
        "orderby": "time-asc",
        "limit": 20000,
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    print(f"  Got {len(df)} earthquakes")
    return df


def download_sunspots():
    """Download daily sunspot numbers from SILSO."""
    print("Downloading sunspot data from SILSO...")
    url = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(
        StringIO(resp.text), sep=";", header=None,
        names=["year", "month", "day", "dec_year", "ssn", "std", "nobs", "definitive"],
        skipinitialspace=True
    )
    # Filter to our date range
    df = df[df["year"] >= 2000].copy()
    # ssn = -1 means no observation
    df.loc[df["ssn"] < 0, "ssn"] = np.nan
    df["ssn"] = df["ssn"].interpolate()
    print(f"  Got {len(df)} daily sunspot records")
    return df


def download_kp_index():
    """Download Kp geomagnetic index from GFZ Potsdam."""
    print("Downloading Kp index from GFZ...")
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    records = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 25:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            if year < 2000:
                continue
            # Format: YYYY MM DD days days_m Bsr dB Kp1..Kp8 ap1..ap8 Ap SN F10.7obs F10.7adj D
            # Kp values at indices 7-14, Ap at index 23
            kp_values = [float(parts[7 + i]) for i in range(8)]
            daily_kp = np.mean(kp_values)
            daily_ap = float(parts[23])
            sn = float(parts[24])  # sunspot number from GFZ
            f107 = float(parts[25])  # F10.7 solar radio flux (observed)
            records.append({
                "year": year, "month": month, "day": day,
                "kp_mean": daily_kp, "ap": daily_ap,
                "kp_max": max(kp_values),
                "sn": sn if sn >= 0 else np.nan,
                "f107": f107 if f107 >= 0 else np.nan,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    print(f"  Got {len(df)} daily Kp records")
    return df


# ─── Data Preprocessing ─────────────────────────────────────────────────────

def make_day_number(df, date_cols=("year", "month", "day")):
    """Convert year/month/day columns to days since INIT_DATE."""
    y, m, d = date_cols
    dates = pd.to_datetime(df[[y, m, d]].rename(columns={y: "year", m: "month", d: "day"}))
    return (dates - pd.Timestamp(INIT_DATE)).dt.days.values


def preprocess_earthquakes(raw):
    """Parse USGS CSV into standard format."""
    df = raw.copy()
    df["time_parsed"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["year"] = df["time_parsed"].dt.year
    df["month"] = df["time_parsed"].dt.month
    df["day"] = df["time_parsed"].dt.day
    df["hour"] = df["time_parsed"].dt.hour
    df["minute"] = df["time_parsed"].dt.minute
    df["magnitude"] = df["mag"]
    df["day_number"] = make_day_number(df)

    # Time in seconds since INIT_DATE
    df["time_seconds"] = (
        df["time_parsed"] - pd.Timestamp(INIT_DATE)
    ).dt.total_seconds()

    return df[["time_parsed", "year", "month", "day", "hour", "minute",
               "magnitude", "depth", "latitude", "longitude",
               "day_number", "time_seconds"]].copy()


# ─── Edit Distance Kernel (from Saldanha et al.) ────────────────────────────

def edit_distance(data1, data2, lambdas, lambda_deletion=1.0):
    """
    Compute edit distance between two earthquake sequences.
    Each row: (time_seconds, magnitude, longitude, latitude, depth)
    Uses Hungarian algorithm for optimal assignment.
    """
    if len(data1) == 0:
        return len(data2)
    if len(data2) == 0:
        return len(data1)

    n_total = len(data1) + len(data2)
    lambdas = np.asarray(lambdas)
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    M = np.zeros([n_total, n_total])
    M[:len(data1), :len(data1)] = lambda_deletion
    M[len(data1):, len(data1):] = lambda_deletion

    for i in range(len(data1)):
        diffs = np.abs(data2 - data1[i, :])
        costs = diffs @ lambdas
        M[i, len(data1):] = costs

    row_ind, col_ind = linear_sum_assignment(M)
    return M[row_ind, col_ind].sum()


# ─── Windowed Feature Construction ──────────────────────────────────────────

def build_windows(quake_df, sunspot_df, kp_df):
    """Build 7-day lookback windows with prediction targets."""
    print("Building windowed features...")

    quake_days = quake_df["day_number"].values
    ss_days = sunspot_df["day_number"].values
    kp_days = kp_df["day_number"].values

    max_day = int(quake_days.max())

    windows = {
        "quake_sequences": [],
        "max_mag": [],
        "mean_mag": [],
        "log_n": [],
        "sunspots": [],
        "kp": [],
        "f107": [],
        "quake_n": [],
        "day": [],
    }

    for i in range(W - 1, max_day + 1 - PRED_WINDOW):
        # Earthquake window
        mask_w = (quake_days > i - W) & (quake_days <= i)
        quake_window = quake_df[mask_w]

        # Prediction window
        mask_p = (quake_days > i) & (quake_days <= i + PRED_WINDOW)
        pred_window = quake_df[mask_p]

        # Sunspot window
        ss_mask = (ss_days > i - W) & (ss_days <= i)
        ss_window = sunspot_df[ss_mask]["ssn"].values

        # Kp window
        kp_mask = (kp_days > i - W) & (kp_days <= i)
        kp_window = kp_df[kp_mask]["kp_mean"].values

        # F10.7 window (solar radio flux — ionospheric driver)
        f107_window = kp_df[kp_mask]["f107"].values

        # Build earthquake sequence matrix
        if len(quake_window) > 0:
            seq = quake_window[["time_seconds", "magnitude", "longitude", "latitude", "depth"]].values.copy()
            seq[:, 0] -= (i - W + 1) * 86400  # relative time
        else:
            seq = np.array([])

        # Prediction targets
        if len(pred_window) > 0:
            windows["max_mag"].append(pred_window["magnitude"].max())
            windows["mean_mag"].append(pred_window["magnitude"].mean())
        else:
            windows["max_mag"].append(0)
            windows["mean_mag"].append(0)

        windows["quake_sequences"].append(seq)
        windows["log_n"].append(np.log(len(pred_window) + 1))
        windows["sunspots"].append(ss_window if len(ss_window) == W else
                                   np.pad(ss_window, (0, W - len(ss_window)), constant_values=np.nan))
        windows["kp"].append(kp_window if len(kp_window) == W else
                             np.pad(kp_window, (0, W - len(kp_window)), constant_values=np.nan))
        windows["f107"].append(f107_window if len(f107_window) == W else
                               np.pad(f107_window, (0, W - len(f107_window)), constant_values=np.nan))
        windows["quake_n"].append(len(quake_window))
        windows["day"].append(i)

    print(f"  Built {len(windows['day'])} windows")
    return windows


# ─── Distance Matrices ──────────────────────────────────────────────────────

# Module-level globals for multiprocessing (workers inherit these)
_mp_sequences = None
_mp_lambdas = None

def _init_worker(sequences, lambdas):
    global _mp_sequences, _mp_lambdas
    _mp_sequences = sequences
    _mp_lambdas = lambdas

def _compute_row(i):
    """Compute one row of the distance matrix (for multiprocessing)."""
    global _mp_sequences, _mp_lambdas
    N = len(_mp_sequences)
    row = []
    for j in range(i + 1, N):
        d = edit_distance(_mp_sequences[i], _mp_sequences[j], _mp_lambdas)
        row.append(d)
    if i % 100 == 0:
        print(f"  Row {i}/{N}", flush=True)
    return row

def compute_earthquake_distances(windows, quake_df):
    """Compute pairwise edit distances using multiprocessing."""
    import multiprocessing as mp

    sequences = windows["quake_sequences"]
    N = len(sequences)

    # Compute normalization from training data
    train_data = quake_df[quake_df["year"] < quake_df["year"].median()]
    time_std = np.std(np.diff(train_data["time_seconds"])) * 100
    mag_std = train_data["magnitude"].std()
    depth_std = train_data["depth"].std()
    lat_std = train_data["latitude"].std()
    lon_std = train_data["longitude"].std()

    lambdas = [1.0 / max(s, 1e-6) for s in [time_std, mag_std, lon_std, lat_std, depth_std]]

    n_workers = min(mp.cpu_count(), 20)
    print(f"Computing {N}x{N} earthquake distance matrix ({n_workers} workers)...")

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with mp.Pool(n_workers, initializer=_init_worker, initargs=(sequences, lambdas)) as pool:
        all_rows = pool.map(_compute_row, range(N), chunksize=4)

    dist_matrix = np.zeros([N, N])
    for i in range(N):
        dist_matrix[i, (i+1):] = all_rows[i]
        dist_matrix[(i+1):, i] = all_rows[i]

    return dist_matrix


def compute_vector_distances(data_list):
    """Compute pairwise Euclidean distances for a list of vectors."""
    arr = np.array([np.nan_to_num(x, nan=0.0) for x in data_list])
    N = len(arr)
    dist = np.zeros([N, N])
    for i in range(N):
        diff_sq = (arr[i] - arr) ** 2
        dist[i, :] = np.sqrt(np.sum(diff_sq, axis=1))
    return dist


# ─── RBF Prediction (from Saldanha et al.) ──────────────────────────────────

def rbf_predict(y, dist_mat, eps, train_size):
    """RBF kernel regression: train on first half, predict second half."""
    train_gram = np.exp(-dist_mat[:train_size, :] ** 2 / eps)
    train_gram = np.hstack([np.ones((train_gram.shape[0], 1)), train_gram])

    w = np.linalg.pinv(train_gram.T @ train_gram) @ (train_gram.T @ y[:train_size])

    test_gram = np.exp(-dist_mat[train_size:, :] ** 2 / eps)
    test_gram = np.hstack([np.ones((test_gram.shape[0], 1)), test_gram])

    predicted = test_gram @ w
    return predicted, y[train_size:]


def run_experiments(y, max_mags, dist_mat, eps, train_size, n_iter=100, label=""):
    """Run Monte Carlo experiments with random feature subsets."""
    correlations = []
    odds_ratios = []

    for i in range(n_iter):
        select_idx = np.random.choice(train_size, size=min(100, train_size), replace=False)
        select_dist = dist_mat[:, select_idx]

        predicted, real = rbf_predict(y, select_dist, eps, train_size)
        mags = np.array(max_mags)[train_size:]

        # Thresholds for Vanuatu (M6+ is significant here)
        a, b = 5.5, 2.5

        tp = np.sum((mags > a) & (predicted > b))
        fp = np.sum((mags <= a) & (predicted > b))
        fn = np.sum((mags > a) & (predicted <= b))
        tn = np.sum((mags <= a) & (predicted <= b))

        corr = np.corrcoef(predicted, mags)[0, 1] if len(predicted) > 1 else 0
        correlations.append(corr)

        denom = max(fp * fn, 1)
        odds_ratios.append((tp * tn) / denom)

    corr_arr = np.array(correlations)
    print(f"  {label}: correlation = {np.median(corr_arr):.4f} "
          f"(IQR: {np.percentile(corr_arr, 25):.4f}–{np.percentile(corr_arr, 75):.4f})")

    return {"correlation": correlations, "odds": odds_ratios}


# ─── KT Framework Analysis ──────────────────────────────────────────────────

def kt_analysis(quake_df, kp_df):
    """
    Test the KT framework prediction:
    CME/geomagnetic storms should produce elevated seismicity
    within 24-72 hours of Kp spike.

    The mechanism: ionospheric charge perturbation → crustal capacitor
    pressure → pushes J through J_c at fault boundaries.
    """
    print("\n═══ KT Framework: Geomagnetic-Seismic Coupling ═══")

    # Merge earthquake counts with Kp data
    quake_daily = quake_df.groupby("day_number").agg(
        n_quakes=("magnitude", "count"),
        max_mag=("magnitude", "max"),
        mean_mag=("magnitude", "mean"),
    ).reset_index()

    kp_daily = kp_df[["day_number", "kp_mean", "ap"]].copy()

    merged = pd.merge(kp_daily, quake_daily, on="day_number", how="left")
    merged["n_quakes"] = merged["n_quakes"].fillna(0)
    merged["max_mag"] = merged["max_mag"].fillna(0)

    # Define geomagnetic storm days (Kp >= 5 is G1 storm)
    storm_threshold = 5.0
    merged["is_storm"] = merged["kp_mean"] >= storm_threshold

    # Look at seismicity 0-3 days after storm
    for lag in [0, 1, 2, 3]:
        merged[f"quakes_lag{lag}"] = merged["n_quakes"].shift(-lag)
        merged[f"maxmag_lag{lag}"] = merged["max_mag"].shift(-lag)

    storm_days = merged[merged["is_storm"]]
    quiet_days = merged[~merged["is_storm"]]

    print(f"\nStorm days (Kp ≥ {storm_threshold}): {len(storm_days)}")
    print(f"Quiet days: {len(quiet_days)}")

    print("\nMean earthquake count by lag after geomagnetic storm:")
    print(f"  {'Lag':>4s}  {'Storm':>8s}  {'Quiet':>8s}  {'Ratio':>8s}  {'p-value':>8s}")
    for lag in [0, 1, 2, 3]:
        col = f"quakes_lag{lag}"
        storm_vals = storm_days[col].dropna()
        quiet_vals = quiet_days[col].dropna()

        if len(storm_vals) > 0 and len(quiet_vals) > 0:
            ratio = storm_vals.mean() / max(quiet_vals.mean(), 0.001)
            _, pval = stats.mannwhitneyu(storm_vals, quiet_vals, alternative="greater")
            print(f"  {lag:>4d}  {storm_vals.mean():>8.3f}  {quiet_vals.mean():>8.3f}  "
                  f"{ratio:>8.2f}×  {pval:>8.4f}")

    # Cross-correlation: Kp vs seismicity rate
    print("\nCross-correlation: Kp index vs daily earthquake count")
    kp_series = merged["kp_mean"].values
    eq_series = merged["n_quakes"].values

    for lag in range(-3, 8):
        if lag >= 0:
            corr = np.corrcoef(kp_series[:len(kp_series)-max(lag,1)],
                               eq_series[lag:lag+len(kp_series)-max(lag,1)])[0, 1]
        else:
            corr = np.corrcoef(kp_series[-lag:],
                               eq_series[:len(eq_series)+lag])[0, 1]
        marker = " ◄" if lag == 1 else ""
        print(f"  Lag {lag:+d} day: r = {corr:+.4f}{marker}")

    return merged


def march_2026_analysis(quake_df, kp_df):
    """
    Specific analysis of the March 2026 Vanuatu sequence:
    M6.1 (Mar 20) → M5.7 (Mar 21) → M4.1 (Mar 26) → M7.3 (Mar 30)
    Coincident with X1.5 flare on Mar 30.
    Framework prediction: aftershock pulse with CME arrival Mar 31.
    """
    print("\n═══ March 2026 Vanuatu Sequence Analysis ═══")

    march = quake_df[(quake_df["year"] == 2026) & (quake_df["month"] == 3)].copy()
    print(f"\nMarch 2026 earthquakes (M≥{MIN_MAG}): {len(march)}")

    if len(march) > 0:
        print("\nDate        Mag   Depth   Lat      Lon")
        for _, row in march.iterrows():
            print(f"{row['time_parsed'].strftime('%Y-%m-%d %H:%M')}  "
                  f"{row['magnitude']:5.1f}  {row['depth']:6.1f}  "
                  f"{row['latitude']:7.3f}  {row['longitude']:8.3f}")

    # Check Kp around the event
    kp_march = kp_df[(kp_df["year"] == 2026) & (kp_df["month"] == 3)]
    if len(kp_march) > 0:
        print("\nKp index around the M7.3 (Mar 28-31):")
        late_march = kp_march[kp_march["day"] >= 28]
        for _, row in late_march.iterrows():
            print(f"  Mar {int(row['day']):2d}: Kp = {row['kp_mean']:.1f}, Ap = {row['ap']:.0f}")

    # Omori law check: aftershocks in days following M7.3
    post_main = quake_df[
        (quake_df["time_parsed"] >= "2026-03-30") &
        (quake_df["time_parsed"] <= "2026-04-07")
    ]
    if len(post_main) > 0:
        print(f"\nPost-mainshock seismicity (Mar 30 – Apr 7): {len(post_main)} events")
        daily = post_main.groupby(post_main["time_parsed"].dt.date).agg(
            n=("magnitude", "count"),
            max_mag=("magnitude", "max")
        )
        print(daily.to_string())


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_results(experiments, labels, colors, filename="correlation_comparison.png"):
    """Box plot comparing correlation across models (Fig. from Saldanha)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    positions = np.arange(len(experiments)) * 0.25 + 0.2
    for i, (exp, label, color) in enumerate(zip(experiments, labels, colors)):
        bp = ax.boxplot(exp["correlation"], positions=[positions[i]],
                        patch_artist=True, widths=0.15,
                        boxprops={"facecolor": color, "alpha": 0.7},
                        medianprops={"color": "black", "linewidth": 2})

    ax.set_xlim([0, positions[-1] + 0.25])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, fontsize=11)
    ax.set_ylabel("Correlation (predicted vs observed max magnitude)", fontsize=12)
    ax.set_title("Vanuatu Earthquake Prediction: Solar-Seismic Coupling\n"
                 f"Edit-distance RBF kernel, {W}-day window, {PRED_WINDOW}-day prediction",
                 fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    print(f"\nSaved: {OUT_DIR / filename}")


def plot_kp_earthquake_timeline(merged, filename="kp_earthquake_timeline.png"):
    """Timeline of Kp index vs earthquake activity."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Last 2 years for visibility
    recent = merged[merged["day_number"] > merged["day_number"].max() - 730].copy()

    ax1.bar(recent["day_number"], recent["n_quakes"], width=1, alpha=0.7, color="steelblue")
    ax1.set_ylabel("Daily earthquake count (M≥4)")
    ax1.set_title("Vanuatu Region: Geomagnetic Activity vs Seismicity")

    ax2.fill_between(recent["day_number"], recent["kp_mean"], alpha=0.5, color="orange")
    ax2.axhline(y=5, color="red", linestyle="--", label="G1 storm threshold (Kp=5)")
    ax2.set_ylabel("Daily mean Kp index")
    ax2.set_xlabel("Days since 2000-01-01")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150)
    print(f"Saved: {OUT_DIR / filename}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VANUATU KT EARTHQUAKE ANALYSIS")
    print("Kuramoto-KT framework: fault lines as phase boundaries")
    print("Testing solar-seismic coupling via ionospheric capacitor mechanism")
    print("=" * 70)

    # Download data
    try:
        eq_raw = download_earthquakes()
        ss_raw = download_sunspots()
        kp_raw = download_kp_index()
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Check internet connection and try again.")
        sys.exit(1)

    # Preprocess
    quake_df = preprocess_earthquakes(eq_raw)
    quake_df = quake_df.sort_values("day_number").reset_index(drop=True)

    ss_raw["day_number"] = make_day_number(ss_raw)
    kp_raw["day_number"] = make_day_number(kp_raw)

    # Trim all datasets to common range
    max_day = min(quake_df["day_number"].max(), ss_raw["day_number"].max(),
                  kp_raw["day_number"].max())
    quake_df = quake_df[quake_df["day_number"] <= max_day]
    ss_df = ss_raw[ss_raw["day_number"] <= max_day]
    kp_df = kp_raw[kp_raw["day_number"] <= max_day]

    print(f"\nData range: day 0 to {max_day} ({max_day / 365.25:.1f} years)")
    print(f"Earthquakes: {len(quake_df)}")
    print(f"Sunspot records: {len(ss_df)}")
    print(f"Kp records: {len(kp_df)}")

    # ─── KT Framework Tests ─────────────────────────────────────────────
    merged = kt_analysis(quake_df, kp_df)
    march_2026_analysis(quake_df, kp_df)

    # ─── Saldanha-style Edit Distance Analysis ──────────────────────────
    print("\n═══ Edit Distance Kernel Analysis (Saldanha et al. adapted) ═══")

    windows_full = build_windows(quake_df, ss_df, kp_df)

    # Subsample every STEP windows to keep distance matrix tractable
    # ~9500 windows → ~3200 at step=3
    STEP = 3
    N_full = len(windows_full["day"])
    idx = list(range(0, N_full, STEP))
    windows = {}
    for key in windows_full:
        if isinstance(windows_full[key], list):
            windows[key] = [windows_full[key][i] for i in idx]
        else:
            windows[key] = windows_full[key]

    N = len(windows["day"])
    train_size = round(TRAIN_FRACTION * N)
    print(f"  Subsampled {N_full} -> {N} windows (step={STEP})")

    y = np.array(windows["log_n"])
    max_mags = windows["max_mag"]

    # Compute distance matrices
    eq_dist = compute_earthquake_distances(windows, quake_df)
    ss_dist = compute_vector_distances(windows["sunspots"])
    kp_dist = compute_vector_distances(windows["kp"])

    # F10.7 distance matrix (solar radio flux — direct ionospheric driver)
    f107_dist = compute_vector_distances(windows["f107"])

    # Mixing coefficients (relative scale normalization)
    k_ss = np.mean(eq_dist) / max(np.mean(ss_dist), 1e-6) / 2
    k_kp = np.mean(eq_dist) / max(np.mean(kp_dist), 1e-6) / 2
    k_f107 = np.mean(eq_dist) / max(np.mean(f107_dist), 1e-6) / 2

    print(f"\nMixing coefficients: k_sunspot = {k_ss:.4f}, k_kp = {k_kp:.4f}, k_f107 = {k_f107:.4f}")

    # Experiment 1: Baseline (earthquakes only)
    train_mat = eq_dist[:train_size, :]
    eps1 = 2 * np.mean(train_mat[train_mat > 0]) ** 2
    exp1 = run_experiments(y, max_mags, eq_dist, eps1, train_size, n_iter=100,
                           label="Baseline (quakes only)")

    # Experiment 2: + Sunspots
    dist2 = eq_dist + k_ss * ss_dist
    eps2 = eps1 + 2 * (k_ss * np.mean(ss_dist[:train_size][ss_dist[:train_size] > 0])) ** 2
    exp2 = run_experiments(y, max_mags, dist2, eps2, train_size, n_iter=100,
                           label="+ Sunspots")

    # Experiment 3: + Kp geomagnetic index
    dist3 = eq_dist + k_kp * kp_dist
    eps3 = eps1 + 2 * (k_kp * np.mean(kp_dist[:train_size][kp_dist[:train_size] > 0])) ** 2
    exp3 = run_experiments(y, max_mags, dist3, eps3, train_size, n_iter=100,
                           label="+ Kp index")

    # Experiment 4: + F10.7 solar radio flux (ionospheric TEC driver)
    dist4 = eq_dist + k_f107 * f107_dist
    eps4 = eps1 + 2 * (k_f107 * np.mean(f107_dist[:train_size][f107_dist[:train_size] > 0])) ** 2
    exp4 = run_experiments(y, max_mags, dist4, eps4, train_size, n_iter=100,
                           label="+ F10.7 flux")

    # Experiment 5: + Kp & F10.7 (full EM coupling)
    dist5 = eq_dist + k_kp * kp_dist + k_f107 * f107_dist
    eps5 = (eps1 +
            2 * (k_kp * np.mean(kp_dist[:train_size][kp_dist[:train_size] > 0])) ** 2 +
            2 * (k_f107 * np.mean(f107_dist[:train_size][f107_dist[:train_size] > 0])) ** 2)
    exp5 = run_experiments(y, max_mags, dist5, eps5, train_size, n_iter=100,
                           label="+ Kp & F10.7")

    # Experiment 6: Everything (sunspots + Kp + F10.7)
    dist6 = eq_dist + k_ss * ss_dist + k_kp * kp_dist + k_f107 * f107_dist
    eps6 = (eps1 +
            2 * (k_ss * np.mean(ss_dist[:train_size][ss_dist[:train_size] > 0])) ** 2 +
            2 * (k_kp * np.mean(kp_dist[:train_size][kp_dist[:train_size] > 0])) ** 2 +
            2 * (k_f107 * np.mean(f107_dist[:train_size][f107_dist[:train_size] > 0])) ** 2)
    exp6 = run_experiments(y, max_mags, dist6, eps6, train_size, n_iter=100,
                           label="+ All solar")

    # Plot
    plot_results(
        [exp1, exp2, exp3, exp4, exp5, exp6],
        ["Baseline\n(quakes)", "+ Sunspots", "+ Kp\n(geomag)",
         "+ F10.7\n(ionosph)", "+ Kp &\nF10.7", "+ All\nsolar"],
        ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
    )

    plot_kp_earthquake_timeline(merged)

    # ─── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Framework prediction: fault lines are KT phase boundaries.
Solar forcing (heat + EM) modulates crustal stiffness J(T).
CME/geomagnetic storms deliver ionospheric charge impulse
that pushes J through J_c = 2/pi at critically stressed faults.

Correlation improvement (median):
  Baseline:              {np.median(exp1['correlation']):.4f}
  + Sunspots:            {np.median(exp2['correlation']):.4f}
  + Kp (geomagnetic):    {np.median(exp3['correlation']):.4f}
  + F10.7 (ionospheric): {np.median(exp4['correlation']):.4f}
  + Kp & F10.7:          {np.median(exp5['correlation']):.4f}
  + All solar:           {np.median(exp6['correlation']):.4f}

Key test: If F10.7 (ionospheric TEC driver) improves more than
sunspots (luminosity proxy), the coupling is electromagnetic
(ionospheric capacitor) not thermal. If Kp (storm index)
improves further, the impulsive CME mechanism dominates.
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
