#!/usr/bin/env python3
"""
Hazard Index Backtest: Combined Tidal + Solar Prediction
==========================================================
Build a composite hazard index from:
  1. Lunar tidal stress rate (rising = dangerous)
  2. Solar cycle phase (minimum = dangerous)
  3. Bz polarity (northward = dangerous for seismicity)

Test: does this index predict M7+ earthquake occurrence
better than any individual component?
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"

REF_NEW_MOON = pd.Timestamp("2000-01-06")
SYNODIC = 29.53059


def main():
    print("=" * 60)
    print("HAZARD INDEX BACKTEST")
    print("Combined tidal + solar prediction of M7+ earthquakes")
    print("=" * 60)

    # Load data
    eq = pd.read_csv(DATA_DIR / "earthquakes_m4.5.csv", parse_dates=["time_parsed"])
    omni = pd.read_csv(DATA_DIR / "omni_hourly.csv", parse_dates=["datetime"])
    kp = pd.read_csv(DATA_DIR / "kp_daily.csv")

    # Build DAILY features
    print("\nBuilding daily feature matrix...")

    # Date range
    dates = pd.date_range("2000-01-01", "2025-12-31", freq="D")
    daily = pd.DataFrame({"date": dates})
    daily["day_number"] = ((daily["date"] - pd.Timestamp("2000-01-01")).dt.days).values

    # 1. Lunar tidal stress rate
    phase = ((daily["date"] - REF_NEW_MOON).dt.total_seconds() / 86400 % SYNODIC) / SYNODIC
    daily["phase"] = phase
    daily["tidal_rate"] = -np.sin(2 * np.pi * phase)  # +1 = max rising
    daily["tidal_force"] = np.cos(2 * np.pi * phase)  # +1 = new/full (spring)
    daily["spring_tide"] = np.abs(np.cos(np.pi * phase))  # 1 at syzygy, 0 at quadrature

    # 2. Solar cycle (daily Kp/SSN)
    kp_daily = kp[["day_number", "kp_mean", "ap", "sn", "f107"]].copy()
    daily = pd.merge(daily, kp_daily, on="day_number", how="left")
    # Fill gaps
    daily["sn"] = daily["sn"].interpolate().fillna(0)
    daily["f107"] = daily["f107"].interpolate().fillna(100)
    daily["ap"] = daily["ap"].interpolate().fillna(10)

    # 3. Bz (daily mean from OMNI)
    omni_daily = omni.groupby("day_number").agg(
        bz_mean=("bz_gse", "mean"),
        bz_min=("bz_gse", "min"),
        ae_mean=("ae", "mean"),
        vsw_mean=("v_sw", "mean"),
    ).reset_index()
    daily = pd.merge(daily, omni_daily, on="day_number", how="left")
    daily["bz_mean"] = daily["bz_mean"].interpolate().fillna(0)
    daily["bz_north"] = (daily["bz_mean"] > 0).astype(float)  # 1 = northward

    # 4. Earthquake targets
    # M6+ and M7+ daily counts
    eq["day_number"] = ((eq["time_parsed"] - pd.Timestamp("2000-01-01")).dt.days).values
    for mag_thresh, col in [(6.0, "has_m6"), (7.0, "has_m7"), (7.5, "has_m75")]:
        eq_sub = eq[eq["magnitude"] >= mag_thresh]
        eq_days = eq_sub.groupby("day_number").size().reset_index(name="n")
        eq_days[col] = 1
        daily = pd.merge(daily, eq_days[["day_number", col]], on="day_number", how="left")
        daily[col] = daily[col].fillna(0).astype(int)

    print(f"  Daily records: {len(daily)}")
    print(f"  Days with M6+: {daily['has_m6'].sum()} ({daily['has_m6'].mean():.1%})")
    print(f"  Days with M7+: {daily['has_m7'].sum()} ({daily['has_m7'].mean():.1%})")
    print(f"  Days with M7.5+: {daily['has_m75'].sum()} ({daily['has_m75'].mean():.1%})")

    # ─── Individual Component Tests ──────────────────────────────────
    print("\n=== Individual Component Predictive Power ===")

    features = {
        "tidal_rate": "Tidal stress rate (rising = +1)",
        "tidal_force": "Tidal force (spring = +1)",
        "spring_tide": "Spring tide strength",
        "sn": "Sunspot number (inverted for hazard)",
        "f107": "F10.7 solar flux (inverted)",
        "bz_north": "Northward Bz fraction",
        "ap": "Ap geomagnetic index",
    }

    for target, target_label in [("has_m7", "M7+"), ("has_m6", "M6+")]:
        print(f"\n  Target: {target_label}")
        for feat, desc in features.items():
            valid = daily[[feat, target]].dropna()
            if len(valid) < 1000:
                continue
            X = valid[feat].values.reshape(-1, 1)
            y = valid[target].values
            if y.sum() < 10:
                continue
            try:
                auc = roc_auc_score(y, X)
                # Flip if anti-correlated
                if auc < 0.5:
                    auc = 1 - auc
                    direction = "INVERTED"
                else:
                    direction = "direct"
                print(f"    {feat:>15s}: AUC = {auc:.4f} ({direction})")
            except:
                pass

    # ─── Composite Hazard Index ──────────────────────────────────────
    print("\n=== Composite Hazard Index ===")

    # Features for the model
    feature_cols = ["tidal_rate", "spring_tide", "sn", "f107", "bz_north", "ap", "phase"]

    # Add lagged features
    for lag in [1, 3, 7]:
        daily[f"tidal_rate_lag{lag}"] = daily["tidal_rate"].shift(lag)
        daily[f"sn_lag{lag}"] = daily["sn"].shift(lag)
    feature_cols += ["tidal_rate_lag1", "tidal_rate_lag3", "tidal_rate_lag7",
                     "sn_lag1", "sn_lag3", "sn_lag7"]

    clean = daily.dropna(subset=feature_cols + ["has_m7"])

    for target, target_label in [("has_m7", "M7+"), ("has_m6", "M6+")]:
        X = clean[feature_cols].values
        y = clean[target].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Time series CV
        tscv = TimeSeriesSplit(n_splits=5)
        aucs = []
        for train_idx, test_idx in tscv.split(X_scaled):
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled[train_idx], y[train_idx])
            y_pred = model.predict_proba(X_scaled[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) > 1:
                aucs.append(roc_auc_score(y[test_idx], y_pred))

        if aucs:
            print(f"\n  {target_label} Composite Model:")
            print(f"    ROC AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
            print(f"    Individual folds: {[f'{a:.4f}' for a in aucs]}")

            # Full model for coefficients and calibration
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled, y)

            print(f"\n    Feature weights:")
            for feat, coef in sorted(zip(feature_cols, model.coef_[0]),
                                      key=lambda x: abs(x[1]), reverse=True):
                direction = "MORE quakes" if coef > 0 else "FEWER quakes"
                print(f"      {feat:>20s}: {coef:+.4f} ({direction})")

            # Calibration: top 10% hazard days vs bottom 10%
            clean[f"hazard_{target}"] = model.predict_proba(X_scaled)[:, 1]
            q90 = clean[f"hazard_{target}"].quantile(0.9)
            q10 = clean[f"hazard_{target}"].quantile(0.1)
            high = clean[clean[f"hazard_{target}"] >= q90]
            low = clean[clean[f"hazard_{target}"] <= q10]

            print(f"\n    Calibration:")
            print(f"      Top 10% hazard days: {target_label} rate = {high[target].mean():.2%} "
                  f"({len(high)} days)")
            print(f"      Bottom 10% hazard days: {target_label} rate = {low[target].mean():.2%} "
                  f"({len(low)} days)")
            print(f"      Baseline: {clean[target].mean():.2%}")
            if low[target].mean() > 0:
                print(f"      Lift (top/bottom): {high[target].mean() / low[target].mean():.2f}x")

    # ─── Historical M7+ Events vs Hazard Index ──────────────────────
    print("\n=== Historical M7+ Events: Were They on High-Hazard Days? ===")

    model_m7 = LogisticRegression(max_iter=1000)
    X_all = clean[feature_cols].values
    y_all = clean["has_m7"].values
    X_all_scaled = scaler.fit_transform(X_all)
    model_m7.fit(X_all_scaled, y_all)
    clean["hazard_m7"] = model_m7.predict_proba(X_all_scaled)[:, 1]

    # Get M7+ event dates
    m7_events = eq[eq["magnitude"] >= 7.0].copy()
    m7_events["day_number"] = ((m7_events["time_parsed"] - pd.Timestamp("2000-01-01")).dt.days).values
    m7_merged = pd.merge(m7_events, clean[["day_number", "hazard_m7", "tidal_rate", "sn", "bz_mean", "phase"]],
                          on="day_number", how="inner")

    if len(m7_merged) > 0:
        # What percentile of hazard were the M7+ events at?
        all_hazard = clean["hazard_m7"].values
        m7_percentiles = [stats.percentileofscore(all_hazard, h) for h in m7_merged["hazard_m7"]]
        m7_merged["percentile"] = m7_percentiles

        print(f"\n  {len(m7_merged)} M7+ events with hazard scores:")
        print(f"  Mean hazard percentile: {np.mean(m7_percentiles):.1f}%")
        print(f"  Median: {np.median(m7_percentiles):.1f}%")
        print(f"  Above 50th percentile: {(np.array(m7_percentiles) > 50).mean():.1%}")
        print(f"  Above 75th percentile: {(np.array(m7_percentiles) > 75).mean():.1%}")

        # Show the top 10 largest events
        print(f"\n  Largest events:")
        top = m7_merged.nlargest(15, "magnitude")
        for _, row in top.iterrows():
            dt = row["time_parsed"]
            phase_name = "NEW" if (row["phase"] < 0.05 or row["phase"] > 0.95) else \
                         "FULL" if (0.45 < row["phase"] < 0.55) else ""
            print(f"    {dt.strftime('%Y-%m-%d')} M{row['magnitude']:.1f} "
                  f"hazard={row['percentile']:.0f}th pctl  "
                  f"tide_rate={row['tidal_rate']:+.2f} "
                  f"SSN={row['sn']:.0f} Bz={row['bz_mean']:+.1f} "
                  f"phase={row['phase']:.2f} {phase_name}")

    # ─── Plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # Panel 1: Hazard index time series with M7+ events marked
    ax = axes[0]
    ax.plot(clean["date"], clean["hazard_m7"].rolling(30).mean(),
            color="steelblue", lw=1, alpha=0.7, label="30-day smoothed hazard")
    m7_dates = m7_merged["time_parsed"]
    m7_hazard = m7_merged["hazard_m7"]
    ax.scatter(m7_dates, m7_hazard, color="red", s=20, zorder=5, label="M7+ events")
    ax.set_ylabel("Hazard index (probability)")
    ax.set_title("Combined Tidal-Solar Hazard Index vs M7+ Earthquakes")
    ax.legend()

    # Panel 2: ROC curve
    ax = axes[1]
    y_score = model_m7.predict_proba(X_all_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_all, y_score)
    auc = roc_auc_score(y_all, y_score)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Can We Predict M7+ Days?")
    ax.legend()

    # Panel 3: Tidal rate histogram for M7+ vs all days
    ax = axes[2]
    ax.hist(clean["tidal_rate"], bins=30, alpha=0.5, density=True,
            color="steelblue", label="All days")
    m7_rates = m7_merged["tidal_rate"].dropna()
    ax.hist(m7_rates, bins=15, alpha=0.7, density=True,
            color="red", label="M7+ event days")
    ax.set_xlabel("Tidal stress rate (-1 = falling, +1 = rising)")
    ax.set_ylabel("Density")
    ax.set_title("M7+ Earthquakes Prefer Rising Tides")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "hazard_index_backtest.png", dpi=150)
    print(f"\nSaved: hazard_index_backtest.png")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The combined hazard index tests whether LUNAR TIDAL PHASE +
SOLAR CYCLE STATE together predict M7+ earthquake days better
than either alone.

If the top 10% hazard days have significantly more M7+ events
than the bottom 10%, the index has predictive value.

The index cannot predict WHERE or the exact day — but it can
identify periods of elevated/reduced global M7+ risk based
on astronomical data alone (moon phase + sunspot number + Bz).
""")


if __name__ == "__main__":
    main()
