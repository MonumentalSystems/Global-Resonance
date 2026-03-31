#!/usr/bin/env python3
"""
Predictive Models: Can we forecast from solar data?
=====================================================
Test whether the framework produces PREDICTIONS, not just correlations.

1. Off-season tornado prediction from SSN + Kp
2. M7+ earthquake probability from solar cycle phase
3. STEVE occurrence from Bz reversal rate
4. Seismic rate anomaly from CME arrival timing
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"


def load_all():
    eq = pd.read_csv(DATA_DIR / "earthquakes_m4.5.csv", parse_dates=["time_parsed"])
    kp = pd.read_csv(DATA_DIR / "kp_daily.csv")
    omni = pd.read_csv(DATA_DIR / "omni_hourly.csv", parse_dates=["datetime"])
    torn = pd.read_csv(DATA_DIR / "tornadoes_1950_2023.csv", low_memory=False)
    flares = pd.read_csv(DATA_DIR / "solar_flares.csv", parse_dates=["beginTime", "peakTime", "endTime"])
    return eq, kp, omni, torn, flares


# ═══════════════════════════════════════════════════════════════════════
# 1. OFF-SEASON TORNADO PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def tornado_prediction(torn, kp):
    print("\n=== Model 1: Off-Season Tornado Prediction ===")
    print("Can we predict monthly off-season tornado rate from solar data?")

    # Monthly tornado counts (off-season: Jul-Mar)
    torn["yr"] = torn["yr"].astype(int)
    torn["mo"] = torn["mo"].astype(int)
    torn_monthly = torn.groupby(["yr", "mo"]).size().reset_index(name="n_torn")

    # Off-season only
    offseason = torn_monthly[~torn_monthly["mo"].isin([4, 5, 6])].copy()

    # Monthly solar data
    kp_monthly = kp.groupby(["year", "month"]).agg(
        sn=("sn", lambda x: x[x >= 0].mean()),
        ap=("ap", "mean"),
        kp_mean=("kp_mean", "mean"),
        f107=("f107", lambda x: x[x > 0].mean()),
    ).reset_index()

    merged = pd.merge(offseason, kp_monthly,
                       left_on=["yr", "mo"], right_on=["year", "month"], how="inner")

    # Also add 1-month lag (does LAST month's solar data predict THIS month's tornadoes?)
    kp_monthly["year_next"] = kp_monthly["year"]
    kp_monthly["month_next"] = kp_monthly["month"] + 1
    kp_monthly.loc[kp_monthly["month_next"] > 12, "year_next"] += 1
    kp_monthly.loc[kp_monthly["month_next"] > 12, "month_next"] = 1
    kp_lag = kp_monthly[["year_next", "month_next", "sn", "ap", "f107"]].copy()
    kp_lag.columns = ["yr", "mo", "sn_lag1", "ap_lag1", "f107_lag1"]
    merged = pd.merge(merged, kp_lag, on=["yr", "mo"], how="left")

    # Binary target: high tornado month (above median)
    median_n = merged["n_torn"].median()
    merged["high_tornado"] = (merged["n_torn"] > median_n).astype(int)

    print(f"  Off-season months: {len(merged)}")
    print(f"  Median tornado count: {median_n:.0f}")
    print(f"  High tornado months: {merged['high_tornado'].sum()} ({merged['high_tornado'].mean():.1%})")

    # Features
    feature_cols = ["sn", "ap", "f107", "sn_lag1", "ap_lag1", "f107_lag1", "mo"]
    X = merged[feature_cols].fillna(0).values
    y = merged["high_tornado"].values

    # Time series cross-validation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="roc_auc")
    print(f"\n  Logistic Regression (5-fold time series CV):")
    print(f"    ROC AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"    Individual folds: {[f'{s:.3f}' for s in scores]}")

    # Train on all data and show coefficients
    model.fit(X_scaled, y)
    print(f"\n  Feature importance (logistic regression coefficients):")
    for feat, coef in sorted(zip(feature_cols, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        direction = "MORE tornadoes" if coef > 0 else "FEWER tornadoes"
        print(f"    {feat:>12s}: {coef:+.3f} ({direction})")

    # Continuous prediction: predicted probability vs actual count
    merged["pred_prob"] = model.predict_proba(X_scaled)[:, 1]
    r, p = stats.pearsonr(merged["pred_prob"], merged["n_torn"])
    print(f"\n  Predicted probability vs actual count: r = {r:+.3f}, p = {p:.4f}")

    return merged, model, scaler, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# 2. M7+ EARTHQUAKE PROBABILITY
# ═══════════════════════════════════════════════════════════════════════

def earthquake_prediction(eq, kp):
    print("\n=== Model 2: M7+ Earthquake Monthly Probability ===")

    # Monthly M7+ count
    eq_m7 = eq[eq["magnitude"] >= 7.0].copy()
    eq_m7["ym"] = eq_m7["time_parsed"].dt.to_period("M")
    monthly_m7 = eq_m7.groupby("ym").size().reset_index(name="n_m7")
    monthly_m7["year"] = monthly_m7["ym"].dt.year
    monthly_m7["month"] = monthly_m7["ym"].dt.month

    # All months (including zeros)
    all_months = pd.DataFrame({"year": np.repeat(range(2000, 2026), 12),
                                "month": np.tile(range(1, 13), 26)})
    monthly = pd.merge(all_months, monthly_m7, on=["year", "month"], how="left")
    monthly["n_m7"] = monthly["n_m7"].fillna(0)
    # Most months have at least 1 M7+. Use "above median" as target.
    med = monthly["n_m7"].median()
    monthly["has_m7"] = (monthly["n_m7"] > med).astype(int)

    # Solar features
    kp_monthly = kp.groupby(["year", "month"]).agg(
        sn=("sn", lambda x: x[x >= 0].mean()),
        ap=("ap", "mean"),
        kp_mean=("kp_mean", "mean"),
        f107=("f107", lambda x: x[x > 0].mean()),
    ).reset_index()

    merged = pd.merge(monthly, kp_monthly, on=["year", "month"], how="inner")

    # Add lagged features
    for lag in [1, 2, 3]:
        merged[f"sn_lag{lag}"] = merged["sn"].shift(lag)
        merged[f"ap_lag{lag}"] = merged["ap"].shift(lag)

    merged = merged.dropna()

    print(f"  Months: {len(merged)}")
    print(f"  Months with M7+: {merged['has_m7'].sum()} ({merged['has_m7'].mean():.1%})")

    feature_cols = ["sn", "ap", "f107", "sn_lag1", "ap_lag1", "sn_lag2", "sn_lag3", "month"]
    X = merged[feature_cols].fillna(0).values
    y = merged["has_m7"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="roc_auc")
    print(f"\n  ROC AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

    model.fit(X_scaled, y)
    print(f"\n  Feature importance:")
    for feat, coef in sorted(zip(feature_cols, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:>12s}: {coef:+.3f}")

    merged["pred_prob"] = model.predict_proba(X_scaled)[:, 1]

    # Calibration: in months where model says >60% chance, how often does M7+ actually occur?
    high_prob = merged[merged["pred_prob"] > 0.6]
    low_prob = merged[merged["pred_prob"] < 0.4]
    print(f"\n  Calibration:")
    print(f"    Model says >60%: actual M7+ rate = {high_prob['has_m7'].mean():.1%} (N={len(high_prob)})")
    print(f"    Model says <40%: actual M7+ rate = {low_prob['has_m7'].mean():.1%} (N={len(low_prob)})")

    return merged


# ═══════════════════════════════════════════════════════════════════════
# 3. NEXT-WEEK SEISMICITY FROM CME
# ═══════════════════════════════════════════════════════════════════════

def cme_seismic_forecast(eq, omni, flares):
    print("\n=== Model 3: Post-Flare Seismicity Forecast ===")
    print("Given an X-class flare, predict seismicity anomaly in next 72h")

    m5_flares = flares[flares["class_numeric"] >= 0.5].dropna(subset=["peakTime"]).copy()
    eq_times = eq["time_parsed"].values.astype("datetime64[h]")

    def count_window(ft, h_start, h_end):
        t0 = np.datetime64(ft, "h")
        return int(np.sum((eq_times >= t0 + np.timedelta64(h_start, "h")) &
                          (eq_times < t0 + np.timedelta64(h_end, "h"))))

    records = []
    for _, fl in m5_flares.iterrows():
        ft = fl["peakTime"]
        # Get OMNI context at flare time
        window = omni[(omni["datetime"] >= ft - pd.Timedelta(hours=6)) &
                       (omni["datetime"] <= ft + pd.Timedelta(hours=1))]
        if len(window) == 0:
            continue

        bz = window["bz_gse"].mean()
        ae = window["ae"].mean()
        vsw = window["v_sw"].mean()

        n_post = count_window(ft, 0, 72)
        n_bg = count_window(ft, -168, -24)
        bg_rate = n_bg / (144 / 72)

        ratio = n_post / max(bg_rate, 1)
        anomaly = 1 if ratio > 1.1 else 0  # >10% above background = enhanced

        records.append({
            "flare_time": ft, "class_num": fl["class_numeric"],
            "bz": bz, "ae": ae, "vsw": vsw,
            "n_post": n_post, "bg_rate": bg_rate, "ratio": ratio, "anomaly": anomaly,
        })

    rdf = pd.DataFrame(records)
    print(f"  Flares with complete data: {len(rdf)}")
    print(f"  Enhanced seismicity events: {rdf['anomaly'].sum()} ({rdf['anomaly'].mean():.1%})")

    # Can we predict which flares produce seismic enhancement?
    feature_cols = ["class_num", "bz", "ae", "vsw"]
    X = rdf[feature_cols].fillna(0).values
    y = rdf["anomaly"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="roc_auc")
    print(f"\n  ROC AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

    model.fit(X_scaled, y)
    print(f"\n  Feature importance:")
    for feat, coef in sorted(zip(feature_cols, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:>12s}: {coef:+.3f}")

    # The key question: does Bz predict the outcome?
    south = rdf[rdf["bz"] < 0]
    north = rdf[rdf["bz"] >= 0]
    print(f"\n  Bz split:")
    print(f"    Southward: {south['anomaly'].mean():.1%} enhanced ({len(south)} flares)")
    print(f"    Northward: {north['anomaly'].mean():.1%} enhanced ({len(north)} flares)")

    return rdf


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PREDICTIVE MODELS: From Correlation to Forecast")
    print("=" * 60)

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("Need: pip install scikit-learn")
        return

    eq, kp, omni, torn, flares = load_all()

    torn_result, torn_model, torn_scaler, torn_features = tornado_prediction(torn, kp)
    eq_result = earthquake_prediction(eq, kp)
    cme_result = cme_seismic_forecast(eq, omni, flares)

    print("\n" + "=" * 60)
    print("PREDICTIVE CAPACITY SUMMARY")
    print("=" * 60)
    print("""
The framework moves from correlation to prediction when:
  ROC AUC > 0.55: barely above chance
  ROC AUC > 0.60: weak but real predictive skill
  ROC AUC > 0.65: moderate skill (useful for alerts)
  ROC AUC > 0.70: good skill (operational potential)

Key question: can solar data ALONE improve over climatological
base rate? If yes, the coupling is real and predictive.
If no, the correlations may be real but too weak for forecasting.
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
