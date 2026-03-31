#!/usr/bin/env python3
"""
Local Analysis — Cache data, then run detailed studies
========================================================
Downloads once, saves to CSV, then runs all analyses locally.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, signal
from io import StringIO
from pathlib import Path
import datetime as dt
import requests
import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
INIT_DATE = dt.datetime(2000, 1, 1)


# ─── Cached Data ─────────────────────────────────────────────────────────────

def get_earthquakes(min_mag=4.5):
    cache = DATA_DIR / f"earthquakes_m{min_mag}.csv"
    if cache.exists():
        print(f"Loading cached {cache.name}...")
        df = pd.read_csv(cache, parse_dates=["time_parsed"])
        print(f"  {len(df)} events")
        return df

    print(f"Downloading global M>={min_mag} earthquakes (will cache)...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_dfs = []
    for year in range(2000, 2027):
        try:
            resp = requests.get(url, params={
                "format": "csv", "starttime": f"{year}-01-01",
                "endtime": f"{year}-12-31", "minmagnitude": min_mag,
                "orderby": "time-asc", "limit": 20000,
            }, timeout=120)
            resp.raise_for_status()
            all_dfs.append(pd.read_csv(StringIO(resp.text)))
            print(f"  {year}: {len(all_dfs[-1])} events")
        except Exception as e:
            print(f"  {year}: failed ({e})")

    df = pd.concat(all_dfs, ignore_index=True)
    df["time_parsed"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["magnitude"] = df["mag"]
    df["day_number"] = ((df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    df[["time_parsed", "magnitude", "depth", "latitude", "longitude", "day_number"]].to_csv(cache, index=False)
    print(f"  Cached {len(df)} events to {cache}")
    return df


def get_kp_daily():
    cache = DATA_DIR / "kp_daily.csv"
    if cache.exists():
        print(f"Loading cached {cache.name}...")
        df = pd.read_csv(cache)
        print(f"  {len(df)} days")
        return df

    print("Downloading Kp (will cache)...")
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url, timeout=60)
    records = []
    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        if len(parts) < 26: continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 2000: continue
            kp_vals = [float(parts[7+i]) for i in range(8)]
            records.append({
                "year": y, "month": m, "day": d,
                "kp_mean": np.mean(kp_vals), "kp_max": max(kp_vals),
                "ap": float(parts[23]), "sn": float(parts[24]),
                "f107": float(parts[25]),
            })
        except: continue
    df = pd.DataFrame(records)
    df["day_number"] = ((pd.to_datetime(df[["year","month","day"]]) - pd.Timestamp(INIT_DATE)).dt.days).values
    df.to_csv(cache, index=False)
    print(f"  Cached {len(df)} days to {cache}")
    return df


def get_sunspots_daily():
    cache = DATA_DIR / "sunspots_daily.csv"
    if cache.exists():
        print(f"Loading cached {cache.name}...")
        return pd.read_csv(cache)

    print("Downloading daily sunspots (will cache)...")
    url = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
    resp = requests.get(url, timeout=60)
    df = pd.read_csv(StringIO(resp.text), sep=";", header=None,
                     names=["year","month","day","dec_year","ssn","std","nobs","definitive"],
                     skipinitialspace=True)
    df = df[df["year"] >= 2000].copy()
    df.loc[df["ssn"] < 0, "ssn"] = np.nan
    df["ssn"] = df["ssn"].interpolate()
    df["day_number"] = ((pd.to_datetime(df[["year","month","day"]].astype(int)) - pd.Timestamp(INIT_DATE)).dt.days).values
    df.to_csv(cache, index=False)
    print(f"  Cached {len(df)} days")
    return df


# ─── Analysis: 27-day periodicity (detailed) ────────────────────────────────

def solar_rotation_detail(eq_df, kp_df):
    """Detailed 27-day analysis with proper significance testing."""
    print("\n=== 27-Day Solar Rotation: Detailed ===")

    # Build daily earthquake count
    eq_daily = eq_df.groupby("day_number").agg(
        n=("magnitude", "count"),
        max_mag=("magnitude", "max"),
        energy=("magnitude", lambda x: np.sum(10**(1.5*x))),
    ).reset_index()

    all_days = pd.DataFrame({"day_number": np.arange(eq_df["day_number"].min(),
                                                       eq_df["day_number"].max()+1)})
    daily = pd.merge(all_days, eq_daily, on="day_number", how="left").fillna(0)
    daily = pd.merge(daily, kp_df[["day_number","kp_mean","kp_max","ap","f107","sn"]],
                     on="day_number", how="left")

    # Detrend with 365-day running mean
    for col in ["n", "kp_mean", "ap"]:
        daily[f"{col}_dt"] = daily[col] - daily[col].rolling(365, center=True, min_periods=30).mean()
    daily = daily.dropna(subset=["n_dt", "kp_mean_dt"])

    n_dt = daily["n_dt"].values
    kp_dt = daily["kp_mean_dt"].values
    ap_dt = daily["ap_dt"].values

    # Lomb-Scargle periodogram
    from scipy.signal import periodogram
    freqs, psd_eq = periodogram(n_dt, fs=1.0)
    _, psd_kp = periodogram(kp_dt, fs=1.0)

    # Significance: compare power at 27 days to surrounding frequencies
    target_period = 27.3
    target_freq = 1.0 / target_period
    freq_band = (freqs > target_freq * 0.8) & (freqs < target_freq * 1.2)
    surrounding = (freqs > target_freq * 0.5) & (freqs < target_freq * 2.0) & ~freq_band

    if np.any(freq_band) and np.any(surrounding):
        peak_power = np.max(psd_eq[freq_band])
        bg_power = np.median(psd_eq[surrounding])
        snr = peak_power / bg_power
        print(f"  27-day earthquake power: {peak_power:.2f}")
        print(f"  Background power: {bg_power:.2f}")
        print(f"  SNR: {snr:.2f}x")

        peak_kp = np.max(psd_kp[freq_band])
        bg_kp = np.median(psd_kp[surrounding])
        snr_kp = peak_kp / bg_kp
        print(f"  27-day Kp power SNR: {snr_kp:.2f}x")

    # Phase-folded analysis: fold earthquake rate at 27.3-day period
    print("\n  Phase-folded earthquake rate at 27.3-day period:")
    phase = (np.arange(len(n_dt)) % 27.3) / 27.3  # 0 to 1
    n_phase_bins = 10
    phase_bins = np.linspace(0, 1, n_phase_bins + 1)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

    phase_means = []
    for i in range(n_phase_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
        phase_means.append(np.mean(n_dt[mask]))

    phase_means = np.array(phase_means)
    phase_amplitude = (np.max(phase_means) - np.min(phase_means)) / 2
    overall_std = np.std(n_dt)
    significance = phase_amplitude / (overall_std / np.sqrt(len(n_dt) / n_phase_bins))

    print(f"  Phase amplitude: {phase_amplitude:.3f}")
    print(f"  Significance: {significance:.1f} sigma")

    for i, c in enumerate(phase_centers):
        bar = "+" * int(max(0, phase_means[i] + 3) * 5)
        print(f"    Phase {c:.2f}: {phase_means[i]:+.3f} {bar}")

    # Cross-correlation with finer resolution
    print("\n  Cross-correlation (Kp -> earthquakes):")
    max_lag = 50
    xcorr = np.correlate(kp_dt[:len(n_dt)], n_dt, mode='full')
    mid = len(n_dt) - 1
    xcorr_norm = xcorr / (np.std(kp_dt[:len(n_dt)]) * np.std(n_dt) * len(n_dt))

    # Find peaks near 27 days
    for target_lag in [13, 14, 27, 28, 41, 54, 55]:
        idx = mid + target_lag
        if 0 <= idx < len(xcorr_norm):
            note = ""
            if target_lag in [27, 28]: note = " <-- 1 solar rotation"
            if target_lag in [54, 55]: note = " <-- 2 solar rotations"
            if target_lag in [13, 14]: note = " <-- half rotation"
            print(f"    Lag +{target_lag:2d}: r = {xcorr_norm[idx]:+.5f}{note}")

    # Split by solar cycle phase: min vs max
    # Solar min: 2008-2010, 2019-2020. Max: 2001-2003, 2013-2015, 2024-2025
    min_years = [2008, 2009, 2010, 2019, 2020]
    max_years = [2001, 2002, 2003, 2013, 2014, 2015, 2024, 2025]

    daily["year"] = pd.to_datetime(daily["day_number"], unit='D', origin=pd.Timestamp(INIT_DATE)).dt.year
    min_data = daily[daily["year"].isin(min_years)]
    max_data = daily[daily["year"].isin(max_years)]

    print(f"\n  27-day signal split by solar cycle phase:")
    for label, subset in [("Solar MIN", min_data), ("Solar MAX", max_data)]:
        n_sub = subset["n_dt"].values
        if len(n_sub) > 100:
            f_sub, p_sub = periodogram(n_sub, fs=1.0)
            band = (f_sub > 1/30) & (f_sub < 1/25)
            surr = (f_sub > 1/50) & (f_sub < 1/15) & ~band
            if np.any(band) and np.any(surr):
                snr_sub = np.max(p_sub[band]) / np.median(p_sub[surr])
                print(f"    {label}: 27-day SNR = {snr_sub:.2f}x ({len(n_sub)} days)")

    # Plot phase-folded
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax = axes[0]
    ax.bar(phase_centers, phase_means, width=0.08, color="steelblue", alpha=0.7)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Phase (27.3-day period)")
    ax.set_ylabel("Detrended earthquake rate anomaly")
    ax.set_title(f"Phase-Folded Earthquake Rate at 27.3-Day Solar Rotation\n"
                 f"Amplitude: {phase_amplitude:.3f}, Significance: {significance:.1f} sigma")

    ax = axes[1]
    lags_plot = np.arange(-max_lag, max_lag+1)
    xcorr_plot = xcorr_norm[mid-max_lag:mid+max_lag+1]
    ax.plot(lags_plot, xcorr_plot, color="steelblue", linewidth=1)
    ax.axvline(27, color="red", linestyle="--", alpha=0.7, label="27 days")
    ax.axvline(54, color="red", linestyle=":", alpha=0.5, label="54 days")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    noise_level = 2 / np.sqrt(len(n_dt))
    ax.axhline(noise_level, color="orange", linestyle="--", alpha=0.5, label=f"2-sigma noise: {noise_level:.4f}")
    ax.axhline(-noise_level, color="orange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Cross-correlation (Kp -> earthquake rate)")
    ax.set_title("Cross-Correlation: Kp Index Leading Earthquake Rate")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "27day_detail.png", dpi=150)
    print(f"\n  Saved: {OUT_DIR / '27day_detail.png'}")


# ─── Analysis: Solar cycle monthly resolution ───────────────────────────────

def monthly_solar_seismic(eq_df, kp_df, ss_df):
    """Monthly resolution: sunspot number vs earthquake rate."""
    print("\n=== Monthly Solar-Seismic Correlation ===")

    # Monthly earthquake counts
    eq_df = eq_df.copy()
    eq_df["ym"] = eq_df["time_parsed"].dt.to_period("M")
    monthly_eq = eq_df.groupby("ym").agg(
        n=("magnitude", "count"),
        n_m6=("magnitude", lambda x: (x >= 6.0).sum()),
        n_m7=("magnitude", lambda x: (x >= 7.0).sum()),
    ).reset_index()
    monthly_eq["year"] = monthly_eq["ym"].dt.year
    monthly_eq["month"] = monthly_eq["ym"].dt.month

    # Monthly Kp and sunspot
    kp_df = kp_df.copy()
    kp_df["ym"] = pd.to_datetime(kp_df[["year","month","day"]]).dt.to_period("M")
    monthly_kp = kp_df.groupby("ym").agg(
        kp_mean=("kp_mean", "mean"),
        ap_mean=("ap", "mean"),
        f107_mean=("f107", "mean"),
        sn_mean=("sn", "mean"),
        n_storms=("kp_max", lambda x: (x >= 5).sum()),
    ).reset_index()

    merged = pd.merge(monthly_eq, monthly_kp, on="ym", how="inner")
    merged["ym_str"] = merged["ym"].astype(str)

    print(f"  {len(merged)} months of data")

    # Correlations at monthly resolution
    print("\n  Monthly correlations:")
    for eq_col, eq_label in [("n", "N(M5+)"), ("n_m6", "N(M6+)"), ("n_m7", "N(M7+)")]:
        for solar_col, solar_label in [("sn_mean", "Sunspot"), ("f107_mean", "F10.7"),
                                        ("ap_mean", "Ap"), ("n_storms", "N(storms)")]:
            valid = merged[[eq_col, solar_col]].dropna()
            if len(valid) > 20:
                r, p = stats.pearsonr(valid[eq_col], valid[solar_col])
                if abs(r) > 0.1 or p < 0.1:
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "
                    print(f"    {eq_label:>8s} vs {solar_label:>10s}: r = {r:+.3f}, p = {p:.4f} {sig}")

    # Lagged monthly correlation
    print("\n  Lagged monthly correlation (Sunspot -> M5+ earthquakes):")
    for lag in range(-6, 13):
        ssn = merged["sn_mean"].values
        eq = merged["n"].values
        if lag >= 0:
            r = np.corrcoef(ssn[:len(ssn)-max(lag,1)],
                           eq[lag:lag+len(ssn)-max(lag,1)])[0,1]
        else:
            r = np.corrcoef(ssn[-lag:], eq[:len(eq)+lag])[0,1]
        marker = " <--" if abs(r) == max(abs(np.corrcoef(ssn[:len(ssn)-max(abs(lag),1)],
                                                           eq[abs(lag):abs(lag)+len(ssn)-max(abs(lag),1)])[0,1])
                                          for lag in range(-6, 13)) else ""
        print(f"    Lag {lag:+3d} months: r = {r:+.4f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    x = np.arange(len(merged))
    ax = axes[0]
    ax.bar(x, merged["n"], width=1, color="steelblue", alpha=0.6)
    ax.set_ylabel("Monthly M5+ count")
    ax.set_title("Monthly Earthquake Count vs Solar Activity (2000-2026)")

    ax = axes[1]
    ax.plot(x, merged["sn_mean"], color="orange", linewidth=1.5, label="Sunspot #")
    ax.set_ylabel("Monthly sunspot number")
    ax.legend()

    ax = axes[2]
    # Scatter: sunspot vs earthquake count
    ax.scatter(merged["sn_mean"], merged["n"], alpha=0.4, s=15, c=merged["ap_mean"],
               cmap="YlOrRd", edgecolors="none")
    r, p = stats.pearsonr(merged["sn_mean"].dropna(), merged["n"][:len(merged["sn_mean"].dropna())])
    ax.set_xlabel("Monthly sunspot number")
    ax.set_ylabel("Monthly M5+ earthquake count")
    ax.set_title(f"Scatter: r = {r:+.3f}, p = {p:.4f}")
    z = np.polyfit(merged["sn_mean"].dropna(), merged["n"][:len(merged["sn_mean"].dropna())], 1)
    xline = np.linspace(0, merged["sn_mean"].max(), 100)
    ax.plot(xline, np.polyval(z, xline), "r--", linewidth=2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "monthly_solar_seismic.png", dpi=150)
    print(f"\n  Saved: {OUT_DIR / 'monthly_solar_seismic.png'}")


# ─── Analysis: Vanuatu aftermath tracker ─────────────────────────────────────

def vanuatu_aftermath(eq_df, kp_df):
    """Track what actually happened after the March 30 M7.3 + CME."""
    print("\n=== Vanuatu March 30, 2026: What Actually Happened ===")

    # Events from March 28 onward
    march_end = eq_df[(eq_df["time_parsed"] >= "2026-03-28") &
                       (eq_df["time_parsed"] <= "2026-03-31")]

    if len(march_end) > 0:
        print(f"\n  Events March 28-31, 2026 (M>={eq_df['magnitude'].min()}):")
        by_day = march_end.groupby(march_end["time_parsed"].dt.date).agg(
            n=("magnitude", "count"),
            max_mag=("magnitude", "max"),
        )
        print(by_day.to_string())

        # Vanuatu region specifically
        van = march_end[(march_end["latitude"] > -22) & (march_end["latitude"] < -12) &
                        (march_end["longitude"] > 164) & (march_end["longitude"] < 174)]
        print(f"\n  Vanuatu region events: {len(van)}")
        if len(van) > 0:
            for _, ev in van.iterrows():
                print(f"    {ev['time_parsed']}  M{ev['magnitude']:.1f}  "
                      f"depth={ev['depth']:.0f}km  ({ev['latitude']:.2f}, {ev['longitude']:.2f})")
    else:
        print("  No events in catalog for this date range yet")

    # Kp during the period
    kp_march = kp_df[(kp_df["year"] == 2026) & (kp_df["month"] == 3) & (kp_df["day"] >= 28)]
    if len(kp_march) > 0:
        print(f"\n  Kp index March 28-31:")
        for _, row in kp_march.iterrows():
            print(f"    Mar {int(row['day'])}: Kp_mean={row['kp_mean']:.1f}, "
                  f"Kp_max={row['kp_max']:.1f}, Ap={row['ap']:.0f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("LOCAL ANALYSIS — Cached Data")
    print("=" * 70)

    # Download/cache all data
    eq_df = get_earthquakes(min_mag=4.5)
    kp_df = get_kp_daily()
    ss_df = get_sunspots_daily()

    # Run analyses
    solar_rotation_detail(eq_df, kp_df)
    monthly_solar_seismic(eq_df, kp_df, ss_df)
    vanuatu_aftermath(eq_df, kp_df)

    print("\nDone. Outputs in:", OUT_DIR)
    print("Cached data in:", DATA_DIR)


if __name__ == "__main__":
    main()
