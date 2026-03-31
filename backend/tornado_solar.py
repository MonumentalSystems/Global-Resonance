#!/usr/bin/env python3
"""
Tornado-Solar Cycle Analysis
==============================
74 years of tornadoes (1950-2023) vs solar cycle.
Also: SSW events, cosmic rays, bombogenesis timing.

The KT framework predicts:
  Solar max → stronger magnetosphere → higher J → more stable
  → fewer extreme vortex events (tornadoes, SSW, bombogenesis)

Same mechanism as earthquakes but through the atmosphere.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, signal
from pathlib import Path
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"


def load_tornadoes():
    print("Loading tornado data...")
    df = pd.read_csv(DATA_DIR / "tornadoes_1950_2023.csv", low_memory=False)
    print(f"  Columns: {list(df.columns)[:15]}")
    print(f"  {len(df)} tornado records")
    # Standardize column names (SPC format varies)
    cols = {c.strip().lower(): c for c in df.columns}
    # Try to find year, magnitude, fatalities columns
    for name_try in ['yr', 'year']:
        if name_try in cols:
            df['year'] = df[cols[name_try]]
            break
    for name_try in ['mag', 'f_scale', 'ef_scale', 'tor_f_scale']:
        if name_try in cols:
            df['magnitude'] = pd.to_numeric(df[cols[name_try]], errors='coerce')
            break
    for name_try in ['fat', 'fatalities', 'deaths']:
        if name_try in cols:
            df['fatalities'] = pd.to_numeric(df[cols[name_try]], errors='coerce')
            break
    for name_try in ['mo', 'month']:
        if name_try in cols:
            df['month'] = df[cols[name_try]]
            break
    print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
    return df


def load_kp_yearly():
    kp = pd.read_csv(DATA_DIR / "kp_daily.csv")
    yearly = kp.groupby("year").agg(
        kp_mean=("kp_mean", "mean"),
        kp_max=("kp_max", "max"),
        ap_mean=("ap", "mean"),
        f107_mean=("f107", lambda x: x[x > 0].mean()),
        sn_mean=("sn", lambda x: x[x >= 0].mean()),
        n_storms=("kp_max", lambda x: (x >= 5).sum()),
    ).reset_index()
    return yearly


def load_ssw():
    df = pd.read_csv(DATA_DIR / "ssw_events.csv", parse_dates=["date"])
    return df


def load_bombogenesis():
    df = pd.read_csv(DATA_DIR / "bombogenesis_events.csv", parse_dates=["date"])
    return df


# ═══════════════════════════════════════════════════════════════════════
# 1. TORNADO COUNT vs SOLAR CYCLE
# ═══════════════════════════════════════════════════════════════════════

def tornado_solar_cycle(torn_df, kp_yearly):
    print("\n=== Tornado Count vs Solar Cycle ===")

    # Annual tornado counts by intensity
    yearly_torn = torn_df.groupby("year").agg(
        n_total=("year", "count"),
        n_strong=("magnitude", lambda x: (x >= 3).sum()),  # EF3+
        n_violent=("magnitude", lambda x: (x >= 4).sum()),  # EF4+
        n_fatal=("fatalities", lambda x: (x > 0).sum()),
        total_fatalities=("fatalities", "sum"),
    ).reset_index()

    merged = pd.merge(yearly_torn, kp_yearly, on="year", how="inner")
    print(f"  Years with both tornado and solar data: {len(merged)}")

    # Correlations
    print("\n  Correlations (raw):")
    print(f"  {'Tornado metric':>20s}  {'Sunspot':>10s}  {'F10.7':>10s}  {'Ap':>10s}  {'N storms':>10s}")

    for tcol, tlabel in [("n_total", "Total tornadoes"),
                          ("n_strong", "EF3+ tornadoes"),
                          ("n_violent", "EF4+ tornadoes"),
                          ("n_fatal", "Fatal tornadoes"),
                          ("total_fatalities", "Total deaths")]:
        row = f"  {tlabel:>20s}"
        for scol in ["sn_mean", "f107_mean", "ap_mean", "n_storms"]:
            valid = merged[[tcol, scol]].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[tcol], valid[scol])
                sig = "*" if p < 0.05 else " "
                row += f"  {r:>+7.3f}{sig:1s} "
            else:
                row += f"  {'N/A':>8s} "
        print(row)

    # Detrended (tornado detection has improved over time)
    print("\n  Detrended correlations (remove linear trend from tornado counts):")
    for tcol, tlabel in [("n_total", "Total"), ("n_strong", "EF3+"),
                          ("n_violent", "EF4+"), ("n_fatal", "Fatal")]:
        x = merged["year"].values
        y = merged[tcol].values
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            continue
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        detrended = y - (slope * x + intercept)

        row = f"  {tlabel:>20s}"
        for scol in ["sn_mean", "f107_mean", "ap_mean"]:
            sv = merged[scol].values
            valid = ~(np.isnan(detrended) | np.isnan(sv))
            if valid.sum() > 10:
                r, p = stats.pearsonr(detrended[valid], sv[valid])
                sig = "*" if p < 0.05 else " "
                row += f"  {r:>+7.3f}{sig:1s} "
            else:
                row += f"  {'N/A':>8s} "
        print(row)

    # Solar phase binning
    median_sn = merged["sn_mean"].median()
    high = merged[merged["sn_mean"] > median_sn * 1.5]
    low = merged[merged["sn_mean"] < median_sn * 0.5]
    mid = merged[(merged["sn_mean"] >= median_sn * 0.5) &
                  (merged["sn_mean"] <= median_sn * 1.5)]

    print(f"\n  Solar phase (median SSN = {median_sn:.0f}):")
    print(f"  {'Phase':>15s} {'Years':>6s} {'Total/yr':>10s} {'EF3+/yr':>10s} {'EF4+/yr':>10s} {'Deaths/yr':>10s}")
    for label, subset in [("Solar MAX", high), ("Solar MID", mid), ("Solar MIN", low)]:
        if len(subset) > 0:
            print(f"  {label:>15s} {len(subset):>6d} "
                  f"{subset['n_total'].mean():>10.0f} "
                  f"{subset['n_strong'].mean():>10.1f} "
                  f"{subset['n_violent'].mean():>10.1f} "
                  f"{subset['total_fatalities'].mean():>10.0f}")

    # Monthly resolution — tornado season (Apr-Jun) vs off-season
    print("\n  Tornado season (Apr-Jun) vs off-season correlation:")
    torn_monthly = torn_df.groupby(["year", "month"]).agg(n=("year", "count")).reset_index()
    season = torn_monthly[torn_monthly["month"].isin([4, 5, 6])]
    season_yearly = season.groupby("year").agg(n_season=("n", "sum")).reset_index()

    off = torn_monthly[~torn_monthly["month"].isin([4, 5, 6])]
    off_yearly = off.groupby("year").agg(n_offseason=("n", "sum")).reset_index()

    for label, sub_df, col in [("Season (AMJ)", season_yearly, "n_season"),
                                 ("Off-season", off_yearly, "n_offseason")]:
        m = pd.merge(sub_df, kp_yearly, on="year", how="inner")
        if len(m) > 10:
            r, p = stats.pearsonr(m[col], m["sn_mean"].fillna(0))
            print(f"    {label}: r(tornadoes, SSN) = {r:+.3f}, p = {p:.3f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    ax = axes[0]
    ax.bar(merged["year"], merged["n_total"], color="steelblue", alpha=0.6, label="Total tornadoes")
    ax2 = ax.twinx()
    ax2.plot(merged["year"], merged["sn_mean"], 'o-', color="orange", linewidth=2,
             markersize=3, label="Sunspot #")
    ax.set_ylabel("Annual tornado count")
    ax2.set_ylabel("Sunspot number", color="orange")
    ax.set_title("US Tornadoes vs Solar Cycle (1950-2023)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1]
    ax.bar(merged["year"], merged["n_strong"], color="#e78ac3", alpha=0.7, label="EF3+ tornadoes")
    ax2 = ax.twinx()
    ax2.plot(merged["year"], merged["sn_mean"], 'o-', color="orange", linewidth=2,
             markersize=3, alpha=0.5)
    ax.set_ylabel("EF3+ tornado count")
    ax2.set_ylabel("Sunspot number", color="orange")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.scatter(merged["sn_mean"], merged["n_strong"], alpha=0.5, s=30, color="#e78ac3")
    valid = merged[["sn_mean", "n_strong"]].dropna()
    if len(valid) > 5:
        z = np.polyfit(valid["sn_mean"], valid["n_strong"], 1)
        xline = np.linspace(0, valid["sn_mean"].max(), 100)
        ax.plot(xline, np.polyval(z, xline), "r--", linewidth=2)
        r, p = stats.pearsonr(valid["sn_mean"], valid["n_strong"])
        ax.set_title(f"EF3+ Tornadoes vs Sunspot Number: r = {r:+.3f}, p = {p:.3f}")
    ax.set_xlabel("Annual mean sunspot number")
    ax.set_ylabel("EF3+ tornadoes per year")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "tornado_solar_cycle.png", dpi=150)
    print(f"\n  Saved: {OUT_DIR / 'tornado_solar_cycle.png'}")

    return merged


# ═══════════════════════════════════════════════════════════════════════
# 2. SSW vs SOLAR CYCLE
# ═══════════════════════════════════════════════════════════════════════

def ssw_solar_cycle(ssw_df, kp_yearly):
    print("\n=== Sudden Stratospheric Warmings vs Solar Cycle ===")
    print("SSW = polar vortex unbinding event")

    ssw_yearly = ssw_df.groupby("year").agg(n_ssw=("year", "count")).reset_index()
    merged = pd.merge(ssw_yearly, kp_yearly, on="year", how="right")
    merged["n_ssw"] = merged["n_ssw"].fillna(0)
    merged = merged[merged["year"] >= 1980]

    r, p = stats.pearsonr(merged["n_ssw"], merged["sn_mean"].fillna(0))
    print(f"  SSW count vs SSN: r = {r:+.3f}, p = {p:.3f}")

    # SSW by solar phase
    median_sn = merged["sn_mean"].median()
    high = merged[merged["sn_mean"] > median_sn]
    low = merged[merged["sn_mean"] <= median_sn]

    print(f"  Solar MAX years: {high['n_ssw'].mean():.2f} SSW/year (N={len(high)} years)")
    print(f"  Solar MIN years: {low['n_ssw'].mean():.2f} SSW/year (N={len(low)} years)")

    # SSW type split
    splits = ssw_df[ssw_df["type"] == "split"]
    displacements = ssw_df[ssw_df["type"] == "displacement"]
    print(f"\n  Split SSW: {len(splits)} events")
    print(f"  Displacement SSW: {len(displacements)} events")

    # Merge SSW events with solar data at the time
    ssw_df = ssw_df.copy()
    ssw_df["year"] = ssw_df["date"].dt.year
    ssw_solar = pd.merge(ssw_df, kp_yearly[["year", "sn_mean", "ap_mean"]], on="year", how="left")

    print(f"\n  SSW events by type and solar phase:")
    for stype in ["split", "displacement"]:
        subset = ssw_solar[ssw_solar["type"] == stype]
        if len(subset) > 3:
            print(f"    {stype}: mean SSN = {subset['sn_mean'].mean():.0f}, "
                  f"mean Ap = {subset['ap_mean'].mean():.1f}")


# ═══════════════════════════════════════════════════════════════════════
# 3. HURRICANE INTENSITY vs SOLAR CYCLE
# ═══════════════════════════════════════════════════════════════════════

def hurricane_solar_cycle(kp_yearly):
    print("\n=== Hurricane Intensity vs Solar Cycle ===")

    ibtracs = pd.read_csv(DATA_DIR / "ibtracs_since1980.csv", low_memory=False,
                          skiprows=[1])  # IBTrACS has a units row
    print(f"  IBTrACS records: {len(ibtracs)}")
    print(f"  Columns: {list(ibtracs.columns)[:10]}...")

    # Parse — IBTrACS has SID, ISO_TIME, LAT, LON, WMO_WIND, WMO_PRES, etc.
    ibtracs["year"] = pd.to_numeric(ibtracs["SEASON"], errors="coerce")
    # USA_WIND is the most complete; strip whitespace and convert
    ibtracs["wind"] = pd.to_numeric(ibtracs["USA_WIND"].astype(str).str.strip(), errors="coerce")
    ibtracs["pres"] = pd.to_numeric(ibtracs["USA_PRES"].astype(str).str.strip(), errors="coerce")

    # Count storms per year, max intensity
    yearly_tc = ibtracs.groupby(["year", "SID"]).agg(
        max_wind=("wind", "max"),
        min_pres=("pres", "min"),
    ).reset_index()
    yearly_tc = yearly_tc.dropna(subset=["max_wind"])

    tc_yearly = yearly_tc.groupby("year").agg(
        n_storms=("SID", "nunique"),
        n_cat3plus=("max_wind", lambda x: (x >= 96).sum()),
        n_cat4plus=("max_wind", lambda x: (x >= 113).sum()),
        n_cat5=("max_wind", lambda x: (x >= 137).sum()),
        mean_max_wind=("max_wind", "mean"),
    ).reset_index()

    merged = pd.merge(tc_yearly, kp_yearly, on="year", how="inner")
    print(f"  Years with both data: {len(merged)}")

    print("\n  Correlations:")
    print(f"  {'Metric':>20s}  {'SSN':>10s}  {'Ap':>10s}")
    for col, label in [("n_storms", "Total TCs"), ("n_cat3plus", "Cat 3+"),
                        ("n_cat4plus", "Cat 4+"), ("mean_max_wind", "Mean max wind")]:
        if col not in merged.columns:
            continue
        row = f"  {label:>20s}"
        for scol in ["sn_mean", "ap_mean"]:
            valid = merged[[col, scol]].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[col], valid[scol])
                sig = "*" if p < 0.05 else " "
                row += f"  {r:>+7.3f}{sig:1s} "
        print(row)

    # ACE proxy
    ibtracs["wind2"] = ibtracs["wind"] ** 2
    ace_yearly = ibtracs.dropna(subset=["wind2"]).groupby("year").agg(
        ace=("wind2", "sum")).reset_index()
    ace_merged = pd.merge(ace_yearly, kp_yearly, on="year", how="inner")
    if len(ace_merged) > 10:
        r, p = stats.pearsonr(ace_merged["ace"], ace_merged["sn_mean"].fillna(0))
        print(f"  ACE vs SSN: r = {r:+.3f}, p = {p:.3f}")

    return merged


# ═══════════════════════════════════════════════════════════════════════
# 4. ALL VORTEX EVENTS: UNIFIED SOLAR MODULATION
# ═══════════════════════════════════════════════════════════════════════

def unified_vortex_solar(torn_merged, kp_yearly, ssw_df):
    print("\n=== Unified Vortex-Solar Picture ===")
    print("Do ALL vortex types anti-correlate with solar activity?")

    # Earthquakes (from our earlier analysis)
    eq_yearly = pd.read_csv(DATA_DIR / "earthquakes_yearly_m5.csv") if \
        (DATA_DIR / "earthquakes_yearly_m5.csv").exists() else None

    print(f"\n  {'Phenomenon':>25s}  {'r(SSN)':>10s}  {'p-value':>10s}  {'Direction':>12s}")

    results = []

    # Tornadoes (EF3+)
    if "n_strong" in torn_merged.columns:
        valid = torn_merged[["n_strong", "sn_mean"]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid["n_strong"], valid["sn_mean"])
            direction = "FEWER at max" if r < 0 else "MORE at max"
            print(f"  {'EF3+ Tornadoes':>25s}  {r:>+10.3f}  {p:>10.3f}  {direction:>12s}")
            results.append(("EF3+ Tornadoes", r, p))

    # All tornadoes (detrended)
    if "n_total" in torn_merged.columns:
        x = torn_merged["year"].values
        y = torn_merged["n_total"].values
        mask = ~np.isnan(y)
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        detrended = y - (slope * x + intercept)
        sn = torn_merged["sn_mean"].values
        valid = ~(np.isnan(detrended) | np.isnan(sn))
        if valid.sum() > 10:
            r, p = stats.pearsonr(detrended[valid], sn[valid])
            direction = "FEWER at max" if r < 0 else "MORE at max"
            print(f"  {'Tornadoes (detrended)':>25s}  {r:>+10.3f}  {p:>10.3f}  {direction:>12s}")
            results.append(("Tornadoes (detrended)", r, p))

    # SSW events
    ssw_yearly = ssw_df.groupby(ssw_df["date"].dt.year).size().reset_index(name="n_ssw")
    ssw_yearly.columns = ["year", "n_ssw"]
    ssw_m = pd.merge(ssw_yearly, kp_yearly, on="year", how="right").fillna(0)
    ssw_m = ssw_m[ssw_m["year"] >= 1980]
    valid = ssw_m[["n_ssw", "sn_mean"]].dropna()
    if len(valid) > 10:
        r, p = stats.pearsonr(valid["n_ssw"], valid["sn_mean"])
        direction = "FEWER at max" if r < 0 else "MORE at max"
        print(f"  {'SSW events':>25s}  {r:>+10.3f}  {p:>10.3f}  {direction:>12s}")
        results.append(("SSW events", r, p))

    # Earthquakes
    if eq_yearly is not None and len(eq_yearly) > 0:
        eq_m = pd.merge(eq_yearly, kp_yearly, on="year", how="inner")
        if len(eq_m) > 10 and "n_m5" in eq_m.columns:
            r, p = stats.pearsonr(eq_m["n_m5"], eq_m["sn_mean"].fillna(0))
            direction = "FEWER at max" if r < 0 else "MORE at max"
            print(f"  {'M5+ Earthquakes':>25s}  {r:>+10.3f}  {p:>10.3f}  {direction:>12s}")
            results.append(("M5+ Earthquakes", r, p))

        if "n_m7" in eq_m.columns:
            r, p = stats.pearsonr(eq_m["n_m7"], eq_m["sn_mean"].fillna(0))
            direction = "FEWER at max" if r < 0 else "MORE at max"
            print(f"  {'M7+ Earthquakes':>25s}  {r:>+10.3f}  {p:>10.3f}  {direction:>12s}")
            results.append(("M7+ Earthquakes", r, p))

    # Summary plot
    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r[0] for r in results]
        corrs = [r[1] for r in results]
        pvals = [r[2] for r in results]
        colors = ["#e41a1c" if p < 0.05 else "#377eb8" for p in pvals]

        bars = ax.barh(names, corrs, color=colors, alpha=0.7)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Correlation with sunspot number (r)")
        ax.set_title("Unified Solar Modulation of ALL Vortex Phenomena\n"
                      "Red = significant (p < 0.05), Blue = not significant")

        for i, (r, p) in enumerate(zip(corrs, pvals)):
            ax.text(r + 0.01 * np.sign(r), i, f"r={r:+.3f}\np={p:.3f}",
                    va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(OUT_DIR / "unified_vortex_solar.png", dpi=150)
        print(f"\n  Saved: {OUT_DIR / 'unified_vortex_solar.png'}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("TORNADO / HURRICANE / SSW vs SOLAR CYCLE")
    print("Does the sun modulate ALL vortex phenomena?")
    print("=" * 70)

    torn_df = load_tornadoes()
    kp_yearly = load_kp_yearly()
    ssw_df = load_ssw()

    torn_merged = tornado_solar_cycle(torn_df, kp_yearly)
    ssw_solar_cycle(ssw_df, kp_yearly)
    hurricane_solar_cycle(kp_yearly)
    unified_vortex_solar(torn_merged, kp_yearly, ssw_df)

    print("\n" + "=" * 70)
    print("FRAMEWORK PREDICTION")
    print("=" * 70)
    print("""
The KT framework predicts: solar activity modulates the effective
coupling stiffness J in ALL oscillator systems coupled to the
heliosphere. Higher solar activity -> higher J -> more ordered
-> fewer vortex unbinding events.

This should produce ANTI-CORRELATION between sunspot number and:
  - Earthquake rate (CONFIRMED: r = -0.355, p = 0.016)
  - Tornado rate (testing now)
  - SSW rate (testing now)
  - Hurricane intensity (testing now)

If ALL show the same sign, the coupling is universal.
One sun, one equation, all vortices.
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
