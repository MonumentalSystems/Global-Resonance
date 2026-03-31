#!/usr/bin/env python3
"""
Granular Earthquake-Geomagnetic Analysis
=========================================
Goes beyond scalar Kp to model the NON-PLANAR field coupling:

1. Hourly Dst (ring current) instead of 3-hourly Kp
2. Local time of geomagnetic impulse
3. Rate of change dKp/dt and dDst/dt (impulse, not level)
4. Fault-zone specific seismicity (subduction segments)
5. Conditional analysis: storms hitting critically stressed faults

The KT framework says: the coupling is through the COMMUTATOR
[F, nabla F] at the fault boundary. The commutator is large when:
  - The field is non-planar (sin alpha large)
  - The gradient is steep (rapid change in ionospheric charge)
  - The fault is near criticality (J ~ J_c = 2/pi)

So we should see signal in dKp/dt (impulse strength) modulated by
recent seismicity (fault criticality proxy).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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
OUT_DIR.mkdir(exist_ok=True)

INIT_DATE = dt.datetime(2000, 1, 1)

# ─── Data Download ───────────────────────────────────────────────────────────

def download_earthquakes(min_mag=4.0, lat_range=(-25, -10), lon_range=(160, 180)):
    """Download earthquake catalog from USGS for the wider Vanuatu-Tonga region."""
    print(f"Downloading earthquakes (M>={min_mag}, lat {lat_range}, lon {lon_range})...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "csv", "starttime": "2000-01-01", "endtime": "2026-03-31",
        "minlatitude": lat_range[0], "maxlatitude": lat_range[1],
        "minlongitude": lon_range[0], "maxlongitude": lon_range[1],
        "minmagnitude": min_mag, "orderby": "time-asc", "limit": 20000,
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df["time_parsed"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["year"] = df["time_parsed"].dt.year
    df["month"] = df["time_parsed"].dt.month
    df["day"] = df["time_parsed"].dt.day
    df["hour"] = df["time_parsed"].dt.hour
    df["magnitude"] = df["mag"]
    df["day_number"] = ((df["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values
    print(f"  Got {len(df)} earthquakes")
    return df


def download_kp_3hourly():
    """Download 3-hourly Kp from GFZ — gives us dKp/dt resolution."""
    print("Downloading 3-hourly Kp from GFZ...")
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
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if year < 2000:
                continue
            # 8 three-hourly Kp values (0-3h, 3-6h, ..., 21-24h UT)
            kp_vals = [float(parts[7 + i]) for i in range(8)]
            ap_vals = [float(parts[15 + i]) for i in range(8)]
            daily_ap = float(parts[23])

            for slot, (kp, ap) in enumerate(zip(kp_vals, ap_vals)):
                hour = slot * 3  # center of 3-hour window
                records.append({
                    "year": year, "month": month, "day": day,
                    "hour": hour, "kp": kp, "ap": ap,
                    "daily_ap": daily_ap,
                })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["day_number"] = ((df["datetime"] - pd.Timestamp(INIT_DATE)).dt.days).values

    # Compute rate of change (impulse)
    df["dkp_dt"] = df["kp"].diff()
    df["dkp_dt"].iloc[0] = 0

    print(f"  Got {len(df)} 3-hourly records ({len(df)//8} days)")
    return df


# ─── Analysis 1: Impulse (dKp/dt) vs Level (Kp) ────────────────────────────

def impulse_analysis(eq_df, kp_df):
    """
    The commutator [F, nabla F] depends on the GRADIENT of the field,
    not the field itself. So dKp/dt should predict better than Kp.
    """
    print("\n=== Analysis 1: Impulse (dKp/dt) vs Level (Kp) ===")

    # Daily max Kp and max |dKp/dt|
    daily_kp = kp_df.groupby("day_number").agg(
        kp_max=("kp", "max"),
        kp_mean=("kp", "mean"),
        dkp_max=("dkp_dt", lambda x: x.abs().max()),
        dkp_impulse=("dkp_dt", lambda x: x.max()),  # largest positive jump
    ).reset_index()

    # Daily earthquake count
    daily_eq = eq_df.groupby("day_number").agg(
        n_quakes=("magnitude", "count"),
        max_mag=("magnitude", "max"),
        total_energy=("magnitude", lambda x: np.sum(10 ** (1.5 * x))),
    ).reset_index()

    merged = pd.merge(daily_kp, daily_eq, on="day_number", how="left")
    merged["n_quakes"] = merged["n_quakes"].fillna(0)
    merged["max_mag"] = merged["max_mag"].fillna(0)
    merged["total_energy"] = merged["total_energy"].fillna(0)
    merged["log_energy"] = np.log10(merged["total_energy"] + 1)

    # Compare correlations at various lags
    print("\nCorrelation with next-day seismicity:")
    print(f"  {'Predictor':>20s}  {'Lag 0':>8s}  {'Lag 1':>8s}  {'Lag 2':>8s}  {'Lag 3':>8s}")

    for col, label in [("kp_max", "Kp (max)"),
                        ("kp_mean", "Kp (mean)"),
                        ("dkp_max", "|dKp/dt| (max)"),
                        ("dkp_impulse", "dKp/dt (impulse)")]:
        corrs = []
        for lag in [0, 1, 2, 3]:
            x = merged[col].values[:-max(lag, 1)]
            y = merged["n_quakes"].shift(-lag).values[:-max(lag, 1)]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 10:
                corrs.append(f"{np.corrcoef(x[mask], y[mask])[0,1]:+.4f}")
            else:
                corrs.append("   N/A")
        print(f"  {label:>20s}  {'  '.join(corrs)}")

    return merged


# ─── Analysis 2: Conditional — storms on critically stressed faults ─────────

def conditional_analysis(eq_df, kp_df):
    """
    The framework says: coupling only works when the fault is near
    criticality (J ~ J_c). Use recent seismicity as a proxy for
    fault stress state. High recent activity = fault near critical.

    Test: geomagnetic impulse FOLLOWING high seismicity should
    predict more than impulse following quiet periods.
    """
    print("\n=== Analysis 2: Conditional — Storms on Critical Faults ===")

    daily_kp = kp_df.groupby("day_number").agg(
        kp_max=("kp", "max"),
        dkp_impulse=("dkp_dt", lambda x: x.max()),
    ).reset_index()

    daily_eq = eq_df.groupby("day_number").agg(
        n_quakes=("magnitude", "count"),
        max_mag=("magnitude", "max"),
    ).reset_index()

    merged = pd.merge(daily_kp, daily_eq, on="day_number", how="left")
    merged["n_quakes"] = merged["n_quakes"].fillna(0)
    merged["max_mag"] = merged["max_mag"].fillna(0)

    # 7-day rolling seismicity (proxy for fault stress state)
    merged["recent_seismicity"] = merged["n_quakes"].rolling(7, min_periods=1).sum()

    # Next-day max magnitude
    merged["next_max_mag"] = merged["max_mag"].shift(-1)
    merged["next_n_quakes"] = merged["n_quakes"].shift(-1)

    # Split into high/low recent seismicity
    median_seis = merged["recent_seismicity"].median()
    high_stress = merged[merged["recent_seismicity"] > median_seis]
    low_stress = merged[merged["recent_seismicity"] <= median_seis]

    # Storm threshold
    storm_mask_h = high_stress["kp_max"] >= 4
    storm_mask_l = low_stress["kp_max"] >= 4

    print(f"\nMedian 7-day seismicity: {median_seis:.0f} events")
    print(f"High-stress days: {len(high_stress)}, Low-stress days: {len(low_stress)}")

    cats = [
        ("High stress + storm", high_stress[storm_mask_h]),
        ("High stress + quiet", high_stress[~storm_mask_h]),
        ("Low stress + storm", low_stress[storm_mask_l]),
        ("Low stress + quiet", low_stress[~storm_mask_l]),
    ]

    print(f"\n{'Category':>25s}  {'N':>5s}  {'Mean next-day quakes':>20s}  {'Mean next max mag':>18s}")
    for label, subset in cats:
        nq = subset["next_n_quakes"].dropna()
        nm = subset["next_max_mag"].dropna()
        print(f"  {label:>23s}  {len(subset):>5d}  {nq.mean():>20.3f}  {nm.mean():>18.2f}")

    # The KT prediction: "High stress + storm" >> "High stress + quiet" > "Low stress + storm"
    hs_storm = high_stress[storm_mask_h]["next_n_quakes"].dropna()
    hs_quiet = high_stress[~storm_mask_h]["next_n_quakes"].dropna()

    if len(hs_storm) > 5 and len(hs_quiet) > 5:
        _, p = stats.mannwhitneyu(hs_storm, hs_quiet, alternative="greater")
        print(f"\nMann-Whitney (high stress: storm > quiet): p = {p:.4f}")
        print(f"  Storm mean: {hs_storm.mean():.3f}, Quiet mean: {hs_quiet.mean():.3f}, "
              f"Ratio: {hs_storm.mean()/max(hs_quiet.mean(), 0.001):.2f}x")

    return merged


# ─── Analysis 3: Depth-dependent coupling ───────────────────────────────────

def depth_analysis(eq_df, kp_df):
    """
    The ionospheric capacitor mechanism should couple most strongly
    to SHALLOW earthquakes (< 70 km) where crustal voids exist.
    Deep earthquakes (> 300 km) should show no coupling.
    """
    print("\n=== Analysis 3: Depth-Dependent Coupling ===")

    daily_kp = kp_df.groupby("day_number").agg(
        kp_max=("kp", "max"),
    ).reset_index()

    depth_bins = [(0, 70, "Shallow (<70 km)"),
                  (70, 300, "Intermediate (70-300 km)"),
                  (300, 800, "Deep (>300 km)")]

    print(f"\n{'Depth bin':>25s}  {'N events':>8s}  {'Corr(Kp, quakes) lag+1':>22s}  {'p-value':>8s}")

    for dmin, dmax, label in depth_bins:
        subset = eq_df[(eq_df["depth"] >= dmin) & (eq_df["depth"] < dmax)]
        if len(subset) < 50:
            print(f"  {label:>23s}  {len(subset):>8d}  {'insufficient data':>22s}")
            continue

        daily_sub = subset.groupby("day_number").agg(n=("magnitude", "count")).reset_index()
        merged = pd.merge(daily_kp, daily_sub, on="day_number", how="left")
        merged["n"] = merged["n"].fillna(0)

        # Lag +1 correlation
        x = merged["kp_max"].values[:-1]
        y = merged["n"].values[1:]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 10:
            r, p = stats.pearsonr(x[mask], y[mask])
            print(f"  {label:>23s}  {len(subset):>8d}  {r:>+22.4f}  {p:>8.4f}")


# ─── Analysis 4: Superposed epoch — CME arrival ────────────────────────────

def superposed_epoch(eq_df, kp_df):
    """
    Superposed epoch analysis: stack earthquake rates around
    sudden storm commencements (SSC = rapid Kp increase >= 3 in 3 hours).
    This is the direct test of the CME impulse mechanism.
    """
    print("\n=== Analysis 4: Superposed Epoch — Geomagnetic Sudden Impulses ===")

    # Find sudden impulses: dKp >= 2.0 in one 3-hour step
    impulses = kp_df[kp_df["dkp_dt"] >= 2.0].copy()

    # Deduplicate (keep first impulse per storm, require 3-day gap)
    impulse_days = impulses["day_number"].unique()
    filtered = [impulse_days[0]]
    for d in impulse_days[1:]:
        if d - filtered[-1] >= 3:
            filtered.append(d)
    impulse_days = np.array(filtered)

    print(f"Found {len(impulse_days)} sudden impulses (dKp >= 2.0, 3-day dedup)")

    # Daily earthquake count
    daily_eq = eq_df.groupby("day_number").agg(n=("magnitude", "count")).reset_index()
    all_days = pd.DataFrame({"day_number": np.arange(daily_eq["day_number"].min(),
                                                       daily_eq["day_number"].max() + 1)})
    daily_eq = pd.merge(all_days, daily_eq, on="day_number", how="left")
    daily_eq["n"] = daily_eq["n"].fillna(0)

    # Stack: earthquake rate at day -7 to +14 relative to each impulse
    window = np.arange(-7, 15)
    stacked = np.zeros((len(impulse_days), len(window)))

    for i, d0 in enumerate(impulse_days):
        for j, offset in enumerate(window):
            day = d0 + offset
            row = daily_eq[daily_eq["day_number"] == day]
            if len(row) > 0:
                stacked[i, j] = row["n"].values[0]

    mean_rate = np.mean(stacked, axis=0)
    sem_rate = stats.sem(stacked, axis=0)

    # Background rate (days -7 to -1)
    bg = np.mean(mean_rate[:7])

    print(f"\nBackground rate (days -7 to -1): {bg:.3f} quakes/day")
    print(f"\nDay   Mean rate   Ratio to BG   SEM")
    for j, offset in enumerate(window):
        ratio = mean_rate[j] / max(bg, 0.001)
        marker = " <-- impulse" if offset == 0 else ""
        marker = " <-- +1 day" if offset == 1 else marker
        marker = " <-- +2 day" if offset == 2 else marker
        print(f"  {offset:+3d}    {mean_rate[j]:.3f}       {ratio:.2f}x       {sem_rate[j]:.3f}{marker}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(window, mean_rate, yerr=sem_rate, alpha=0.7, color="steelblue", capsize=2)
    ax.axhline(bg, color="red", linestyle="--", alpha=0.7, label=f"Background: {bg:.2f}")
    ax.axvline(0, color="orange", linestyle="-", alpha=0.8, label="Geomagnetic impulse")
    ax.set_xlabel("Days relative to sudden impulse (dKp >= 2.0)")
    ax.set_ylabel("Mean earthquake count (M >= 4.0)")
    ax.set_title("Superposed Epoch: Seismicity Around Geomagnetic Sudden Impulses\n"
                 f"Vanuatu-Tonga Region, {len(impulse_days)} events stacked, 2000-2026")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "superposed_epoch.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'superposed_epoch.png'}")

    return mean_rate, sem_rate, bg


# ─── Analysis 5: Local time dependence ──────────────────────────────────────

def local_time_analysis(eq_df, kp_df):
    """
    The ionospheric coupling depends on local time.
    Vanuatu is at ~167E = UT+11. Daytime ionosphere (more TEC,
    stronger coupling) vs nighttime (weaker).

    Check: do storms arriving during Vanuatu daytime correlate
    better with seismicity than nighttime storms?
    """
    print("\n=== Analysis 5: Local Time Dependence ===")

    VANUATU_UTC_OFFSET = 11  # hours

    # Tag each 3-hourly Kp with Vanuatu local time
    kp_df = kp_df.copy()
    kp_df["local_hour"] = (kp_df["hour"] + VANUATU_UTC_OFFSET) % 24
    kp_df["is_daytime"] = (kp_df["local_hour"] >= 6) & (kp_df["local_hour"] <= 18)

    # Find storm slots (Kp >= 4)
    storms = kp_df[kp_df["kp"] >= 4].copy()

    # Daily earthquake count
    daily_eq = eq_df.groupby("day_number").agg(n=("magnitude", "count")).reset_index()

    # Daytime vs nighttime storm days
    day_storms = storms[storms["is_daytime"]]["day_number"].unique()
    night_storms = storms[~storms["is_daytime"]]["day_number"].unique()

    # Next-day seismicity after daytime vs nighttime storms
    def next_day_rate(storm_days):
        rates = []
        for d in storm_days:
            row = daily_eq[daily_eq["day_number"] == d + 1]
            rates.append(row["n"].values[0] if len(row) > 0 else 0)
        return np.array(rates)

    day_rates = next_day_rate(day_storms)
    night_rates = next_day_rate(night_storms)

    print(f"\nDaytime storms (Vanuatu local): {len(day_storms)} days")
    print(f"  Next-day mean quakes: {day_rates.mean():.3f}")
    print(f"Nighttime storms: {len(night_storms)} days")
    print(f"  Next-day mean quakes: {night_rates.mean():.3f}")

    if len(day_rates) > 5 and len(night_rates) > 5:
        _, p = stats.mannwhitneyu(day_rates, night_rates, alternative="greater")
        ratio = day_rates.mean() / max(night_rates.mean(), 0.001)
        print(f"\nDaytime/Nighttime ratio: {ratio:.2f}x  (p = {p:.4f})")
        print("Framework predicts: daytime > nighttime (stronger ionospheric TEC)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GRANULAR EARTHQUAKE-GEOMAGNETIC ANALYSIS")
    print("Non-planar field coupling: impulse, depth, local time, fault state")
    print("=" * 70)

    eq_df = download_earthquakes(min_mag=4.0, lat_range=(-25, -10), lon_range=(160, 180))
    kp_df = download_kp_3hourly()

    merged1 = impulse_analysis(eq_df, kp_df)
    merged2 = conditional_analysis(eq_df, kp_df)
    depth_analysis(eq_df, kp_df)
    epoch_rate, epoch_sem, epoch_bg = superposed_epoch(eq_df, kp_df)
    local_time_analysis(eq_df, kp_df)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The KT framework predicts:
1. dKp/dt (impulse) > Kp (level) as predictor [commutator = gradient]
2. Effect only on critically stressed faults [J ~ J_c]
3. Shallow > deep earthquakes [capacitor mechanism]
4. Enhancement at day +1 to +2 after sudden impulse [CME propagation time]
5. Daytime > nighttime storms [ionospheric TEC amplification]

Each positive result is evidence for the non-planar field coupling
mechanism. Each null result constrains where the model needs refinement.
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
