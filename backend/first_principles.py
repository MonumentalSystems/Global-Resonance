#!/usr/bin/env python3
"""
First Principles Prediction + Solar Cycle + Schumann
======================================================

Part 1: PREDICT the Vanuatu M7.3 aftermath from first principles
  - CME arrival March 31, G3-G4 storm
  - The jelly ball model says: suppression at subsolar, wave at 60-90 deg
  - The bearing model says: E-W propagation preferred
  - The volcanic model says: equatorial weak-B regions squeeze

Part 2: 11-YEAR SOLAR CYCLE coarse-grained correlation
  - Solar max = strong B, more CMEs, higher Kp
  - Solar min = weak B, fewer perturbations
  - The FLIP (Hale cycle, ~22 years) = complete magnetic reversal
  - Does global seismicity track the cycle?

Part 3: CORONAL HOLES (solar voids)
  - High-speed solar wind streams from coronal holes
  - Different from CMEs: sustained (days) vs impulsive (hours)
  - Check if recurrent high-speed streams correlate with seismicity

Part 4: SCHUMANN RESONANCE as the KT order parameter
  - f_S ~ 7.83 Hz = c / (2*pi*R_earth)
  - If Earth's EM cavity is a Kuramoto oscillator at KT criticality,
    the Schumann frequency should modulate with geomagnetic activity
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
OUT_DIR.mkdir(exist_ok=True)
INIT_DATE = dt.datetime(2000, 1, 1)

PI = np.pi


# ─── Data ────────────────────────────────────────────────────────────────────

def download_earthquakes_yearly():
    """Download global M5+ earthquakes, return yearly counts + energy."""
    print("Downloading global M5+ earthquakes by year...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    yearly = []
    for year in range(1980, 2027):
        try:
            resp = requests.get(url, params={
                "format": "csv", "starttime": f"{year}-01-01",
                "endtime": f"{year}-12-31", "minmagnitude": 5.0,
                "orderby": "time-asc", "limit": 20000,
            }, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            n = len(df)
            if n > 0:
                energy = np.sum(10**(1.5 * df["mag"]))
                max_mag = df["mag"].max()
                m7_count = len(df[df["mag"] >= 7.0])
            else:
                energy, max_mag, m7_count = 0, 0, 0
            yearly.append({"year": year, "n_m5": n, "log_energy": np.log10(energy + 1),
                          "max_mag": max_mag, "n_m7": m7_count})
            print(f"  {year}: {n} M5+, {m7_count} M7+, max={max_mag:.1f}")
        except Exception as e:
            print(f"  {year}: failed ({e})")
    return pd.DataFrame(yearly)


def download_sunspots_yearly():
    """Download yearly sunspot numbers from SILSO."""
    print("Downloading yearly sunspot numbers...")
    url = "https://www.sidc.be/SILSO/DATA/SN_y_tot_V2.0.csv"
    resp = requests.get(url, timeout=30)
    df = pd.read_csv(StringIO(resp.text), sep=";", header=None,
                     names=["year", "ssn", "std", "nobs", "definitive"],
                     skipinitialspace=True)
    df = df[df["year"] >= 1980].copy()
    df["year"] = df["year"].astype(int)
    print(f"  {len(df)} yearly records")
    return df


def download_kp_yearly():
    """Download Kp and compute yearly statistics."""
    print("Downloading Kp for yearly aggregation...")
    url = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url, timeout=60)
    records = []
    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        if len(parts) < 26: continue
        try:
            y = int(parts[0])
            if y < 1980: continue
            kp_vals = [float(parts[7+i]) for i in range(8)]
            ap = float(parts[23])
            f107 = float(parts[25])
            records.append({"year": y, "month": int(parts[1]), "day": int(parts[2]),
                           "kp_max": max(kp_vals), "kp_mean": np.mean(kp_vals),
                           "ap": ap, "f107": f107 if f107 > 0 else np.nan})
        except: continue

    df = pd.DataFrame(records)
    yearly = df.groupby("year").agg(
        kp_mean=("kp_mean", "mean"),
        kp_max_year=("kp_max", "max"),
        ap_mean=("ap", "mean"),
        f107_mean=("f107", "mean"),
        n_storms=("kp_max", lambda x: (x >= 5).sum()),
    ).reset_index()
    print(f"  {len(yearly)} years")
    return yearly


# ─── Part 1: Vanuatu First Principles Prediction ────────────────────────────

def vanuatu_prediction():
    """
    From first principles, predict what should happen after:
    - M7.3 Vanuatu earthquake (March 30, 2026, 08:44 UTC, depth 115 km)
    - X1.5 solar flare (March 30, 02:47 UTC)
    - CME arriving March 31 (~10:40 UTC), speed 1845 km/s
    - G3-G4 geomagnetic storm forecast
    """
    print("\n" + "=" * 70)
    print("PART 1: FIRST PRINCIPLES PREDICTION — Vanuatu March 30-31, 2026")
    print("=" * 70)

    # Event parameters
    eq_lat, eq_lon = -15.4, 167.0  # M7.3 epicenter
    eq_depth = 115  # km (intermediate depth, within slab)
    eq_mag = 7.3
    flare_class = "X1.5"
    cme_speed = 1845  # km/s
    kp_forecast = 7  # G3 storm

    # Subsolar point at CME arrival (~10:40 UTC March 31)
    # March 31: solar declination ~ +4 deg
    # 10:40 UTC: subsolar lon = 180 - 15*10.67 = 20 deg E
    ss_lat, ss_lon = 4.0, 20.0

    # Angular distance: Vanuatu to subsolar point
    from transverse_volcanism import angular_distance, bearing
    dist_to_subsolar = angular_distance(eq_lat, eq_lon, ss_lat, ss_lon)
    bear_from_subsolar = bearing(ss_lat, ss_lon, eq_lat, eq_lon)

    print(f"""
EVENT:
  M7.3 earthquake: ({eq_lat}, {eq_lon}), depth {eq_depth} km
  X1.5 flare: 02:47 UTC March 30
  CME: {cme_speed} km/s, arriving ~10:40 UTC March 31
  Storm forecast: G3-G4 (Kp 7-9)

GEOMETRY:
  Subsolar point at CME arrival: ({ss_lat:.1f}, {ss_lon:.1f})
  Vanuatu distance from subsolar: {dist_to_subsolar:.1f} degrees
  Bearing from subsolar to Vanuatu: {bear_from_subsolar:.1f} degrees

PREDICTIONS FROM THE FRAMEWORK:
""")

    # 1. Jelly ball: Vanuatu is at ~135 degrees from subsolar
    # From our data: 120-150 deg band showed 0.82x (suppression) for wave arrival
    print(f"  1. JELLY BALL: Vanuatu at {dist_to_subsolar:.0f} deg from subsolar point")
    if dist_to_subsolar > 120:
        print(f"     -> In the FAR zone (120-180 deg)")
        print(f"     -> Our data showed 0.82x suppression in this zone for day +1-3")
        print(f"     -> PREDICTION: Vanuatu aftershock rate should be BELOW Omori baseline")
        print(f"        during the G3-G4 storm (March 31 - April 2)")
    elif dist_to_subsolar > 60:
        print(f"     -> In the MID zone (60-120 deg)")
        print(f"     -> Our data showed 1.36x enhancement at 68 deg for day +1-3")
        print(f"     -> PREDICTION: Enhanced aftershock rate during storm")

    # 2. Which regions SHOULD see enhancement?
    # The 60-90 degree ring from the subsolar point
    print(f"\n  2. ENHANCED REGIONS (60-90 deg from subsolar):")
    test_regions = {
        "Eastern Mediterranean": (35, 25),
        "Iran/Afghanistan": (33, 60),
        "Central Africa": (-5, 25),
        "South Atlantic": (-30, -20),
        "Eastern Brazil": (-15, -45),
        "North Atlantic/Azores": (38, -28),
    }
    for rname, (rlat, rlon) in test_regions.items():
        d = angular_distance(ss_lat, ss_lon, rlat, rlon)
        if 50 <= d <= 100:
            print(f"     {rname}: {d:.0f} deg from subsolar -> IN THE WAVE FRONT")

    # 3. Bearing-dependent: perpendicular to B preferentially
    print(f"\n  3. BEARING: The CME subsolar point is over equatorial Africa (~20E)")
    print(f"     B field there points roughly N-S (dipole)")
    print(f"     Perpendicular = E-W propagation preferred")
    print(f"     -> Watch for triggered seismicity to the EAST (Iran, India)")
    print(f"        and WEST (Atlantic, Caribbean) of the subsolar point")

    # 4. Volcanic squeeze
    print(f"\n  4. VOLCANIC SQUEEZE: Equatorial weak-B volcanic regions at risk:")
    print(f"     Our data showed Andes_N (Ecuador, 0 deg lat): 1.48x after M7.5+")
    print(f"     Indonesia (-3 deg): 1.21x")
    print(f"     These are at 60-90 deg from the subsolar point AND equatorial")
    print(f"     -> PREDICTION: Volcanic unrest (shallow swarms) at equatorial")
    print(f"        volcanic arcs within 7 days of CME arrival")

    # 5. The combined effect: earthquake + CME
    print(f"\n  5. COMBINED EFFECT: M7.3 + G3 storm hitting simultaneously")
    print(f"     The M7.3 creates its own jelly-ball stress wave")
    print(f"     The G3 storm creates a separate geomagnetic perturbation")
    print(f"     Where both waves OVERLAP (constructive interference):")
    print(f"     -> Enhanced triggering at the intersection")
    print(f"     Where they OPPOSE (destructive interference):")
    print(f"     -> Suppression")

    # 6. Omori prediction
    K, c, p = 10, 0.1, 1.1  # typical Omori parameters for M7.3
    print(f"\n  6. OMORI BASELINE (for comparison):")
    print(f"     Standard Omori: n(t) = K/(t+c)^p, K={K}, c={c}, p={p}")
    print(f"     Day 1: {K/(1+c)**p:.1f} aftershocks")
    print(f"     Day 2: {K/(2+c)**p:.1f} aftershocks")
    print(f"     Day 3: {K/(3+c)**p:.1f} aftershocks")
    print(f"     Day 7: {K/(7+c)**p:.1f} aftershocks")
    print(f"     If aftershock rate DEVIATES from Omori during the storm,")
    print(f"     that deviation IS the geomagnetic coupling signal")

    # 7. Schumann resonance prediction
    print(f"\n  7. SCHUMANN RESONANCE:")
    print(f"     f_Schumann = c / (2*pi*R_earth) ~ 7.83 Hz")
    print(f"     A G3 storm compresses the ionosphere -> R_eff decreases")
    print(f"     -> f_Schumann should INCREASE during the storm")
    print(f"     -> If Schumann monitors show a frequency shift coincident")
    print(f"        with the aftershock rate anomaly, that confirms the")
    print(f"        EM cavity is the coupling medium")


# ─── Part 2: Solar Cycle vs Global Seismicity ───────────────────────────────

def solar_cycle_analysis(eq_yearly, ss_yearly, kp_yearly):
    """
    Coarse-grained analysis: does the 11-year solar cycle
    correlate with global seismicity?

    The Hale cycle (22 years) includes the magnetic polarity flip.
    Solar max: ~2001, 2014, 2025
    Solar min: ~1996, 2008, 2019
    Polarity flip: at solar max (field reverses)
    """
    print("\n" + "=" * 70)
    print("PART 2: 11-YEAR SOLAR CYCLE vs GLOBAL SEISMICITY")
    print("=" * 70)

    # Merge all yearly data
    merged = pd.merge(eq_yearly, ss_yearly[["year", "ssn"]], on="year", how="inner")
    merged = pd.merge(merged, kp_yearly, on="year", how="inner")

    print(f"\nYears with complete data: {len(merged)} ({merged['year'].min()}-{merged['year'].max()})")

    # Basic correlations
    print("\nCorrelations (raw):")
    for eq_col, eq_label in [("n_m5", "N(M5+)"), ("n_m7", "N(M7+)"),
                              ("log_energy", "log(Energy)"), ("max_mag", "Max mag")]:
        for solar_col, solar_label in [("ssn", "Sunspot #"),
                                        ("f107_mean", "F10.7"),
                                        ("ap_mean", "Ap"),
                                        ("n_storms", "N(storms)")]:
            if solar_col in merged.columns and eq_col in merged.columns:
                valid = merged[[eq_col, solar_col]].dropna()
                if len(valid) > 5:
                    r, p = stats.pearsonr(valid[eq_col], valid[solar_col])
                    sig = "*" if p < 0.05 else " "
                    if abs(r) > 0.3 or p < 0.1:
                        print(f"  {eq_label:>12s} vs {solar_label:>12s}: r = {r:+.3f}, p = {p:.3f} {sig}")

    # Detrended correlation (remove long-term trend in earthquake detection)
    print("\nDetrended (remove linear trend from eq counts):")
    for eq_col, eq_label in [("n_m5", "N(M5+)"), ("n_m7", "N(M7+)")]:
        # Remove linear trend
        x = merged["year"].values
        y = merged[eq_col].values
        slope, intercept = np.polyfit(x, y, 1)
        detrended = y - (slope * x + intercept)

        for solar_col, solar_label in [("ssn", "Sunspot #"), ("f107_mean", "F10.7")]:
            if solar_col in merged.columns:
                valid_mask = ~np.isnan(merged[solar_col].values)
                r, p = stats.pearsonr(detrended[valid_mask],
                                       merged[solar_col].values[valid_mask])
                sig = "*" if p < 0.05 else " "
                print(f"  {eq_label:>12s} vs {solar_label:>12s}: r = {r:+.3f}, p = {p:.3f} {sig}")

    # Solar cycle phase binning
    # Define phases by sunspot number relative to cycle
    median_ssn = merged["ssn"].median()
    high_solar = merged[merged["ssn"] > median_ssn * 1.5]
    low_solar = merged[merged["ssn"] < median_ssn * 0.5]
    mid_solar = merged[(merged["ssn"] >= median_ssn * 0.5) &
                       (merged["ssn"] <= median_ssn * 1.5)]

    print(f"\nSolar phase analysis (median SSN = {median_ssn:.0f}):")
    for phase, subset in [("Solar MAX (SSN > 1.5x median)", high_solar),
                          ("Solar MID", mid_solar),
                          ("Solar MIN (SSN < 0.5x median)", low_solar)]:
        if len(subset) > 0:
            print(f"  {phase}: {len(subset)} years, "
                  f"mean M5+={subset['n_m5'].mean():.0f}, "
                  f"mean M7+={subset['n_m7'].mean():.1f}, "
                  f"mean max_mag={subset['max_mag'].mean():.1f}")

    # Polarity flip years (approximate: when the sun's polar field reverses)
    # Solar cycle 23 flip: ~2001, cycle 24 flip: ~2013-2014, cycle 25 flip: ~2024-2025
    flip_years = [2001, 2013, 2024]
    near_flip = merged[merged["year"].isin(
        [y + d for y in flip_years for d in [-1, 0, 1]])]
    far_flip = merged[~merged["year"].isin(
        [y + d for y in flip_years for d in [-1, 0, 1]])]

    print(f"\nMagnetic polarity flip years ({flip_years} +/- 1):")
    print(f"  Near flip ({len(near_flip)} years): mean M7+ = {near_flip['n_m7'].mean():.2f}")
    print(f"  Far from flip ({len(far_flip)} years): mean M7+ = {far_flip['n_m7'].mean():.2f}")
    if len(near_flip) > 2 and len(far_flip) > 2:
        _, p = stats.mannwhitneyu(near_flip["n_m7"], far_flip["n_m7"], alternative="two-sided")
        print(f"  Mann-Whitney p = {p:.3f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax = axes[0]
    ax.bar(merged["year"], merged["n_m5"], color="steelblue", alpha=0.7, label="N(M5+)")
    ax2 = ax.twinx()
    ax2.plot(merged["year"], merged["ssn"], 'o-', color="orange", linewidth=2, label="Sunspot #")
    ax.set_ylabel("Earthquake count (M5+)")
    ax2.set_ylabel("Sunspot number", color="orange")
    ax.set_title("Global Seismicity vs Solar Cycle")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1]
    ax.bar(merged["year"], merged["n_m7"], color="#e78ac3", alpha=0.7, label="N(M7+)")
    ax2 = ax.twinx()
    ax2.plot(merged["year"], merged["ssn"], 'o-', color="orange", linewidth=2, alpha=0.5)
    ax.set_ylabel("M7+ earthquake count")
    ax2.set_ylabel("Sunspot number", color="orange")
    ax.legend(loc="upper left")
    # Mark polarity flips
    for fy in flip_years:
        ax.axvline(fy, color="red", linestyle="--", alpha=0.5)
    ax.text(flip_years[0], ax.get_ylim()[1]*0.9, "flip", color="red", fontsize=9)

    ax = axes[2]
    ax.plot(merged["year"], merged["log_energy"], 'o-', color="green", linewidth=2, label="log(Energy)")
    ax2 = ax.twinx()
    ax2.plot(merged["year"], merged["f107_mean"], 's-', color="red", linewidth=1.5, alpha=0.7, label="F10.7")
    ax.set_ylabel("log(Seismic energy)")
    ax2.set_ylabel("F10.7 solar flux", color="red")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "solar_cycle_seismicity.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'solar_cycle_seismicity.png'}")

    return merged


# ─── Part 3: High-speed solar wind streams (coronal holes) ──────────────────

def solar_wind_analysis(kp_yearly):
    """
    Coronal holes produce recurrent high-speed solar wind streams.
    During solar minimum, these are the dominant geomagnetic driver.
    They produce moderate but SUSTAINED Kp elevation (3-5) for days,
    unlike CMEs which produce impulsive spikes.

    The framework predicts: sustained forcing should drive the system
    toward the KT boundary more effectively than impulses, because
    it has time to thermalize. The 27-day solar rotation recurrence
    should show up as a 27-day periodicity in seismicity.
    """
    print("\n" + "=" * 70)
    print("PART 3: 27-DAY SOLAR ROTATION IN SEISMICITY")
    print("=" * 70)

    # Download daily data
    print("Downloading daily Kp and earthquakes...")
    url_kp = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"
    resp = requests.get(url_kp, timeout=60)
    records = []
    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip(): continue
        parts = line.split()
        if len(parts) < 25: continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 2000: continue
            kp_vals = [float(parts[7+i]) for i in range(8)]
            records.append({"year":y, "month":m, "day":d,
                           "kp_mean": np.mean(kp_vals), "ap": float(parts[23])})
        except: continue
    kp_daily = pd.DataFrame(records)
    kp_daily["date"] = pd.to_datetime(kp_daily[["year","month","day"]])
    kp_daily["day_number"] = ((kp_daily["date"] - pd.Timestamp(INIT_DATE)).dt.days).values

    # Daily earthquake counts
    url_eq = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    eq_dfs = []
    for year in range(2000, 2027):
        try:
            resp = requests.get(url_eq, params={
                "format":"csv", "starttime":f"{year}-01-01", "endtime":f"{year}-12-31",
                "minmagnitude": 5.0, "limit": 20000,
            }, timeout=60)
            resp.raise_for_status()
            eq_dfs.append(pd.read_csv(StringIO(resp.text)))
        except: pass
    eq_all = pd.concat(eq_dfs, ignore_index=True)
    eq_all["time_parsed"] = pd.to_datetime(eq_all["time"], utc=True).dt.tz_localize(None)
    eq_all["day_number"] = ((eq_all["time_parsed"] - pd.Timestamp(INIT_DATE)).dt.days).values

    eq_daily = eq_all.groupby("day_number").agg(n=("mag","count")).reset_index()
    merged = pd.merge(kp_daily[["day_number","kp_mean","ap"]], eq_daily, on="day_number", how="left")
    merged["n"] = merged["n"].fillna(0)

    # Spectral analysis: look for 27-day periodicity
    print("\nSpectral analysis: looking for 27-day solar rotation period...")

    # Detrend
    eq_series = merged["n"].values
    eq_detrended = eq_series - np.convolve(eq_series, np.ones(365)/365, mode='same')

    kp_series = merged["kp_mean"].values
    kp_detrended = kp_series - np.convolve(kp_series, np.ones(365)/365, mode='same')

    # Lomb-Scargle periodogram (handles any gaps)
    from scipy.signal import periodogram

    freqs_eq, psd_eq = periodogram(eq_detrended, fs=1.0)  # 1 sample/day
    freqs_kp, psd_kp = periodogram(kp_detrended, fs=1.0)

    # Look at periods 20-35 days
    period_mask = (1.0/freqs_eq > 20) & (1.0/freqs_eq < 35) & (freqs_eq > 0)
    if np.any(period_mask):
        peak_freq_eq = freqs_eq[period_mask][np.argmax(psd_eq[period_mask])]
        peak_period_eq = 1.0 / peak_freq_eq
        print(f"  Earthquake peak period (20-35 day band): {peak_period_eq:.1f} days")

    period_mask_kp = (1.0/freqs_kp > 20) & (1.0/freqs_kp < 35) & (freqs_kp > 0)
    if np.any(period_mask_kp):
        peak_freq_kp = freqs_kp[period_mask_kp][np.argmax(psd_kp[period_mask_kp])]
        peak_period_kp = 1.0 / peak_freq_kp
        print(f"  Kp peak period (20-35 day band): {peak_period_kp:.1f} days")

    # Cross-correlation at 27-day lag
    max_lag = 40
    xcorr = np.correlate(kp_detrended[:len(eq_detrended)],
                          eq_detrended, mode='full')
    xcorr = xcorr[len(eq_detrended)-max_lag-1:len(eq_detrended)+max_lag]
    lags = np.arange(-max_lag, max_lag+1)
    xcorr_norm = xcorr / (np.std(kp_detrended[:len(eq_detrended)]) *
                           np.std(eq_detrended) * len(eq_detrended))

    print(f"\n  Cross-correlation Kp vs earthquakes:")
    for lag in [0, 1, 2, 3, 13, 14, 27, 28]:
        idx = lag + max_lag
        if 0 <= idx < len(xcorr_norm):
            marker = " <-- solar rotation" if lag == 27 else ""
            marker = " <-- half rotation" if lag == 14 else marker
            print(f"    Lag {lag:+3d} days: r = {xcorr_norm[idx]:+.4f}{marker}")

    # Plot periodogram
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    period_range = (1.0/freqs_eq > 5) & (1.0/freqs_eq < 100) & (freqs_eq > 0)
    ax.semilogy(1.0/freqs_eq[period_range], psd_eq[period_range], color="steelblue", label="Earthquakes")
    ax.semilogy(1.0/freqs_kp[period_range], psd_kp[period_range] * np.max(psd_eq[period_range])/np.max(psd_kp[period_range]),
                color="orange", alpha=0.7, label="Kp (scaled)")
    ax.axvline(27.3, color="red", linestyle="--", alpha=0.7, label="27.3 day (solar rotation)")
    ax.axvline(13.6, color="green", linestyle="--", alpha=0.5, label="13.6 day (half rotation)")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Power spectral density")
    ax.set_title("Periodogram: Earthquakes and Kp Index\nLooking for 27-day solar rotation signal")
    ax.legend()
    ax.set_xlim(5, 100)

    ax = axes[1]
    ax.plot(lags, xcorr_norm, color="steelblue", linewidth=1.5)
    ax.axvline(27, color="red", linestyle="--", alpha=0.5, label="27 days")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Normalized cross-correlation")
    ax.set_title("Cross-correlation: Kp index vs earthquake rate")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "solar_rotation_seismicity.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'solar_rotation_seismicity.png'}")


# ─── Part 4: Schumann Resonance ─────────────────────────────────────────────

def schumann_analysis():
    """
    Theoretical framework connection:
    The Schumann resonance f = c/(2*pi*R) ~ 7.83 Hz is the
    fundamental mode of the Earth-ionosphere cavity.

    In the KT framework:
    - The cavity is a Kuramoto oscillator
    - The ionosphere height sets R_eff
    - Geomagnetic storms lower the ionosphere -> increase f
    - The Schumann frequency is the ORDER PARAMETER of the cavity
    - At KT criticality (J = J_c = 2/pi), the order parameter
      should show maximum fluctuations

    Connection to 2/pi:
    f_S = c / (2*pi*R) implies the cavity resonates when
    circumference = wavelength. The 2*pi is geometric (winding).
    J_c = 2/pi is the critical coupling where phase coherence
    emerges. Both involve the ratio of winding to geometry.
    """
    print("\n" + "=" * 70)
    print("PART 4: SCHUMANN RESONANCE AND THE KT ORDER PARAMETER")
    print("=" * 70)

    c = 3e8  # m/s
    R_earth = 6.371e6  # m
    f_schumann = c / (2 * PI * R_earth)

    print(f"""
  Schumann fundamental: f = c/(2*pi*R) = {f_schumann:.2f} Hz
  Observed: ~7.83 Hz (the cavity has losses, conductivity corrections)

  KT CONNECTION:
  The ionosphere-Earth cavity is a resonator with quality factor Q.
  During geomagnetic storms:
    - Ionosphere drops from ~85 km to ~70 km (compression)
    - R_eff decreases -> f increases by ~0.1-0.3 Hz
    - Q decreases (more dissipation from charged particles)
    - Amplitude increases (more energy injected into cavity)

  The framework says this is the SAME coupling:
    [F, nabla F] at the ionospheric boundary drives both:
    - Schumann frequency shift (the EM cavity response)
    - Crustal stress perturbation (the elastic response)

  If earthquake lights and volcanic lightning are the grade-0
  (EM) projection of [F, nabla F] at the rupture, then they
  should occur at frequencies near the Schumann resonance or
  its harmonics: {f_schumann:.2f}, {2*f_schumann:.2f}, {3*f_schumann:.2f} Hz...

  TESTABLE PREDICTION:
  Earthquake precursor EM signals (if real) should show power
  at Schumann harmonics, not at arbitrary frequencies.
  Volcanic lightning frequency content should also cluster
  near Schumann harmonics.

  The 2/pi in J_c = 2/pi and the 2*pi in f = c/(2*pi*R)
  are the SAME ratio: the competition between topological
  winding (2*pi for a full phase cycle) and the critical
  coupling (2/pi where vortex energy = vortex entropy).

  This means: the Schumann resonance IS the KT order parameter
  of the Earth's electromagnetic cavity. When J_EM = J_c = 2/pi
  (at the ionospheric boundary), the cavity is at the phase
  transition between conducting (ordered, ionosphere intact)
  and insulating (disordered, ionosphere disrupted).

  Earthquake lights: the grade-0 projection of [F, nabla F]
  at the crustal boundary where J crosses J_c mechanically.
  The EM emission frequency should be set by the local cavity
  geometry — effectively a local Schumann mode.

  Volcanic lightning: same mechanism but at the volcanic vent,
  where the charged particle plume creates a local cavity
  resonator. The lightning frequency should match the vent
  geometry's fundamental mode.
""")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FIRST PRINCIPLES + SOLAR CYCLE + SCHUMANN")
    print("=" * 70)

    # Part 1: Vanuatu prediction
    vanuatu_prediction()

    # Part 2: Solar cycle
    eq_yearly = download_earthquakes_yearly()
    ss_yearly = download_sunspots_yearly()
    kp_yearly = download_kp_yearly()
    solar_cycle_analysis(eq_yearly, ss_yearly, kp_yearly)

    # Part 3: 27-day rotation
    solar_wind_analysis(kp_yearly)

    # Part 4: Schumann theory
    schumann_analysis()

    print("\n" + "=" * 70)
    print("COMPLETE FRAMEWORK PICTURE")
    print("=" * 70)
    print("""
The Earth is a resonating sphere in a heliospheric cavity:

  SCALE          OSCILLATOR              COUPLING        KT BOUNDARY
  -----          ----------              --------        -----------
  Solar cycle    Dynamo (22 yr Hale)     B_helio         Polarity flip
  CME/CIR        Magnetosphere           Kp, Dst         G3+ storm
  27-day         Solar rotation          Recurrent HSS   Coronal hole facing
  Schumann       EM cavity (7.83 Hz)     Ionosphere J    J_c = 2/pi
  Seismic        Crustal oscillators     Fault coupling   Earthquake
  Volcanic       Magma conduits          Pressure         Eruption
  EM emission    Local cavity modes      [F, nabla F]_0   Earthquake lights

All governed by ONE equation: dF/dt = v^2 nabla^2 F + [F, nabla F]
The commutator is everywhere. The coupling is always 1.
The KT transition at J_c = 2/pi determines when each oscillator
transitions from ordered to disordered.

The jelly ball is the Earth. The sun hits it. It rings.
Where it rings depends on the geometry of the magnetic field.
Where it breaks depends on where J ~ J_c.
Where it squeezes depends on where the crust is weak.
The lights are the commutator becoming visible.
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
