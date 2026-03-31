#!/usr/bin/env python3
"""
Bz Split: The Reconnection Switch
====================================
Southward Bz → magnetic reconnection → aurora → coupling ON
Northward Bz → deflection → no reconnection → coupling OFF

This is the sharpest test of the framework. If EVERY signal
we found (three grades, jelly ball, bearing, volcanic squeeze)
only appears during southward Bz and vanishes during northward,
the coupling goes through reconnection at the magnetopause —
which IS the commutator [F, ∇F] at the boundary where
Earth's field meets the solar wind field.

Uses OMNI hourly data: Bz, V_sw, Dst, AE all together.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import datetime as dt
import sys, os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"
INIT_DATE = dt.datetime(2000, 1, 1)


def load_data():
    print("Loading cached data...")
    omni = pd.read_csv(DATA_DIR / "omni_hourly.csv", parse_dates=["datetime"])
    eq = pd.read_csv(DATA_DIR / "earthquakes_m4.5.csv", parse_dates=["time_parsed"])
    flares = pd.read_csv(DATA_DIR / "solar_flares.csv", parse_dates=["beginTime","peakTime","endTime"])
    print(f"  OMNI: {len(omni)} hourly records")
    print(f"  Earthquakes: {len(eq)} M4.5+ events")
    print(f"  Flares: {len(flares)} events")
    return omni, eq, flares


def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def subsolar_point(dt_utc):
    doy = dt_utc.timetuple().tm_yday
    decl = -23.44 * np.cos(np.radians(360 / 365 * (doy + 10)))
    hour = dt_utc.hour + dt_utc.minute / 60.0
    lon = 180.0 - 15.0 * hour
    if lon < -180: lon += 360
    if lon > 180: lon -= 360
    return decl, lon


# ═══════════════════════════════════════════════════════════════════════
# 1. Bz SPLIT: Three-Grade Test
# ═══════════════════════════════════════════════════════════════════════

def bz_three_grade(omni, eq, flares):
    """
    Repeat the three-grade test (hour 0, +18, +41) but split by
    Bz polarity during the 24h BEFORE the flare.
    Southward Bz (negative) = reconnection primed.
    Northward Bz (positive) = magnetosphere closed.
    """
    print("\n=== Bz Split: Three-Grade Test ===")

    m5_flares = flares[flares["class_numeric"] >= 0.5].dropna(subset=["peakTime"]).copy()
    eq_times = eq["time_parsed"].values.astype('datetime64[h]')

    def count_window(ref_time, h_start, h_end):
        t0 = np.datetime64(ref_time, 'h')
        return int(np.sum((eq_times >= t0 + np.timedelta64(h_start,'h')) &
                          (eq_times < t0 + np.timedelta64(h_end,'h'))))

    # For each flare, get mean Bz in the 24h before
    results = []
    for _, fl in m5_flares.iterrows():
        ft = fl["peakTime"]
        # Get OMNI Bz in 24h window before flare
        pre_window = omni[(omni["datetime"] >= ft - pd.Timedelta(hours=24)) &
                          (omni["datetime"] < ft)]
        bz_mean = pre_window["bz_gse"].mean()
        ae_mean = pre_window["ae"].mean()
        dst_mean = pre_window["dst"].mean()

        if np.isnan(bz_mean):
            continue

        n_g0 = count_window(ft, 0, 6)
        n_g2 = count_window(ft, 24, 72)
        n_bg = count_window(ft, -168, -24)
        bg_6h = n_bg / (144/6)
        bg_48h = n_bg / (144/48)

        results.append({
            "flare_time": ft, "class_num": fl["class_numeric"],
            "bz_mean": bz_mean, "ae_mean": ae_mean, "dst_mean": dst_mean,
            "n_g0": n_g0, "n_g2": n_g2, "bg_6h": bg_6h, "bg_48h": bg_48h,
        })

    rdf = pd.DataFrame(results)
    print(f"Flares with Bz data: {len(rdf)}")

    # Split by Bz polarity
    south = rdf[rdf["bz_mean"] < 0]
    north = rdf[rdf["bz_mean"] >= 0]

    print(f"Southward Bz (reconnection ON): {len(south)}")
    print(f"Northward Bz (reconnection OFF): {len(north)}")

    for label, subset in [("SOUTHWARD Bz (reconnection)", south),
                          ("NORTHWARD Bz (no reconnection)", north)]:
        if len(subset) < 20:
            continue
        r0 = subset["n_g0"].mean() / max(subset["bg_6h"].mean(), 0.001)
        r2 = subset["n_g2"].mean() / max(subset["bg_48h"].mean(), 0.001)
        print(f"\n  {label} ({len(subset)} flares):")
        print(f"    Grade-0 (0-6h):   {subset['n_g0'].mean():.2f} vs bg {subset['bg_6h'].mean():.2f} = {r0:.3f}x")
        print(f"    Grade-2 (24-72h): {subset['n_g2'].mean():.2f} vs bg {subset['bg_48h'].mean():.2f} = {r2:.3f}x")

    # Also split by Bz magnitude (strong south vs weak south)
    strong_south = rdf[rdf["bz_mean"] < -2]
    weak_south = rdf[(rdf["bz_mean"] >= -2) & (rdf["bz_mean"] < 0)]

    for label, subset in [("STRONG south (Bz < -2 nT)", strong_south),
                          ("WEAK south (-2 < Bz < 0)", weak_south)]:
        if len(subset) < 15:
            continue
        r0 = subset["n_g0"].mean() / max(subset["bg_6h"].mean(), 0.001)
        r2 = subset["n_g2"].mean() / max(subset["bg_48h"].mean(), 0.001)
        print(f"\n  {label} ({len(subset)} flares):")
        print(f"    Grade-0: {r0:.3f}x    Grade-2: {r2:.3f}x")

    # Hourly epoch split by Bz
    hour_window = np.arange(-24, 73)

    def hourly_epoch(subset, label):
        hourly = np.zeros((len(subset), len(hour_window)))
        for i, (_, row) in enumerate(subset.iterrows()):
            ft = row["flare_time"]
            for j, h in enumerate(hour_window):
                hourly[i, j] = count_window(ft, h, h+1)
        return np.mean(hourly, axis=0)

    rate_south = hourly_epoch(south, "South")
    rate_north = hourly_epoch(north, "North")
    bg_south = np.mean(rate_south[:24])
    bg_north = np.mean(rate_north[:24])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    kernel = np.ones(3)/3

    ax = axes[0]
    s_smooth = np.convolve(rate_south/max(bg_south,0.001), kernel, 'same')
    n_smooth = np.convolve(rate_north/max(bg_north,0.001), kernel, 'same')
    ax.plot(hour_window, s_smooth, color="#e41a1c", linewidth=2.5,
            label=f"Southward Bz (N={len(south)}) — reconnection ON")
    ax.plot(hour_window, n_smooth, color="#377eb8", linewidth=2.5,
            label=f"Northward Bz (N={len(north)}) — reconnection OFF")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="orange", linewidth=2, alpha=0.5, label="Flare")
    ax.axvspan(24, 72, alpha=0.08, color="green")
    ax.set_ylabel("Earthquake rate / background")
    ax.set_title("THE RECONNECTION SWITCH\n"
                 "Seismicity after M5+ flares, split by pre-flare Bz polarity")
    ax.legend(fontsize=11)
    ax.set_xlim(-24, 72)

    # Raw bars comparison
    ax = axes[1]
    w = 0.35
    x = np.arange(5)
    labels_bar = ["Grade-0\n(0-6h)", "Grade-4\n(12-24h)", "Grade-2\n(24-48h)",
                  "Grade-2\n(48-72h)", "All\n(0-72h)"]
    windows = [(0,6), (12,24), (24,48), (48,72), (0,72)]

    south_ratios = []
    north_ratios = []
    for h_start, h_end in windows:
        s_rate = np.mean([count_window(row["flare_time"], h_start, h_end)
                          for _, row in south.iterrows()])
        s_bg = np.mean([row["bg_6h"] * (h_end-h_start)/6 for _, row in south.iterrows()])
        n_rate = np.mean([count_window(row["flare_time"], h_start, h_end)
                          for _, row in north.iterrows()])
        n_bg = np.mean([row["bg_6h"] * (h_end-h_start)/6 for _, row in north.iterrows()])
        south_ratios.append(s_rate / max(s_bg, 0.001))
        north_ratios.append(n_rate / max(n_bg, 0.001))

    ax.bar(x - w/2, south_ratios, w, color="#e41a1c", alpha=0.7, label="Southward Bz")
    ax.bar(x + w/2, north_ratios, w, color="#377eb8", alpha=0.7, label="Northward Bz")
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel("Earthquake rate / background")
    ax.set_title("Grade-by-Grade Comparison: Southward vs Northward Bz")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "bz_reconnection_switch.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'bz_reconnection_switch.png'}")

    return rdf


# ═══════════════════════════════════════════════════════════════════════
# 2. AE INDEX: Aurora Timing vs Seismic Timing
# ═══════════════════════════════════════════════════════════════════════

def ae_aurora_correlation(omni, eq):
    """
    The AE index measures auroral electrojet current — a direct
    measure of reconnection and energy input to the ionosphere.

    Test: does the HOURLY AE index correlate with hourly earthquake
    rate at various lags?

    Framework: aurora (grade-0 at magnetopause) and earthquakes
    (grade-2 at crust) should be driven by the same commutator
    but with different propagation times.
    """
    print("\n=== AE Index: Aurora-Earthquake Hour-by-Hour ===")

    # Build hourly earthquake count
    eq_hourly = eq.copy()
    eq_hourly["hour_bin"] = eq_hourly["time_parsed"].dt.floor("h")
    eq_counts = eq_hourly.groupby("hour_bin").agg(n=("magnitude","count")).reset_index()
    eq_counts.columns = ["datetime", "n_eq"]

    # Merge with OMNI
    omni_clean = omni.dropna(subset=["ae", "bz_gse"]).copy()
    merged = pd.merge(omni_clean[["datetime","ae","bz_gse","dst","v_sw"]],
                      eq_counts, on="datetime", how="left")
    merged["n_eq"] = merged["n_eq"].fillna(0)

    print(f"Merged hourly records: {len(merged)}")

    # Lag correlation: AE → earthquakes
    print("\nCross-correlation: AE index -> earthquake rate")
    print(f"  {'Lag':>6s}  {'r(AE,EQ)':>10s}  {'r(Bz,EQ)':>10s}  {'r(Dst,EQ)':>10s}  {'r(Vsw,EQ)':>10s}")

    max_lag = 72
    ae_corrs, bz_corrs = [], []
    for lag in range(0, max_lag + 1, 3):
        ae_vals = merged["ae"].values[:-max(lag,1)]
        bz_vals = merged["bz_gse"].values[:-max(lag,1)]
        dst_vals = merged["dst"].values[:-max(lag,1)]
        vsw_vals = merged["v_sw"].values[:-max(lag,1)]
        eq_vals = merged["n_eq"].shift(-lag).values[:-max(lag,1)]

        mask = ~(np.isnan(ae_vals) | np.isnan(eq_vals) | np.isnan(bz_vals))
        if mask.sum() > 100:
            r_ae = np.corrcoef(ae_vals[mask], eq_vals[mask])[0,1]
            r_bz = np.corrcoef(bz_vals[mask], eq_vals[mask])[0,1]
            r_dst = np.corrcoef(dst_vals[mask & ~np.isnan(dst_vals)],
                                eq_vals[mask & ~np.isnan(dst_vals)])[0,1] if np.sum(mask & ~np.isnan(dst_vals)) > 100 else np.nan
            r_vsw = np.corrcoef(vsw_vals[mask & ~np.isnan(vsw_vals)],
                                eq_vals[mask & ~np.isnan(vsw_vals)])[0,1] if np.sum(mask & ~np.isnan(vsw_vals)) > 100 else np.nan
            ae_corrs.append((lag, r_ae))
            bz_corrs.append((lag, r_bz))

            marker = ""
            if lag == 0: marker = " <-- simultaneous"
            if lag == 18: marker = " <-- 18h (grade-4?)"
            if lag == 42: marker = " <-- 42h (grade-2 peak)"
            print(f"  {lag:>4d}h  {r_ae:>+10.5f}  {r_bz:>+10.5f}  {r_dst:>+10.5f}  {r_vsw:>+10.5f}{marker}")

    # Storm-time analysis: AE > 500 nT = substorm
    substorms = omni[omni["ae"] > 500].copy()
    substorm_hours = substorms["datetime"].values.astype('datetime64[h]')
    eq_times = eq["time_parsed"].values.astype('datetime64[h]')

    # Deduplicate substorms (keep first per 12h window)
    if len(substorm_hours) > 0:
        deduped = [substorm_hours[0]]
        for t in substorm_hours[1:]:
            if (t - deduped[-1]) > np.timedelta64(12, 'h'):
                deduped.append(t)
        substorm_hours = np.array(deduped)

    print(f"\nSubstorms (AE > 500 nT, 12h dedup): {len(substorm_hours)}")

    # Superposed epoch: earthquake rate around substorms
    window = np.arange(-24, 49)
    stacked = np.zeros((len(substorm_hours), len(window)))
    for i, t0 in enumerate(substorm_hours):
        for j, h in enumerate(window):
            t_start = t0 + np.timedelta64(h, 'h')
            t_end = t0 + np.timedelta64(h+1, 'h')
            stacked[i, j] = np.sum((eq_times >= t_start) & (eq_times < t_end))

    mean_rate = np.mean(stacked, axis=0)
    bg = np.mean(mean_rate[:24])

    print(f"\nSuperposed epoch around substorms:")
    print(f"Background: {bg:.3f}/hr")
    for h_idx in [20, 21, 22, 23, 24, 25, 26, 27, 30, 36, 42, 48]:
        h = window[h_idx]
        ratio = mean_rate[h_idx] / max(bg, 0.001)
        marker = " <-- SUBSTORM" if h == 0 else ""
        marker = " <-- +18h" if h == 18 else marker
        marker = " <-- +42h" if h == 42 else marker
        print(f"  {h:+4d}h: {mean_rate[h_idx]:.3f} ({ratio:.2f}x){marker}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    lags_plot = [l for l, _ in ae_corrs]
    ae_r_plot = [r for _, r in ae_corrs]
    bz_r_plot = [r for _, r in bz_corrs]
    ax.plot(lags_plot, ae_r_plot, 'o-', color="#e41a1c", linewidth=2, label="AE → earthquakes")
    ax.plot(lags_plot, bz_r_plot, 's-', color="#377eb8", linewidth=2, label="Bz → earthquakes")
    noise = 2 / np.sqrt(len(merged))
    ax.axhline(noise, color="gray", linestyle="--", alpha=0.5, label=f"2-sigma: {noise:.5f}")
    ax.axhline(-noise, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", alpha=0.3)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("Hourly Correlation: Solar Wind → Earthquake Rate\n"
                 "AE = aurora, Bz = reconnection geometry")
    ax.legend()

    ax = axes[1]
    ax.bar(window, mean_rate, width=0.8, color="steelblue", alpha=0.6)
    ax.axhline(bg, color="red", linestyle="--", alpha=0.5, label=f"Background: {bg:.3f}/hr")
    ax.axvline(0, color="orange", linewidth=2, alpha=0.8, label="Substorm (AE > 500)")
    ax.set_xlabel("Hours relative to substorm onset")
    ax.set_ylabel("Mean earthquake rate (M4.5+/hr)")
    ax.set_title(f"Superposed Epoch: {len(substorm_hours)} Substorms")
    ax.legend()
    ax.set_xlim(-24, 48)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "ae_aurora_correlation.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'ae_aurora_correlation.png'}")


# ═══════════════════════════════════════════════════════════════════════
# 3. Bz SPLIT: Jelly Ball Spatial Pattern
# ═══════════════════════════════════════════════════════════════════════

def bz_jelly_ball(omni, eq):
    """
    Repeat the jelly ball analysis but split by Bz polarity
    during the impulse. If the spatial pattern only appears
    for southward Bz, it confirms reconnection as the gate.
    """
    print("\n=== Bz Split: Jelly Ball Spatial Pattern ===")

    # Find geomagnetic impulses from OMNI (AE spike > 500 with Bz data)
    impulses = omni[(omni["ae"] > 500) & omni["bz_gse"].notna()].copy()

    # Deduplicate
    days = impulses["day_number"].values
    if len(days) == 0:
        print("No impulses with AE > 500 found")
        return
    filtered = [0]
    for i in range(1, len(days)):
        if days[i] - days[filtered[-1]] >= 3:
            filtered.append(i)
    impulses = impulses.iloc[filtered].copy()

    # Split by Bz
    south_imp = impulses[impulses["bz_gse"] < 0]
    north_imp = impulses[impulses["bz_gse"] >= 0]
    print(f"Impulses: {len(south_imp)} southward, {len(north_imp)} northward")

    bins = np.arange(0, 181, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    solid_angles = 2 * np.pi * np.abs(
        np.cos(np.radians(bins[:-1])) - np.cos(np.radians(bins[1:])))

    def compute_jelly(imp_subset, eq_df, label):
        wave_counts = np.zeros((len(imp_subset), len(bins)-1))
        bg_counts = np.zeros((len(imp_subset), len(bins)-1))

        for idx, (_, imp) in enumerate(imp_subset.iterrows()):
            imp_day = imp["day_number"]
            imp_dt = imp["datetime"]
            ss_lat, ss_lon = subsolar_point(imp_dt)

            eq_wave = eq_df[(eq_df["day_number"] >= imp_day + 1) &
                            (eq_df["day_number"] <= imp_day + 3)]
            if len(eq_wave) > 0:
                dists = angular_distance(ss_lat, ss_lon,
                                         eq_wave["latitude"].values,
                                         eq_wave["longitude"].values)
                counts, _ = np.histogram(dists, bins=bins)
                wave_counts[idx] = counts / 3.0

            eq_bg = eq_df[(eq_df["day_number"] >= imp_day - 7) &
                          (eq_df["day_number"] <= imp_day - 3)]
            if len(eq_bg) > 0:
                dists = angular_distance(ss_lat, ss_lon,
                                         eq_bg["latitude"].values,
                                         eq_bg["longitude"].values)
                counts, _ = np.histogram(dists, bins=bins)
                bg_counts[idx] = counts / 5.0

        mean_wave = np.mean(wave_counts, axis=0) / solid_angles
        mean_bg = np.mean(bg_counts, axis=0) / solid_angles
        ratio = np.where(mean_bg > 0, mean_wave / mean_bg, 1.0)
        return ratio

    # Need M5+ for jelly ball (to get enough signal)
    eq_m5 = eq[eq["magnitude"] >= 5.0].copy()

    ratio_south = compute_jelly(south_imp, eq_m5, "South Bz")
    ratio_north = compute_jelly(north_imp, eq_m5, "North Bz")

    print(f"\n{'Angle':>6s}  {'South Bz':>10s}  {'North Bz':>10s}")
    for i, c in enumerate(bin_centers):
        print(f"  {c:>4.0f}   {ratio_south[i]:>9.2f}x  {ratio_north[i]:>9.2f}x")

    near_s = np.mean(ratio_south[:3])
    near_n = np.mean(ratio_north[:3])
    mid_s = np.mean(ratio_south[3:6])
    mid_n = np.mean(ratio_north[3:6])
    far_s = np.mean(ratio_south[6:])
    far_n = np.mean(ratio_north[6:])

    print(f"\n  Zone summary:")
    print(f"    Near (0-60):   South={near_s:.3f}, North={near_n:.3f}")
    print(f"    Mid (60-120):  South={mid_s:.3f}, North={mid_n:.3f}")
    print(f"    Far (120-180): South={far_s:.3f}, North={far_n:.3f}")
    print(f"\n  Framework predicts: South should show stronger jelly ball pattern")
    print(f"    (more suppression near, more enhancement mid)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_centers, ratio_south, 'o-', color="#e41a1c", linewidth=2.5,
            label=f"Southward Bz (N={len(south_imp)})")
    ax.plot(bin_centers, ratio_north, 's-', color="#377eb8", linewidth=2.5,
            label=f"Northward Bz (N={len(north_imp)})")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(bin_centers, 0.8, 1.2, alpha=0.08, color="gray")
    ax.set_xlabel("Angular distance from subsolar point (degrees)")
    ax.set_ylabel("Seismicity ratio (wave / background)")
    ax.set_title("JELLY BALL: Southward Bz (reconnection) vs Northward Bz\n"
                 "Day +1-3 after AE > 500 substorm onset")
    ax.legend(fontsize=12)
    ax.set_ylim(0.4, 2.0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bz_jelly_ball.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'bz_jelly_ball.png'}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Bz RECONNECTION SWITCH + AURORA CORRELATION")
    print("The sharpest test: does everything vanish for northward Bz?")
    print("=" * 70)

    omni, eq, flares = load_data()

    # Quick OMNI stats
    bz = omni["bz_gse"].dropna()
    ae = omni["ae"].dropna()
    print(f"\nOMNI Bz: mean={bz.mean():.2f}, std={bz.std():.2f}, "
          f"south fraction={( bz < 0).mean():.1%}")
    print(f"OMNI AE: mean={ae.mean():.0f}, max={ae.max():.0f}, "
          f"substorms (>500)={( ae > 500).mean():.1%}")

    bz_three_grade(omni, eq, flares)
    ae_aurora_correlation(omni, eq)
    bz_jelly_ball(omni, eq)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If the signals appear ONLY for southward Bz:
  -> Coupling goes through magnetic reconnection
  -> Reconnection IS the commutator [F_earth, F_solar] at the magnetopause
  -> The aurora is the grade-0 visible projection of the same commutator
  -> The seismic response is the grade-2 mechanical projection

If the signals appear for BOTH polarities:
  -> Coupling is compressive (ram pressure), not reconnective
  -> This would weaken the commutator interpretation

If the signals are STRONGER for southward:
  -> Both mechanisms contribute, but reconnection dominates
  -> Consistent with the full geometric product F*nabla(F)
     having both scalar (compression) and bivector (reconnection) parts
""")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
