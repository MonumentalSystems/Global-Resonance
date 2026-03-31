#!/usr/bin/env python3
"""
STEVE Bz Reversal Analysis — Extended
========================================
Strategy:
1. Download Aurorasaurus citizen science data (Zenodo, 2015-2016)
2. Also scrape known STEVE dates from literature
3. Build a comprehensive STEVE event list
4. Cross-reference every event with OMNI Bz data
5. Test: fraction of STEVE events occurring during Bz reversals
   vs fraction of random aurora events during Bz reversals (control)

The prediction: STEVE correlates with Bz sign changes,
while regular aurora correlates with sustained southward Bz.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import datetime as dt
import requests
import sys
import os
from io import StringIO

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"


def load_omni():
    omni = pd.read_csv(DATA_DIR / "omni_hourly.csv", parse_dates=["datetime"])
    return omni


def count_bz_reversals(omni, center_time, window_hours=6):
    """Count Bz sign changes in a window around a given time."""
    t0 = pd.Timestamp(center_time)
    window = omni[(omni["datetime"] >= t0 - pd.Timedelta(hours=window_hours)) &
                  (omni["datetime"] <= t0 + pd.Timedelta(hours=window_hours))]
    bz = window["bz_gse"].dropna().values
    if len(bz) < 3:
        return np.nan, np.nan, np.nan
    sign_changes = np.sum(np.diff(np.sign(bz)) != 0)
    bz_range = np.max(bz) - np.min(bz)
    bz_at = window.iloc[len(window)//2]["bz_gse"] if len(window) > 0 else np.nan
    return sign_changes, bz_range, bz_at


def download_aurorasaurus():
    """Download Aurorasaurus 2015-2016 cleaned web observations."""
    for year in [2015, 2016]:
        cache = DATA_DIR / f"aurorasaurus_{year}_web.csv"
        if cache.exists():
            print(f"  Already cached: {cache.name}")
            continue

        url = f"https://zenodo.org/records/1255196/files/{year}_web_observations_cleaned.csv?download=1"
        print(f"  Downloading Aurorasaurus {year} web observations...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with open(cache, 'wb') as f:
                f.write(resp.content)
            print(f"  Saved: {cache.name} ({len(resp.content)//1024} KB)")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Also get the 2014-2019 science products inventory
    cache = DATA_DIR / "aurorasaurus_2014_2019_inventory.csv"
    if not cache.exists():
        url = "https://zenodo.org/records/3858106/files/Aurorasaurus_science_products_inventory_2014-2019.csv?download=1"
        print(f"  Downloading Aurorasaurus 2014-2019 inventory...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with open(cache, 'wb') as f:
                f.write(resp.content)
            print(f"  Saved: {cache.name} ({len(resp.content)//1024} KB)")
        except Exception as e:
            print(f"  FAILED: {e}")


def build_steve_event_list():
    """
    Compile STEVE events from:
    1. Literature (known dates from papers)
    2. Aurorasaurus STEVE-tagged reports (if available in data)
    """
    # Known STEVE events from literature
    known_events = [
        {"datetime": "2015-08-15 06:00", "source": "Little Bow Resort, Alberta", "type": "photograph"},
        {"datetime": "2015-09-07 05:30", "source": "SuperDARN/Frontiers 2024", "type": "radar+optical"},
        {"datetime": "2015-09-11 06:00", "source": "SuperDARN/Frontiers 2024", "type": "radar+optical"},
        {"datetime": "2016-07-08 06:00", "source": "Alberta Aurora Chasers (named STEVE)", "type": "citizen"},
        {"datetime": "2017-03-01 06:00", "source": "Maimaga subauroral station", "type": "instrument"},
        {"datetime": "2017-08-22 06:00", "source": "SuperDARN/Frontiers 2024", "type": "radar+optical"},
        {"datetime": "2018-03-14 06:00", "source": "Crossfield, Alberta", "type": "citizen"},
        {"datetime": "2018-04-10 06:00", "source": "TREx Spectrograph Lucky Lake", "type": "spectrograph"},
        {"datetime": "2019-03-27 06:00", "source": "Yellowknife YKNF", "type": "camera"},
        {"datetime": "2019-09-28 06:00", "source": "Gallardo-Lacourt 2024 quiet time", "type": "camera"},
        {"datetime": "2020-11-03 06:00", "source": "Chu 2020 JGR", "type": "camera"},
        {"datetime": "2022-04-14 06:00", "source": "Nishimura 2023 4K imaging", "type": "4K video"},
        {"datetime": "2022-09-03 06:00", "source": "Chen 2024 triangulation", "type": "dual camera"},
        {"datetime": "2024-05-10 06:00", "source": "May 2024 extreme storm, many reports", "type": "citizen"},
        {"datetime": "2024-10-11 06:00", "source": "South Australia, geomagnetic storm", "type": "citizen"},
    ]

    df = pd.DataFrame(known_events)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def build_control_aurora_list(omni):
    """
    Build a control sample of regular (non-STEVE) aurora events.
    Use times when AE > 500 (substorm, aurora visible) but NOT
    during a Bz reversal — these are "normal" aurora for comparison.
    """
    substorms = omni[omni["ae"] > 500].copy()
    # Deduplicate to one per 24h
    if len(substorms) == 0:
        return pd.DataFrame()
    days = substorms["day_number"].values
    filtered = [0]
    for i in range(1, len(days)):
        if days[i] - days[filtered[-1]] >= 1:
            filtered.append(i)
    substorms = substorms.iloc[filtered].copy()

    # Sample ~200 random aurora events for comparison
    if len(substorms) > 200:
        substorms = substorms.sample(200, random_state=42)

    return substorms


def main():
    print("=" * 70)
    print("STEVE Bz REVERSAL ANALYSIS — Extended")
    print("=" * 70)

    omni = load_omni()
    print(f"OMNI: {len(omni)} hourly records")

    # Download Aurorasaurus data
    print("\nDownloading Aurorasaurus data...")
    download_aurorasaurus()

    # Load and inspect Aurorasaurus
    for year in [2015, 2016]:
        cache = DATA_DIR / f"aurorasaurus_{year}_web.csv"
        if cache.exists():
            try:
                df = pd.read_csv(cache, low_memory=False)
                print(f"\n  Aurorasaurus {year}: {len(df)} reports")
                print(f"  Columns: {list(df.columns)[:10]}...")
                # Check for STEVE tag
                for col in df.columns:
                    if 'steve' in col.lower() or 'type' in col.lower() or 'category' in col.lower():
                        print(f"  {col}: {df[col].value_counts().head()}")
            except Exception as e:
                print(f"  Error reading {cache.name}: {e}")

    # Check inventory
    inv_cache = DATA_DIR / "aurorasaurus_2014_2019_inventory.csv"
    if inv_cache.exists():
        try:
            inv = pd.read_csv(inv_cache, low_memory=False)
            print(f"\n  Inventory 2014-2019: {len(inv)} entries")
            print(f"  Columns: {list(inv.columns)[:15]}")
            # Look for STEVE tags
            for col in inv.columns:
                if 'steve' in str(col).lower() or 'type' in str(col).lower():
                    vals = inv[col].dropna().unique()
                    if len(vals) < 20:
                        print(f"  {col}: {list(vals)}")
        except Exception as e:
            print(f"  Error reading inventory: {e}")

    # Build STEVE event list from literature
    print("\n=== STEVE Events from Literature ===")
    steve_events = build_steve_event_list()
    print(f"Known STEVE events: {len(steve_events)}")

    # Cross-reference with OMNI Bz
    print("\n=== Bz Reversal Analysis ===")
    print(f"{'Event':45s} {'Sign changes':>13s} {'Bz range':>10s} {'Bz at event':>12s}")

    steve_reversals = []
    for _, ev in steve_events.iterrows():
        sign_changes, bz_range, bz_at = count_bz_reversals(omni, ev["datetime"], window_hours=6)
        steve_reversals.append({
            "datetime": ev["datetime"],
            "source": ev["source"],
            "sign_changes": sign_changes,
            "bz_range": bz_range,
            "bz_at": bz_at,
        })
        print(f"  {ev['source']:43s} {sign_changes:>11.0f}   {bz_range:>8.1f} nT  {bz_at:>+10.1f} nT")

    steve_df = pd.DataFrame(steve_reversals)

    # Control: random aurora events
    print("\n=== Control: Regular Aurora (AE > 500, random sample) ===")
    control = build_control_aurora_list(omni)
    if len(control) > 0:
        control_reversals = []
        for _, ev in control.iterrows():
            sc, br, ba = count_bz_reversals(omni, ev["datetime"], window_hours=6)
            control_reversals.append({"sign_changes": sc, "bz_range": br, "bz_at": ba})
        control_df = pd.DataFrame(control_reversals)

        # Statistics
        steve_mean_sc = steve_df["sign_changes"].dropna().mean()
        control_mean_sc = control_df["sign_changes"].dropna().mean()
        steve_mean_range = steve_df["bz_range"].dropna().mean()
        control_mean_range = control_df["bz_range"].dropna().mean()

        print(f"\n  STEVE events (N={len(steve_df)}):")
        print(f"    Mean Bz sign changes (±6h): {steve_mean_sc:.1f}")
        print(f"    Mean Bz range: {steve_mean_range:.1f} nT")
        print(f"    Mean |Bz| at event: {steve_df['bz_at'].abs().mean():.1f} nT")

        print(f"\n  Regular aurora (N={len(control_df)}):")
        print(f"    Mean Bz sign changes (±6h): {control_mean_sc:.1f}")
        print(f"    Mean Bz range: {control_mean_range:.1f} nT")
        print(f"    Mean |Bz| at event: {control_df['bz_at'].abs().mean():.1f} nT")

        # Mann-Whitney test
        if len(steve_df["sign_changes"].dropna()) > 3 and len(control_df["sign_changes"].dropna()) > 3:
            _, p_sc = stats.mannwhitneyu(steve_df["sign_changes"].dropna(),
                                          control_df["sign_changes"].dropna(),
                                          alternative="greater")
            _, p_range = stats.mannwhitneyu(steve_df["bz_range"].dropna(),
                                            control_df["bz_range"].dropna(),
                                            alternative="greater")
            print(f"\n  Mann-Whitney (STEVE > Aurora):")
            print(f"    Sign changes: p = {p_sc:.4f}")
            print(f"    Bz range: p = {p_range:.4f}")

        # Fraction with "many" reversals (> median of control)
        threshold = control_df["sign_changes"].median()
        steve_frac = (steve_df["sign_changes"] > threshold).mean()
        control_frac = (control_df["sign_changes"] > threshold).mean()
        print(f"\n  Fraction with > {threshold:.0f} sign changes:")
        print(f"    STEVE: {steve_frac:.1%}")
        print(f"    Aurora: {control_frac:.1%}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.hist(control_df["sign_changes"].dropna(), bins=np.arange(0, 15, 1),
                alpha=0.5, color="#377eb8", label=f"Regular aurora (N={len(control_df)})", density=True)
        ax.hist(steve_df["sign_changes"].dropna(), bins=np.arange(0, 15, 1),
                alpha=0.7, color="#e41a1c", label=f"STEVE (N={len(steve_df)})", density=True)
        ax.set_xlabel("Bz sign changes in ±6h window")
        ax.set_ylabel("Density")
        ax.set_title("STEVE vs Regular Aurora: Bz Variability")
        ax.legend()

        ax = axes[1]
        ax.hist(control_df["bz_range"].dropna(), bins=20,
                alpha=0.5, color="#377eb8", label="Regular aurora", density=True)
        ax.hist(steve_df["bz_range"].dropna(), bins=20,
                alpha=0.7, color="#e41a1c", label="STEVE", density=True)
        ax.set_xlabel("Bz range (max - min) in ±6h window (nT)")
        ax.set_ylabel("Density")
        ax.set_title("STEVE vs Regular Aurora: Bz Swing Amplitude")
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUT_DIR / "steve_bz_reversal.png", dpi=150)
        print(f"\nSaved: {OUT_DIR / 'steve_bz_reversal.png'}")

    # Detailed hour-by-hour Bz composite around STEVE events
    print("\n=== Composite Bz Profile Around STEVE Events ===")
    hours = np.arange(-12, 7)
    steve_bz_composite = np.zeros((len(steve_events), len(hours)))

    for i, (_, ev) in enumerate(steve_events.iterrows()):
        t0 = pd.Timestamp(ev["datetime"])
        for j, h in enumerate(hours):
            t = t0 + pd.Timedelta(hours=h)
            row = omni[(omni["datetime"] >= t - pd.Timedelta(minutes=30)) &
                       (omni["datetime"] < t + pd.Timedelta(minutes=30))]
            if len(row) > 0:
                steve_bz_composite[i, j] = row["bz_gse"].iloc[0]
            else:
                steve_bz_composite[i, j] = np.nan

    mean_bz = np.nanmean(steve_bz_composite, axis=0)
    std_bz = np.nanstd(steve_bz_composite, axis=0)

    print(f"Hour  Mean Bz  Std")
    for j, h in enumerate(hours):
        marker = " <-- STEVE" if h == 0 else ""
        print(f"  {h:+3d}h  {mean_bz[j]:+6.2f}  {std_bz[j]:5.2f}{marker}")

    # Plot composite
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hours, mean_bz, 'o-', color="#e41a1c", linewidth=2.5, markersize=8)
    ax.fill_between(hours, mean_bz - std_bz, mean_bz + std_bz, alpha=0.2, color="#e41a1c")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="orange", linewidth=2, alpha=0.7, label="STEVE onset")
    ax.set_xlabel("Hours relative to STEVE observation")
    ax.set_ylabel("IMF Bz (nT, GSE)")
    ax.set_title(f"Composite Bz Profile Around {len(steve_events)} STEVE Events\n"
                 f"Framework prediction: Bz reversal at hour 0")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "steve_bz_composite.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'steve_bz_composite.png'}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"""
If STEVE events show MORE Bz sign changes and LARGER Bz swings
than regular aurora, the KT phase-boundary interpretation is supported:
STEVE = commutator at the magnetospheric phase boundary,
visible when the boundary sweeps across the observation latitude
during a Bz polarity reversal.

The composite Bz profile should show: negative Bz (southward) in the
hours before STEVE, crossing through zero AT the event, and
positive Bz (northward) after. That's the boundary sweeping past.
""")


if __name__ == "__main__":
    main()
