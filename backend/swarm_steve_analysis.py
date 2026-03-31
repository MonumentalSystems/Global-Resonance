#!/usr/bin/env python3
"""
Swarm STEVE Analysis: Plasma at the KT Boundaries
====================================================
Use Swarm EFI, AOB, and FAC data to test:
1. Is there a density gradient at the auroral boundary during STEVE?
2. Does T_elec spike at the boundary (commutator heating)?
3. Do the FAC show the vortex current structure?
4. Is the equatorward boundary (STEVE) symmetric with the poleward boundary (Anti-STEVE)?
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent / "data"

events = [
    ("STEVE_SuperDARN1", "2015-09-07"),
    ("STEVE_TREx", "2018-04-10"),
    ("STEVE_SouthAustralia", "2024-10-11"),
]

def analyze_event(label, date_str):
    print(f"\n{'='*60}")
    print(f"  {label} ({date_str})")
    print(f"{'='*60}")

    efi = pd.read_csv(DATA_DIR / f"swarm_{label}_efi.csv", index_col=0, parse_dates=True)
    aob = pd.read_csv(DATA_DIR / f"swarm_{label}_aob.csv", index_col=0, parse_dates=True)
    fac = pd.read_csv(DATA_DIR / f"swarm_{label}_fac.csv", index_col=0, parse_dates=True)

    print(f"  EFI: {len(efi)} records, AOB: {len(aob)} boundaries, FAC: {len(fac)} records")

    # EFI columns
    print(f"  EFI columns: {list(efi.columns)}")
    print(f"  AOB columns: {list(aob.columns)}")

    # QDLat = quasi-dipole latitude (magnetic latitude)
    if "QDLat" in efi.columns:
        lat_col = "QDLat"
    elif "Latitude" in efi.columns:
        lat_col = "Latitude"
    else:
        lat_col = [c for c in efi.columns if "lat" in c.lower()][0] if any("lat" in c.lower() for c in efi.columns) else None

    if lat_col is None:
        print("  Cannot find latitude column!")
        return None

    print(f"  Using latitude column: {lat_col}")
    print(f"  Latitude range: {efi[lat_col].min():.1f} to {efi[lat_col].max():.1f}")

    # Auroral boundary positions
    if "Latitude_QD" in aob.columns:
        print(f"\n  Auroral boundaries:")
        for _, row in aob.iterrows():
            flag = row.get("Boundary_Flag", "?")
            lat = row.get("Latitude_QD", "?")
            mlt = row.get("MLT_QD", "?")
            print(f"    Flag={flag}, Lat={lat:.1f}, MLT={mlt:.1f}")

    # Find high-latitude passes (where STEVE would be, ~50-65 QDLat)
    steve_lat_range = (45, 70)  # QDLat range where STEVE typically appears
    high_lat = efi[(efi[lat_col].abs() >= steve_lat_range[0]) &
                    (efi[lat_col].abs() <= steve_lat_range[1])].copy()

    if len(high_lat) == 0:
        print(f"  No data in STEVE latitude range ({steve_lat_range})")
        return None

    print(f"\n  Data in STEVE latitude range ({steve_lat_range[0]}-{steve_lat_range[1]} QDLat): {len(high_lat)} records")

    # Electron density and temperature profiles vs latitude
    ne_col = [c for c in efi.columns if "N_elec" in c or "Ne" in c]
    te_col = [c for c in efi.columns if "T_elec" in c or "Te" in c]
    ni_col = [c for c in efi.columns if "N_ion" in c or "Ni" in c]

    ne_col = ne_col[0] if ne_col else None
    te_col = te_col[0] if te_col else None
    ni_col = ni_col[0] if ni_col else None

    print(f"  Ne column: {ne_col}")
    print(f"  Te column: {te_col}")
    print(f"  Ni column: {ni_col}")

    if ne_col and te_col:
        # Bin by latitude
        lat_bins = np.arange(40, 85, 2)
        lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

        ne_profile = []
        te_profile = []
        for i in range(len(lat_bins) - 1):
            mask = (efi[lat_col].abs() >= lat_bins[i]) & (efi[lat_col].abs() < lat_bins[i+1])
            subset = efi[mask]
            ne_vals = pd.to_numeric(subset[ne_col], errors="coerce")
            te_vals = pd.to_numeric(subset[te_col], errors="coerce")
            ne_profile.append(ne_vals.median())
            te_profile.append(te_vals.median())

        ne_profile = np.array(ne_profile)
        te_profile = np.array(te_profile)

        # Find the density gradient (the KT boundary)
        ne_grad = np.abs(np.diff(ne_profile))
        peak_grad_idx = np.nanargmax(ne_grad)
        boundary_lat = lat_centers[peak_grad_idx]

        print(f"\n  Density gradient analysis:")
        print(f"    Peak gradient at QDLat = {boundary_lat:.1f}")
        print(f"    Ne below boundary: {np.nanmedian(ne_profile[:peak_grad_idx]):.0f} /cm3")
        print(f"    Ne above boundary: {np.nanmedian(ne_profile[peak_grad_idx:]):.0f} /cm3")
        ratio = np.nanmedian(ne_profile[peak_grad_idx:]) / max(np.nanmedian(ne_profile[:peak_grad_idx]), 1)
        print(f"    Density ratio: {ratio:.2f}x")

        # Temperature at the boundary
        te_at_boundary = te_profile[peak_grad_idx]
        te_away = np.nanmedian(np.concatenate([te_profile[:max(peak_grad_idx-2,0)],
                                                te_profile[min(peak_grad_idx+3,len(te_profile)):]]))
        print(f"\n  Temperature at boundary: {te_at_boundary:.0f} K")
        print(f"  Temperature away from boundary: {te_away:.0f} K")
        if te_away > 0:
            print(f"  Enhancement: {te_at_boundary/te_away:.2f}x")

        return {
            "label": label, "boundary_lat": boundary_lat,
            "ne_ratio": ratio, "te_at": te_at_boundary, "te_away": te_away,
            "te_enhancement": te_at_boundary / max(te_away, 1),
            "lat_centers": lat_centers, "ne_profile": ne_profile, "te_profile": te_profile,
        }

    return None


def main():
    print("=" * 60)
    print("SWARM STEVE ANALYSIS: Plasma at the KT Boundaries")
    print("=" * 60)

    results = []
    for label, date in events:
        try:
            r = analyze_event(label, date)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    if not results:
        print("\nNo results to plot.")
        return

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Event':>30s} {'Boundary':>10s} {'Ne ratio':>10s} {'Te enh':>10s}")
    for r in results:
        print(f"  {r['label']:>28s} {r['boundary_lat']:>8.1f} QD "
              f"{r['ne_ratio']:>9.2f}x {r['te_enhancement']:>9.2f}x")

    print(f"\n  Framework prediction:")
    print(f"    Density gradient >3x at boundary (Swarm literature: 5x)")
    print(f"    Temperature enhancement ~2x at boundary (literature: 5900K vs ~3000K)")
    print(f"    The boundary IS the KT phase transition in the plasma")

    # Plot profiles for each event
    n_events = len(results)
    fig, axes = plt.subplots(n_events, 2, figsize=(14, 5 * n_events))
    if n_events == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(results):
        ax = axes[i, 0]
        valid = ~np.isnan(r["ne_profile"])
        ax.plot(r["lat_centers"][valid], r["ne_profile"][valid], "o-", color="steelblue", lw=2)
        ax.axvline(r["boundary_lat"], color="red", linestyle="--", label=f"Boundary: {r['boundary_lat']:.0f}")
        ax.set_xlabel("Quasi-Dipole Latitude (deg)")
        ax.set_ylabel("Electron Density (cm-3)")
        ax.set_title(f"{r['label']}: Density Profile")
        ax.legend()

        ax = axes[i, 1]
        valid = ~np.isnan(r["te_profile"])
        ax.plot(r["lat_centers"][valid], r["te_profile"][valid], "o-", color="red", lw=2)
        ax.axvline(r["boundary_lat"], color="orange", linestyle="--", label=f"Boundary: {r['boundary_lat']:.0f}")
        ax.set_xlabel("Quasi-Dipole Latitude (deg)")
        ax.set_ylabel("Electron Temperature (K)")
        ax.set_title(f"{r['label']}: Temperature Profile")
        ax.legend()

    plt.suptitle("Swarm EFI: Plasma at STEVE KT Boundaries", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "swarm_steve_profiles.png", dpi=150)
    print(f"\nSaved: swarm_steve_profiles.png")


if __name__ == "__main__":
    main()
