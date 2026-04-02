#!/usr/bin/env python3
"""
Correlate historical records (earthquakes, eruptions, auroras)
with the pfm9k.2 paleomagnetic field reconstruction.

Test: do periods of rapid field change (dF/dt) correlate with
increased seismicity, volcanic activity, and aurora sightings?
"""
import sys, os, json, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bronze_age_pmagpy import load_pfm9k2, eval_field_pmagpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DATA = Path(__file__).parent.parent / "data"
HIST = DATA / "historical"
OUT = Path(__file__).parent.parent / "frontend" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def load_historical_earthquakes():
    f = HIST / "all_historical_earthquakes.json"
    if not f.exists(): return []
    with open(f) as fh:
        items = json.load(fh)
    eqs = []
    for item in items:
        year = item.get("year")
        if year is None: continue
        eqs.append({
            "year": year,
            "mag": item.get("eqMagnitude"),
            "lat": item.get("latitude"),
            "lon": item.get("longitude"),
            "deaths": item.get("deaths"),
            "location": item.get("locationName", ""),
            "country": item.get("country", ""),
        })
    return eqs


def compute_field_timeseries():
    """Compute field at key sites using pfm9k.2."""
    times, coeffs = load_pfm9k2()

    sites = {
        "Levant": (31.8, 35.2),
        "Greece": (37.7, 22.8),
        "China": (36.1, 114.4),
        "Egypt": (25.7, 32.6),
        "Italy": (41.9, 12.5),
    }

    result = {"times": times}
    for name, (lat, lon) in sites.items():
        F_vals = []
        for i in range(len(times)):
            _, _, _, F = eval_field_pmagpy(coeffs[:, i], lat, lon)
            F_vals.append(F / 1000)  # uT
        result[name] = np.array(F_vals)
        # dF/dt in uT/century
        result[f"{name}_dFdt"] = np.gradient(np.array(F_vals), 0.5) # 50-year steps = 0.5 century

    return result


def main():
    print("HISTORICAL RECORDS vs PALEOMAGNETIC FIELD")
    print("=" * 70)

    # Load data
    eqs = load_historical_earthquakes()
    print(f"Historical earthquakes: {len(eqs)}")

    field = compute_field_timeseries()
    times = field["times"]  # CE (negative = BCE)
    print(f"Field model: {len(times)} time steps, {times[0]:.0f} to {times[-1]:.0f} CE")

    # Bin earthquakes by century
    eq_years = [e["year"] for e in eqs if e["year"] is not None]
    century_bins = np.arange(-2200, 2000, 100)
    eq_counts, _ = np.histogram(eq_years, bins=century_bins)
    century_centers = (century_bins[:-1] + century_bins[1:]) / 2

    # Bin by region
    regions = {
        "Mediterranean": lambda e: e.get("country", "") in ("GREECE", "ITALY", "TURKEY", "SYRIA", "ISRAEL", "JORDAN", "LEBANON", "EGYPT", "CYPRUS"),
        "China": lambda e: e.get("country", "") == "CHINA",
        "Other": lambda e: True,
    }

    print("\nEarthquake counts by region and era:")
    for rname, rfunc in regions.items():
        reg_eqs = [e for e in eqs if rfunc(e)]
        for era, lo, hi in [("Bronze", -2000, -1000), ("Iron", -1000, -500),
                             ("Classical", -500, 500), ("Medieval", 500, 1500)]:
            n = sum(1 for e in reg_eqs if e["year"] is not None and lo <= e["year"] < hi)
            if n > 0:
                print(f"  {rname:>15} {era:>10}: {n}")

    # Compute field derivatives at earthquake times
    # For each earthquake, find the nearest pfm9k.2 time step and get dF/dt
    print("\nField conditions at earthquake times:")
    for site in ["Levant", "China", "Italy"]:
        dFdt = field[f"{site}_dFdt"]
        eq_dFdt = []
        for e in eqs:
            if e["year"] is None: continue
            idx = np.argmin(np.abs(times - e["year"]))
            eq_dFdt.append(dFdt[idx])

        if eq_dFdt:
            eq_dFdt = np.array(eq_dFdt)
            print(f"  {site}: mean dF/dt at EQ times = {eq_dFdt.mean():.2f} uT/century")
            print(f"           overall mean dF/dt    = {dFdt.mean():.2f} uT/century")
            # Are earthquakes preferentially during negative dF/dt?
            neg_frac = (eq_dFdt < 0).mean()
            overall_neg = (dFdt < 0).mean()
            print(f"           frac negative at EQ = {neg_frac:.2f} vs overall {overall_neg:.2f}")

    # === PLOT ===
    fig = plt.figure(figsize=(16, 14), facecolor="#0a0a1a")
    gs = GridSpec(4, 1, figure=fig, hspace=0.25, left=0.08, right=0.95, top=0.94, bottom=0.05)

    def style(ax, title=""):
        ax.set_facecolor("#0d0d2b")
        ax.tick_params(colors="#999", labelsize=8)
        for s in ["bottom", "left"]: ax.spines[s].set_color("#333")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        if title: ax.set_title(title, color="#0cf", fontsize=11, fontweight="bold")

    # Panel 1: Earthquake count by century
    ax1 = fig.add_subplot(gs[0])
    style(ax1, "Historical Earthquake Count by Century")
    ax1.bar(-century_centers, eq_counts, width=80, color="#ff4444", alpha=0.7)
    ax1.set_ylabel("Earthquakes / century", color="#999")
    ax1.set_xlim(2200, -500)
    ax1.text(1800, eq_counts.max() * 0.8, "sparse\nrecords", color="#666", fontsize=8, ha="center")
    ax1.text(500, eq_counts.max() * 0.8, "Classical\nGreece/Rome", color="#ff8844", fontsize=8, ha="center")

    # Panel 2: Levant field + dF/dt
    ax2 = fig.add_subplot(gs[1])
    style(ax2, "Levant Field Intensity and Rate of Change (pfm9k.2)")
    mask = (times >= -2200) & (times <= 500)
    t = times[mask]
    F = field["Levant"][mask]
    dF = field["Levant_dFdt"][mask]

    ax2.plot(-t, F, color="#ff4444", linewidth=2, label="F (uT)")
    ax2.set_ylabel("Intensity (uT)", color="#ff4444")
    ax2b = ax2.twinx()
    ax2b.fill_between(-t, dF, 0, where=dF < 0, color="#ff4444", alpha=0.15)
    ax2b.fill_between(-t, dF, 0, where=dF > 0, color="#44ff44", alpha=0.1)
    ax2b.plot(-t, dF, color="#ffaa44", linewidth=1, alpha=0.7, label="dF/dt")
    ax2b.set_ylabel("dF/dt (uT/century)", color="#ffaa44")
    ax2b.tick_params(colors="#999")
    ax2.set_xlim(2200, -500)

    # Annotate key events
    events = [
        (1200, "Bronze Age\nCollapse", "#ff4444"),
        (1177, "Sea Peoples", "#ff6644"),
        (586, "Fall of\nJerusalem", "#ff8888"),
        (464, "Sparta\nEarthquake", "#44aaff"),
        (373, "Helice\nTsunami", "#44aaff"),
        (226, "Rhodes\nEarthquake", "#44aaff"),
    ]
    for yr, lbl, col in events:
        ax2.axvline(yr, color=col, linewidth=0.7, linestyle="--", alpha=0.5)

    # Panel 3: China field + Chinese earthquake record
    ax3 = fig.add_subplot(gs[2])
    style(ax3, "China Field Intensity vs Chinese Earthquake Record")
    F_chi = field["China"][mask]
    dF_chi = field["China_dFdt"][mask]
    ax3.plot(-t, F_chi, color="#ffff44", linewidth=2, label="China F (uT)")
    ax3.set_ylabel("Intensity (uT)", color="#ffff44")

    # Overlay Chinese earthquake counts
    chi_eqs = [e["year"] for e in eqs if e.get("country") == "CHINA" and e["year"] is not None]
    if chi_eqs:
        chi_counts, _ = np.histogram(chi_eqs, bins=century_bins)
        ax3b = ax3.twinx()
        ax3b.bar(-century_centers, chi_counts, width=80, color="#ff4444", alpha=0.4, label="EQs")
        ax3b.set_ylabel("Earthquakes / century", color="#ff4444")
        ax3b.tick_params(colors="#999")
    ax3.set_xlim(2200, -500)

    # Panel 4: All regions earthquake rate vs global mean field
    ax4 = fig.add_subplot(gs[3])
    style(ax4, "Earthquake Rate vs Global Mean dF/dt")

    # Global mean dF/dt (average of all sites)
    global_dFdt = np.mean([field[f"{s}_dFdt"][mask] for s in ["Levant", "Greece", "China", "Egypt", "Italy"]], axis=0)

    # Smooth earthquake rate
    smooth_eq = np.convolve(eq_counts, np.ones(3)/3, mode="same")

    # Plot both normalized
    t_eq = -century_centers
    mask_eq = (t_eq >= -500) & (t_eq <= 2200)

    ax4.plot(-t, -global_dFdt, color="#ff4444", linewidth=2, label="-dF/dt (inverted: positive = field dropping)")
    ax4.axhline(0, color="#444", linewidth=0.5)
    ax4.set_ylabel("-dF/dt (uT/century)", color="#ff4444")
    ax4.set_xlabel("Year BCE", color="#999")
    ax4.set_xlim(2200, -500)
    ax4.legend(fontsize=8, facecolor="#0d0d2b", edgecolor="#333", labelcolor="#ccc")

    fig.suptitle("Historical Records and the Geomagnetic KT Transition",
                 color="#0cf", fontsize=13, fontweight="bold", y=0.97)

    outfile = OUT / "historical_correlation.png"
    fig.savefig(outfile, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved {outfile}")


if __name__ == "__main__":
    main()
