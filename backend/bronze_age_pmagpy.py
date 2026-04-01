#!/usr/bin/env python3
"""
Bronze Age field reconstruction using PmagPy's full spherical harmonic evaluation.
Replaces the rough dipole+quadrupole approximation with magsyn().
"""
import sys, os, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from pathlib import Path
from pmagpy import pmag

DATA = Path(__file__).parent.parent / "data" / "paleomag"
COEFF_FILE = DATA / "pfm9k2" / "pfm9k2_coeffs" / "pfm9k2-mean.txt"
OUT = Path(__file__).parent.parent / "frontend" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def load_pfm9k2():
    """Load pfm9k.2 mean Gauss coefficients."""
    with open(COEFF_FILE) as f:
        lines = f.readlines()
    times = np.array([float(x) for x in lines[0].split()])
    coeffs = []
    for line in lines[1:]:
        coeffs.append([float(x) for x in line.split()])
    return times, np.array(coeffs)


def eval_field_pmagpy(gh_column, lat, lon, alt=0):
    """
    Evaluate field at a point using PmagPy's magsyn.

    gh_column: Gauss coefficients for one time step (from pfm9k.2)
    lat, lon: geographic coordinates in degrees
    alt: altitude in km (0 = sea level)

    Returns: (x_north, y_east, z_down, f_total) in nT
    """
    colat = 90.0 - lat
    elong = lon % 360

    # magsyn expects: gh, sv, base_date, date, itype, alt, colat, elong
    # For a static model (no secular variation), set sv=zeros, date=base_date
    gh = np.array(gh_column, dtype='f')
    sv = np.zeros_like(gh)

    try:
        x, y, z, f = pmag.magsyn(gh, sv, 2000.0, 2000.0, 1, alt, colat, elong)
        return x, y, z, f
    except Exception as e:
        # Fallback: the gh array may be too short/long for magsyn
        # Pad or truncate to 120 elements (degree 10)
        gh_padded = np.zeros(120, dtype='f')
        n = min(len(gh), 120)
        gh_padded[:n] = gh[:n]
        sv_padded = np.zeros(120, dtype='f')
        x, y, z, f = pmag.magsyn(gh_padded, sv_padded, 2000.0, 2000.0, 1, alt, colat, elong)
        return x, y, z, f


def main():
    times, coeffs = load_pfm9k2()
    print(f"pfm9k.2: {len(times)} time steps, {coeffs.shape[0]} Gauss coefficients (degree {int((-1+math.sqrt(1+coeffs.shape[0]))/1):.0f})")

    sites = [
        ("Levant (Jerusalem)", 31.8, 35.2),
        ("Levant (Tel Megiddo)", 32.6, 35.2),
        ("Greece (Mycenae)", 37.7, 22.8),
        ("Egypt (Thebes)", 25.7, 32.6),
        ("Anatolia (Hattusa)", 40.0, 34.6),
        ("Mesopotamia (Babylon)", 32.5, 44.4),
        ("China (Anyang)", 36.1, 114.4),
        ("India (Harappa)", 27.5, 68.0),
        ("Europe (Rome)", 41.9, 12.5),
        ("S Atlantic (SAA)", -25.0, -50.0),
    ]

    # Focus period
    mask = (times >= -2000) & (times <= 500)
    t_focus = times[mask]
    c_focus = coeffs[:, mask]

    print(f"\nFull SH evaluation via PmagPy magsyn()")
    print(f"Period: {t_focus[0]:.0f} to {t_focus[-1]:.0f} CE")
    print("=" * 80)

    all_data = {}

    for name, lat, lon in sites:
        intensities = []
        for i in range(len(t_focus)):
            x, y, z, f = eval_field_pmagpy(c_focus[:, i], lat, lon)
            intensities.append(f)

        intensities = np.array(intensities)
        all_data[name] = intensities

        peak_idx = np.argmax(intensities)
        trough_idx = np.argmin(intensities)

        bc1200 = np.argmin(np.abs(t_focus - (-1200)))
        bc1000 = np.argmin(np.abs(t_focus - (-1000)))
        bc800 = np.argmin(np.abs(t_focus - (-800)))
        bc600 = np.argmin(np.abs(t_focus - (-600)))

        print(f"\n{name} ({lat}N, {lon}E):")
        print(f"  Peak: {intensities[peak_idx]/1000:.1f} uT at {t_focus[peak_idx]:.0f} CE ({-t_focus[peak_idx]:.0f} BCE)")
        print(f"  Min:  {intensities[trough_idx]/1000:.1f} uT at {t_focus[trough_idx]:.0f} CE")
        print(f"  1200 BCE: {intensities[bc1200]/1000:.1f} uT")
        print(f"  1000 BCE: {intensities[bc1000]/1000:.1f} uT")
        print(f"   800 BCE: {intensities[bc800]/1000:.1f} uT")
        print(f"   600 BCE: {intensities[bc600]/1000:.1f} uT")

        # Rate of change at collapse
        if bc1200 > 0:
            dFdt_1200 = (intensities[bc1200] - intensities[bc1200-1]) / 50 / 1000 * 100
            print(f"  dF/dt at 1200 BCE: {dFdt_1200:+.2f} uT/century")

    # Compare with our dipole approximation
    print("\n" + "=" * 80)
    print("COMPARISON: PmagPy full SH vs our dipole+quadrupole approx")
    print("=" * 80)

    from bronze_age_field import load_pfm9k2 as load2, gauss_to_field

    for name, lat, lon in [("Levant (Jerusalem)", 31.8, 35.2), ("China (Anyang)", 36.1, 114.4)]:
        print(f"\n{name}:")
        print(f"  {'Year':>6} {'PmagPy':>10} {'Approx':>10} {'Diff%':>8}")
        for year in [-1400, -1200, -1000, -800, -600]:
            idx = np.argmin(np.abs(t_focus - year))
            # PmagPy
            _, _, _, f_pmag = eval_field_pmagpy(c_focus[:, idx], lat, lon)
            # Our approx
            _, _, _, f_approx = gauss_to_field(c_focus[:, idx], lat, lon)
            diff = (f_approx - f_pmag) / f_pmag * 100
            print(f"  {-year:>6} {f_pmag/1000:>9.1f} {f_approx/1000:>9.1f} {diff:>+7.1f}%")

    # Save improved time series as JSON for dashboard
    import json
    series = {
        "times": (-t_focus).tolist(),
        "model": "pfm9k.2 via PmagPy magsyn (full SH)",
        "sites": {},
    }
    colors = {"Levant (Jerusalem)": "#ff4444", "Greece (Mycenae)": "#44aaff",
              "Egypt (Thebes)": "#44ff44", "Anatolia (Hattusa)": "#ffaa44",
              "Mesopotamia (Babylon)": "#cc44cc", "China (Anyang)": "#ffff44",
              "India (Harappa)": "#ff88ff", "S Atlantic (SAA)": "#888888",
              "Europe (Rome)": "#4488ff"}

    for name, vals in all_data.items():
        series["sites"][name] = {
            "values": (vals / 1000).tolist(),  # uT
            "color": colors.get(name, "#888888"),
        }

    events = [
        {"year": 1200, "label": "Bronze Age Collapse", "color": "#ff4444"},
        {"year": 1177, "label": "Sea Peoples", "color": "#ff6644"},
        {"year": 1000, "label": "Levantine Anomaly", "color": "#ffaa44"},
        {"year": 1046, "label": "Zhou Dynasty", "color": "#ffff44"},
        {"year": 800, "label": "Iron Age", "color": "#44ff44"},
        {"year": 586, "label": "Fall of Jerusalem", "color": "#ff8888"},
        {"year": 1600, "label": "Hittite Peak", "color": "#44aaff"},
        {"year": 1500, "label": "Shang Dynasty", "color": "#ffff44"},
    ]
    series["events"] = events

    outfile = OUT / "bronze_age_field.json"
    with open(outfile, "w") as f:
        json.dump(series, f, separators=(",", ":"))
    print(f"\nSaved {outfile}")

    # Generate improved plot
    print("\nGenerating improved plots...")
    generate_plots(t_focus, all_data, events)


def generate_plots(t, data, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 12), facecolor="#0a0a1a")
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25,
                  left=0.07, right=0.95, top=0.93, bottom=0.06)

    def style(ax, title=""):
        ax.set_facecolor("#0d0d2b")
        ax.tick_params(colors="#999", labelsize=8)
        for s in ["bottom", "left"]:
            ax.spines[s].set_color("#333")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        if title:
            ax.set_title(title, color="#0cf", fontsize=11, fontweight="bold")

    # Panel 1: All sites
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, "Geomagnetic Field Intensity: Full SH Evaluation (pfm9k.2 via PmagPy)")
    primary = ["Levant (Jerusalem)", "China (Anyang)"]
    secondary = ["Greece (Mycenae)", "Anatolia (Hattusa)", "Egypt (Thebes)",
                  "Mesopotamia (Babylon)", "India (Harappa)", "S Atlantic (SAA)"]

    colors = {"Levant (Jerusalem)": "#ff4444", "China (Anyang)": "#ffff44",
              "Greece (Mycenae)": "#44aaff", "Anatolia (Hattusa)": "#ffaa44",
              "Egypt (Thebes)": "#44ff44", "Mesopotamia (Babylon)": "#cc44cc",
              "India (Harappa)": "#ff88ff", "S Atlantic (SAA)": "#888888"}

    for name in secondary:
        if name in data:
            ax1.plot(-t, data[name]/1000, color=colors[name], linewidth=0.8, alpha=0.4, label=name)
    for name in primary:
        if name in data:
            ax1.plot(-t, data[name]/1000, color=colors[name], linewidth=2.5, alpha=1.0, label=name)

    for ev in events:
        ax1.axvline(ev["year"], color=ev["color"], linewidth=0.7, linestyle="--", alpha=0.5)

    ax1.set_xlabel("Year BCE", color="#999", fontsize=9)
    ax1.set_ylabel("Total Intensity (uT)", color="#999", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left", facecolor="#0d0d2b", edgecolor="#333",
               labelcolor="#ccc", ncol=4)
    ax1.set_xlim(2000, -500)

    # Panel 2: Levant vs China
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, "Levant vs China: The Anomaly Migration")
    lev = data.get("Levant (Jerusalem)", np.zeros(len(t))) / 1000
    chi = data.get("China (Anyang)", np.zeros(len(t))) / 1000
    ax2.plot(-t, lev, color="#ff4444", linewidth=2, label="Levant")
    ax2.plot(-t, chi, color="#ffff44", linewidth=2, label="China")
    ax2.fill_between(-t, lev, chi, where=lev > chi, color="#ff4444", alpha=0.1)
    ax2.fill_between(-t, lev, chi, where=chi > lev, color="#ffff44", alpha=0.1)
    ax2.axvline(1200, color="#ff4444", linewidth=1.5, linestyle="--")
    ax2.text(1200, ax2.get_ylim()[0] + 2 if len(lev) > 0 else 35,
             "Collapse", color="#ff4444", fontsize=8, ha="center")
    ax2.legend(fontsize=8, facecolor="#0d0d2b", edgecolor="#333", labelcolor="#ccc")
    ax2.set_xlabel("Year BCE", color="#999")
    ax2.set_ylabel("uT", color="#999")
    ax2.set_xlim(2000, -500)

    # Panel 3: dF/dt
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, "dF/dt: Field Change Rate (KT Order Parameter)")
    dt = 50
    for name, col, lw in [("Levant (Jerusalem)", "#ff4444", 2),
                           ("China (Anyang)", "#ffff44", 2),
                           ("Greece (Mycenae)", "#44aaff", 1)]:
        if name in data:
            dFdt = np.gradient(data[name] / 1000, dt) * 100
            ax3.plot(-t, dFdt, color=col, linewidth=lw, alpha=0.8, label=name.split("(")[0].strip())

    ax3.axhline(0, color="#444", linewidth=0.5)
    lev_dFdt = np.gradient(data.get("Levant (Jerusalem)", np.zeros(len(t))) / 1000, dt) * 100
    ax3.fill_between(-t, lev_dFdt, 0, where=lev_dFdt < 0, color="#ff4444", alpha=0.15)
    ax3.axvline(1200, color="#ff4444", linewidth=1.5, linestyle="--")
    ax3.legend(fontsize=7, facecolor="#0d0d2b", edgecolor="#333", labelcolor="#ccc")
    ax3.set_xlabel("Year BCE", color="#999")
    ax3.set_ylabel("uT / century", color="#999")
    ax3.set_xlim(2000, -500)

    # Panel 4 & 5: Global maps at 1200 BCE and 600 BCE
    lats = np.arange(-60, 65, 4)
    lons = np.arange(-180, 184, 4)

    for panel_idx, (year, title) in enumerate([(-1200, "1200 BCE (Collapse)"), (-600, "600 BCE (Anomaly Peak)")]):
        ax = fig.add_subplot(gs[2, panel_idx])
        style(ax, f"Field Intensity at {title}")
        tidx = np.argmin(np.abs(t - year))
        coeffs_at_t = load_pfm9k2()[1][:, np.argmin(np.abs(load_pfm9k2()[0] - year))]

        grid = np.zeros((len(lats), len(lons)))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                _, _, _, f = eval_field_pmagpy(coeffs_at_t, lat, lon)
                grid[i, j] = f / 1000

        im = ax.pcolormesh(lons, lats, grid, cmap="inferno", vmin=20, vmax=80, shading="auto")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 60)
        ax.set_aspect("equal")

        # Mark sites
        site_marks = [("Levant", 31.8, 35.2, "#ff4444"), ("China", 36.1, 114.4, "#ffff44"),
                      ("Greece", 37.7, 22.8, "#44aaff"), ("Egypt", 25.7, 32.6, "#44ff44"),
                      ("India", 27.5, 68.0, "#ff88ff")]
        for sname, slat, slon, scol in site_marks:
            ax.plot(slon, slat, "o", color=scol, markersize=5, markeredgecolor="white", markeredgewidth=0.5)
            ax.text(slon + 4, slat, sname, color="white", fontsize=6)

        plt.colorbar(im, ax=ax, label="uT", shrink=0.7, pad=0.02)
        ax.set_xlabel("Longitude", color="#999", fontsize=7)
        ax.set_ylabel("Latitude", color="#999", fontsize=7)

    fig.suptitle("The KT Transition in the Geodynamo: Full Spherical Harmonic Evaluation",
                 color="#0cf", fontsize=13, fontweight="bold", y=0.97)

    outfile = OUT / "bronze_age_field.png"
    fig.savefig(outfile, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
