#!/usr/bin/env python3
"""
Generate Bronze Age paleomagnetic field plots.
Visualizes the Levantine anomaly and its migration.
"""
import sys, os, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path(__file__).parent.parent / "frontend" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# Import field computation
sys.path.insert(0, str(Path(__file__).parent))
from bronze_age_field import load_pfm9k2, gauss_to_field


def build_plots():
    times, coeffs = load_pfm9k2()
    mask = (times >= -2500) & (times <= 500)
    t = times[mask]
    c = coeffs[:, mask]

    # Sites
    sites = [
        ("Levant", 31.8, 35.2, "#ff4444"),
        ("Greece", 37.7, 22.8, "#44aaff"),
        ("Anatolia", 40.0, 34.6, "#ffaa44"),
        ("Egypt", 25.7, 32.6, "#44ff44"),
        ("Mesopotamia", 32.5, 44.4, "#cc44cc"),
        ("China", 36.1, 114.4, "#ffff44"),
        ("S Atlantic", -25.0, -50.0, "#888888"),
    ]

    # Compute field at all sites
    data = {}
    for name, lat, lon, _ in sites:
        F_vals = []
        for i in range(len(t)):
            _, _, _, F = gauss_to_field(c[:, i], lat, lon)
            F_vals.append(F / 1000)  # convert to uT
        data[name] = np.array(F_vals)

    # Historical events
    events = [
        (-1200, "Bronze Age\nCollapse", "#ff4444"),
        (-1000, "Levantine\nAnomaly", "#ffaa44"),
        (-800, "Iron Age\nExpansion", "#44ff44"),
        (-586, "Fall of\nJerusalem", "#ff8888"),
        (-1046, "Zhou Dynasty\n(China)", "#ffff44"),
        (-1177, "Sea Peoples\nInvasion", "#ff6644"),
    ]

    # === FIGURE 1: Multi-site time series ===
    fig = plt.figure(figsize=(16, 10), facecolor="#0a0a1a")
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25,
                  left=0.08, right=0.95, top=0.92, bottom=0.08)

    def style(ax, title=""):
        ax.set_facecolor("#0d0d2b")
        ax.tick_params(colors="#999", labelsize=8)
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if title:
            ax.set_title(title, color="#0cf", fontsize=11, fontweight="bold")

    # Panel 1: All sites overlaid
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, "Geomagnetic Field Intensity: Bronze Age to Iron Age (pfm9k.2)")
    for name, lat, lon, col in sites:
        lw = 2.5 if name in ("Levant", "China") else 1.2
        alpha = 1.0 if name in ("Levant", "China") else 0.6
        ax1.plot(-t, data[name], color=col, linewidth=lw, alpha=alpha, label=name)

    for yr, label, col in events:
        ax1.axvline(-yr, color=col, linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.text(-yr, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 70,
                 label, color=col, fontsize=6, ha="center", va="bottom", rotation=0)

    ax1.set_xlabel("Year BCE", color="#999", fontsize=9)
    ax1.set_ylabel("Field Intensity (uT)", color="#999", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left", facecolor="#0d0d2b",
               edgecolor="#333", labelcolor="#ccc", ncol=4)
    ax1.set_xlim(2500, -500)
    ax1.invert_xaxis()

    # Panel 2: Levant vs China with KT annotation
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, "Levant vs China: Anomaly Migration")
    ax2.plot(-t, data["Levant"], color="#ff4444", linewidth=2, label="Levant")
    ax2.plot(-t, data["China"], color="#ffff44", linewidth=2, label="China")
    ax2.fill_between(-t, data["Levant"], data["China"],
                     where=data["Levant"] > data["China"],
                     color="#ff4444", alpha=0.1, label="Levant > China")
    ax2.fill_between(-t, data["Levant"], data["China"],
                     where=data["China"] > data["Levant"],
                     color="#ffff44", alpha=0.1, label="China > Levant")

    # Mark the crossover points
    diff = data["Levant"] - data["China"]
    for i in range(1, len(diff)):
        if diff[i-1] * diff[i] < 0:
            ax2.axvline(-t[i], color="white", linewidth=0.5, linestyle=":")
            ax2.text(-t[i], ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 35,
                     f"{-t[i]:.0f}", color="white", fontsize=7, ha="center")

    ax2.legend(fontsize=7, loc="upper left", facecolor="#0d0d2b",
               edgecolor="#333", labelcolor="#ccc")
    ax2.set_xlabel("Year BCE", color="#999", fontsize=9)
    ax2.set_ylabel("uT", color="#999", fontsize=9)
    ax2.set_xlim(2500, -500)

    # Panel 3: Rate of change (dF/dt) - the KT order parameter
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, "dF/dt: Rate of Field Change (KT Order Parameter)")
    dt = 50  # years between samples
    for name, col in [("Levant", "#ff4444"), ("China", "#ffff44"), ("Greece", "#44aaff")]:
        dFdt = np.gradient(data[name], dt) * 100  # uT per century
        ax3.plot(-t, dFdt, color=col, linewidth=1.5, alpha=0.8, label=name)

    ax3.axhline(0, color="#444", linewidth=0.5)
    ax3.fill_between(-t, np.gradient(data["Levant"], dt) * 100, 0,
                     where=np.gradient(data["Levant"], dt) < 0,
                     color="#ff4444", alpha=0.15)

    # The Bronze Age Collapse is where dF/dt is most NEGATIVE for the Levant
    ax3.axvline(1200, color="#ff4444", linewidth=1.5, linestyle="--")
    ax3.text(1200, ax3.get_ylim()[1] * 0.9 if ax3.get_ylim()[1] > 0 else 5,
             " 1200 BCE\n Collapse", color="#ff4444", fontsize=8)

    ax3.legend(fontsize=7, loc="lower left", facecolor="#0d0d2b",
               edgecolor="#333", labelcolor="#ccc")
    ax3.set_xlabel("Year BCE", color="#999", fontsize=9)
    ax3.set_ylabel("uT / century", color="#999", fontsize=9)
    ax3.set_xlim(2500, -500)

    # Panel 4: Field intensity map at 1200 BCE vs 600 BCE
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4, "Field Intensity at 1200 BCE (Collapse)")
    ax5 = fig.add_subplot(gs[2, 1])
    style(ax5, "Field Intensity at 600 BCE (Anomaly Peak)")

    lats = np.arange(-60, 65, 5)
    lons = np.arange(-180, 185, 5)

    for ax, year, title in [(ax4, -1200, "1200 BCE"), (ax5, -600, "600 BCE")]:
        tidx = np.argmin(np.abs(t - year))
        grid = np.zeros((len(lats), len(lons)))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                _, _, _, F = gauss_to_field(c[:, tidx], lat, lon)
                grid[i, j] = F / 1000

        im = ax.pcolormesh(lons, lats, grid, cmap="inferno", vmin=25, vmax=80, shading="auto")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 60)
        ax.set_aspect("equal")

        # Mark collapse sites
        for name, lat, lon, col in sites[:6]:
            ax.plot(lon, lat, "o", color=col, markersize=4, markeredgecolor="white", markeredgewidth=0.5)
            ax.text(lon + 3, lat, name, color="white", fontsize=5)

        plt.colorbar(im, ax=ax, label="uT", shrink=0.7, pad=0.02)
        ax.set_xlabel("Longitude", color="#999", fontsize=7)
        ax.set_ylabel("Latitude", color="#999", fontsize=7)

    fig.suptitle("The KT Transition in the Geodynamo: Bronze Age Collapse to Levantine Anomaly",
                 color="#0cf", fontsize=13, fontweight="bold", y=0.97)

    outfile = OUT / "bronze_age_field.png"
    fig.savefig(outfile, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {outfile}")

    # Also save the time series as JSON for the dashboard
    import json
    series = {
        "times": (-t).tolist(),
        "sites": {},
    }
    for name, _, _, col in sites:
        series["sites"][name] = {
            "values": data[name].tolist(),
            "color": col,
        }
    series["events"] = [{"year": -yr, "label": lbl, "color": c} for yr, lbl, c in events]

    json_file = OUT / "bronze_age_field.json"
    with open(json_file, "w") as f:
        json.dump(series, f, separators=(",", ":"))
    print(f"Saved {json_file}")


if __name__ == "__main__":
    build_plots()
