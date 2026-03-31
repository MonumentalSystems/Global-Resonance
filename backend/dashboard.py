#!/usr/bin/env python3
"""
Real-time CME impact dashboard.

Generates a multi-panel PNG showing:
  1. Solar wind (Bz, speed, density) — last 24h
  2. Kp index — last 3 days
  3. Earthquake map with Jelly Ball zones
  4. Zone histogram (angular distance from subsolar)
  5. Swarm F field (if available)
  6. Lunar phase + tidal stress
  7. Timeline: Grade-0 / Grade-4 / Grade-2 windows
  8. Prediction scorecard

Run:  python dashboard.py
Output: output/dashboard.png
"""
import sys, os, json, math, csv
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta, timezone
from pathlib import Path
import urllib.request

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# === Event parameters ===
FLARE_PEAK = datetime(2026, 3, 30, 3, 19, tzinfo=timezone.utc)
CME_ARRIVAL = datetime(2026, 3, 31, 15, 7, tzinfo=timezone.utc)
CME_SPEED = 1689
VANUATU_EQ = datetime(2026, 3, 28, 22, 4, tzinfo=timezone.utc)
SUBSOLAR_LAT = 3.6
SUBSOLAR_LON = -46.8

# Lunar
REF_NEW_MOON = datetime(2000, 1, 6, tzinfo=timezone.utc)
SYNODIC = 29.53059

def lunar_phase(dt):
    days = (dt - REF_NEW_MOON).total_seconds() / 86400
    return (days % SYNODIC) / SYNODIC

def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = [math.radians(x) for x in [lat1, lon1, lat2, lon2]]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))

def fetch_json(url, timeout=15):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'GR-Dashboard/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [WARN] {url}: {e}")
        return None

# === Data fetchers ===

def get_goes_xrs():
    """Get GOES X-ray flux (1-min, 7-day) -- Schumann proxy."""
    data = fetch_json('https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json')
    if not data: return [], []
    times, flux = [], []
    for row in data:
        if row.get('energy') != '0.1-0.8nm': continue
        try:
            t = datetime.strptime(row['time_tag'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            f = float(row['flux'])
            if f > 0:
                times.append(t)
                flux.append(f)
        except: pass
    return times, flux

def get_kp():
    data = fetch_json('https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json')
    if not data: return [], []
    times, vals = [], []
    for row in data[1:]:  # skip header
        try:
            t = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            times.append(t)
            vals.append(float(row[1]))
        except: pass
    return times, vals

def get_solar_wind_mag():
    data = fetch_json('https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json')
    if not data: return [], [], []
    times, bz, bt = [], [], []
    for row in data[1:]:
        try:
            t = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            bz_val = float(row[3]) if row[3] not in (None, '', 'null') else np.nan
            bt_val = float(row[6]) if row[6] not in (None, '', 'null') else np.nan
            times.append(t)
            bz.append(bz_val)
            bt.append(bt_val)
        except: pass
    return times, bz, bt

def get_solar_wind_plasma():
    data = fetch_json('https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json')
    if not data: return [], [], []
    times, density, speed = [], [], []
    for row in data[1:]:
        try:
            t = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            d = float(row[1]) if row[1] not in (None, '', 'null') else np.nan
            v = float(row[2]) if row[2] not in (None, '', 'null') else np.nan
            times.append(t)
            density.append(d)
            speed.append(v)
        except: pass
    return times, density, speed

def get_earthquakes(hours_back=72):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_back)
    url = (f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson'
           f'&starttime={start.strftime("%Y-%m-%dT%H:%M:%S")}'
           f'&minmagnitude=4.5&orderby=time&limit=500')
    data = fetch_json(url)
    if not data or 'features' not in data: return []
    eqs = []
    for f in data['features']:
        p = f['properties']
        c = f['geometry']['coordinates']
        try:
            eqs.append({
                'mag': p.get('mag', 0),
                'time': datetime.fromtimestamp(p['time']/1000, tz=timezone.utc),
                'lon': c[0], 'lat': c[1], 'depth': c[2],
                'place': p.get('place', ''),
                'ang_dist': angular_distance(SUBSOLAR_LAT, SUBSOLAR_LON, c[1], c[0]),
            })
        except: pass
    return eqs

def load_swarm():
    """Load Swarm data if available from earlier pull."""
    f = OUT_DIR / 'swarm_fast_20260331.csv'
    if not f.exists(): return None
    try:
        import pandas as pd
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        return df
    except:
        return None

# === Dashboard ===

def build_dashboard():
    now = datetime.now(timezone.utc)
    print(f"Building dashboard at {now.strftime('%Y-%m-%d %H:%M:%S')} UTC...")

    # Fetch all data
    print("  Fetching Kp...")
    kp_t, kp_v = get_kp()
    print("  Fetching solar wind mag...")
    sw_t, bz, bt = get_solar_wind_mag()
    print("  Fetching solar wind plasma...")
    pl_t, density, speed = get_solar_wind_plasma()
    print("  Fetching GOES XRS (Schumann proxy)...")
    xrs_t, xrs_f = get_goes_xrs()
    print("  Fetching earthquakes...")
    eqs = get_earthquakes(72)
    print("  Loading Swarm...")
    swarm = load_swarm()

    phase = lunar_phase(now)
    hours_since_flare = (now - FLARE_PEAK).total_seconds() / 3600
    hours_since_arrival = (now - CME_ARRIVAL).total_seconds() / 3600

    # --- Figure ---
    fig = plt.figure(figsize=(20, 17), facecolor='#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')
    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.97, top=0.94, bottom=0.03)

    title_color = '#00ccff'
    text_color = '#cccccc'
    grid_color = '#222244'
    warn_color = '#ff4444'
    safe_color = '#44ff44'

    def style_ax(ax, title=''):
        ax.set_facecolor('#0d0d2b')
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines['bottom'].set_color(grid_color)
        ax.spines['left'].set_color(grid_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if title:
            ax.set_title(title, color=title_color, fontsize=10, fontweight='bold', pad=6)

    # === Title ===
    fig.suptitle(
        f"CME IMPACT MONITOR  |  X1.4 / 1689 km/s  |  {now.strftime('%Y-%m-%d %H:%M')} UTC  |  "
        f"Flare +{hours_since_flare:.0f}h  |  Arrival {'+' if hours_since_arrival>=0 else ''}{hours_since_arrival:.0f}h  |  "
        f"Moon: {phase:.0%} ({'FULL' if abs(phase-0.5)<0.05 else 'Gibbous' if phase>0.4 else 'Crescent'})",
        color=title_color, fontsize=13, fontweight='bold', y=0.97
    )

    # === Panel 1: Bz + Bt (top-left) ===
    ax1 = fig.add_subplot(gs[0, 0:2])
    style_ax(ax1, 'IMF Bz & Bt (1-day)')
    if sw_t:
        ax1.plot(sw_t, bz, color='#ff6666', linewidth=0.8, label='Bz')
        ax1.plot(sw_t, bt, color='#6666ff', linewidth=0.8, alpha=0.5, label='Bt')
        ax1.axhline(0, color='#444444', linewidth=0.5)
        ax1.axhline(-10, color=warn_color, linewidth=0.5, linestyle='--', alpha=0.5)
        ax1.fill_between(sw_t, bz, 0, where=[b < 0 for b in bz],
                        color='#ff4444', alpha=0.15)
        ax1.axvline(CME_ARRIVAL, color='#ffff00', linewidth=1, linestyle='--', alpha=0.7)
        ax1.text(CME_ARRIVAL, ax1.get_ylim()[1]*0.9, ' CME\n arrival',
                color='#ffff00', fontsize=7, va='top')
    ax1.set_ylabel('nT', color=text_color, fontsize=8)
    ax1.legend(fontsize=7, loc='upper left', facecolor='#0d0d2b', edgecolor=grid_color,
              labelcolor=text_color)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # === Panel 2: Solar wind speed + density (top-right area) ===
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, 'V_sw & Density')
    if pl_t:
        ax2.plot(pl_t, speed, color='#44ffcc', linewidth=0.8)
        ax2.axhline(500, color=warn_color, linewidth=0.5, linestyle='--', alpha=0.5)
        ax2.axvline(CME_ARRIVAL, color='#ffff00', linewidth=1, linestyle='--', alpha=0.5)
        current_v = speed[-1] if speed else 0
        ax2.text(0.95, 0.95, f'{current_v:.0f} km/s',
                transform=ax2.transAxes, color='#44ffcc', fontsize=12,
                fontweight='bold', ha='right', va='top')
    ax2.set_ylabel('km/s', color=text_color, fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # === Panel 3: Kp index ===
    ax3 = fig.add_subplot(gs[0, 3])
    style_ax(ax3, 'Kp Index (3-day)')
    if kp_t:
        colors_kp = ['#44ff44' if v < 4 else '#ffff44' if v < 6 else '#ff8844' if v < 8 else '#ff4444'
                     for v in kp_v]
        ax3.bar(kp_t, kp_v, width=timedelta(hours=2.8), color=colors_kp, alpha=0.8)
        ax3.axhline(5, color=warn_color, linewidth=0.5, linestyle='--', alpha=0.5)
        ax3.axvline(CME_ARRIVAL, color='#ffff00', linewidth=1, linestyle='--', alpha=0.5)
        current_kp = kp_v[-1] if kp_v else 0
        ax3.text(0.95, 0.95, f'Kp={current_kp:.0f}',
                transform=ax3.transAxes, color='#44ff44' if current_kp < 5 else '#ff4444',
                fontsize=14, fontweight='bold', ha='right', va='top')
    ax3.set_ylabel('Kp', color=text_color, fontsize=8)
    ax3.set_ylim(0, 9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

    # === Panel 4: Earthquake map ===
    ax4 = fig.add_subplot(gs[1, 0:2])
    style_ax(ax4, 'M4.5+ Earthquakes (72h) + Jelly Ball Zones')
    ax4.set_xlim(-180, 180)
    ax4.set_ylim(-90, 90)
    ax4.set_aspect('equal')
    ax4.set_xlabel('Longitude', color=text_color, fontsize=7)
    ax4.set_ylabel('Latitude', color=text_color, fontsize=7)

    # Draw zone circles from subsolar point
    for radius, color, label in [(30, '#4444ff', 'Eye'), (60, '#44ff44', ''),
                                   (100, '#ff4444', 'Wavefront'), (140, '#ffff44', '')]:
        theta = np.linspace(0, 2*np.pi, 100)
        # Approximate circle on map (not great-circle, but close enough for viz)
        clat = SUBSOLAR_LAT + radius * np.sin(theta)
        clon = SUBSOLAR_LON + radius * np.cos(theta) / max(0.1, np.cos(np.radians(SUBSOLAR_LAT)))
        ax4.plot(clon, clat, color=color, linewidth=0.5, alpha=0.3)

    # Subsolar point
    ax4.plot(SUBSOLAR_LON, SUBSOLAR_LAT, '*', color='#ffff00', markersize=12, zorder=10)
    ax4.text(SUBSOLAR_LON+5, SUBSOLAR_LAT+5, 'Subsolar', color='#ffff00', fontsize=6)

    # Earthquakes
    if eqs:
        lons = [e['lon'] for e in eqs]
        lats = [e['lat'] for e in eqs]
        mags = [e['mag'] for e in eqs]
        dists = [e['ang_dist'] for e in eqs]
        # Color by zone
        eq_colors = []
        for d in dists:
            if d < 30: eq_colors.append('#4444ff')      # eye
            elif d < 60: eq_colors.append('#44ff44')     # inner
            elif d < 100: eq_colors.append('#ff4444')    # wavefront
            elif d < 140: eq_colors.append('#ffff44')    # outer
            else: eq_colors.append('#888888')            # far/antipodal
        sizes = [(m - 3.5)**2.5 * 3 for m in mags]
        ax4.scatter(lons, lats, s=sizes, c=eq_colors, alpha=0.7, edgecolors='white',
                   linewidths=0.3, zorder=5)

        # Highlight M6+
        big = [e for e in eqs if e['mag'] >= 6.0]
        for e in big:
            ax4.annotate(f"M{e['mag']:.1f}", (e['lon'], e['lat']),
                        color='white', fontsize=6, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')

    # High-risk targets
    targets = [('Iceland', 64.1, -21.9), ('Italy', 42.5, 13.5),
               ('Turkey', 41.0, 29.0), ('Cascadia', 45.5, -122.7),
               ('Iran', 38.1, 46.3), ('Alaska', 61.2, -149.9)]
    for name, lat, lon in targets:
        ax4.plot(lon, lat, 'v', color='#ff4444', markersize=5, alpha=0.7)
        ax4.text(lon, lat-6, name, color='#ff4444', fontsize=5, ha='center')

    # === Panel 5: Zone histogram ===
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, 'Quakes by Angular Distance')
    if eqs:
        bins = [0, 30, 60, 100, 140, 165, 180]
        bin_labels = ['0-30\nEye', '30-60', '60-100\nWave', '100-140', '140-165', '165-180\nAnti']
        counts, _ = np.histogram([e['ang_dist'] for e in eqs], bins=bins)
        # Solid angle normalization
        solid_angles = [2*np.pi*(np.cos(np.radians(bins[i])) - np.cos(np.radians(bins[i+1])))
                       for i in range(len(bins)-1)]
        solid_angles = np.array(solid_angles) / (4*np.pi)  # fraction of sphere
        density = counts / np.maximum(solid_angles, 1e-6)
        density = density / np.mean(density) if np.mean(density) > 0 else density

        bar_colors = ['#4444ff', '#44cc44', '#ff4444', '#ffff44', '#888888', '#cc88cc']
        ax5.bar(range(len(counts)), density, color=bar_colors, alpha=0.8)
        ax5.axhline(1.0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
        ax5.set_xticks(range(len(counts)))
        ax5.set_xticklabels(bin_labels, fontsize=6, color=text_color)
        ax5.set_ylabel('Rate / baseline', color=text_color, fontsize=8)
        # Annotate predicted vs actual
        predicted = [0.85, 1.05, 1.36, 1.10, 1.05, 1.16]
        for i, (d, p) in enumerate(zip(density, predicted)):
            ax5.text(i, d + 0.05, f'{d:.2f}', ha='center', fontsize=6, color='white')

    # === Panel 6: Timeline ===
    ax6 = fig.add_subplot(gs[1, 3])
    style_ax(ax6, 'Grade Timeline')
    ax6.set_xlim(-5, 55)
    ax6.set_ylim(-0.5, 4.5)
    ax6.set_xlabel('Hours since flare', color=text_color, fontsize=7)

    # Grade windows
    grade_bars = [
        (0, 3, '#4444ff', 'Grade-0\nSID suppress'),
        (15, 21, '#cc44cc', 'Grade-4\nIono relax'),
        (30, 50, '#ff4444', 'Grade-2\nCME mech'),
    ]
    for start, end, color, label in grade_bars:
        ax6.barh(2, end-start, left=start, height=0.6, color=color, alpha=0.6)
        ax6.text((start+end)/2, 2, label, ha='center', va='center',
                color='white', fontsize=6, fontweight='bold')

    # Current time marker
    ax6.axvline(hours_since_flare, color='#00ff00', linewidth=2)
    ax6.text(hours_since_flare, 3.5, f'NOW\n+{hours_since_flare:.0f}h',
            ha='center', color='#00ff00', fontsize=8, fontweight='bold')

    # Vanuatu M7.3 marker
    vanuatu_h = (VANUATU_EQ - FLARE_PEAK).total_seconds() / 3600
    if -50 < vanuatu_h < 55:
        ax6.axvline(vanuatu_h, color='#ffff00', linewidth=1, linestyle='--')
        ax6.text(vanuatu_h, 0.5, 'M7.3\nVanuatu', ha='center', color='#ffff00', fontsize=6)

    ax6.set_yticks([])

    # === Panel 7: GOES X-ray flux (Schumann proxy) ===
    ax7 = fig.add_subplot(gs[2, 0:2])
    style_ax(ax7, 'GOES X-ray Flux (Schumann Order Parameter Proxy)')
    if xrs_t and len(xrs_t) > 10:
        xrs_arr = np.array(xrs_f)
        ax7.semilogy(xrs_t, xrs_f, color='#ff8844', linewidth=0.6, alpha=0.9)
        # Flare class lines
        for level, label, col in [(1e-4, 'X', '#ff4444'), (1e-5, 'M', '#ffaa44'),
                                   (1e-6, 'C', '#44aaff'), (1e-7, 'B', '#444488')]:
            ax7.axhline(level, color=col, linewidth=0.4, linestyle='--', alpha=0.4)
            ax7.text(xrs_t[0], level * 1.2, f' {label}', color=col, fontsize=6, va='bottom')
        # Mark X1.4 and M7.3
        ax7.axvline(FLARE_PEAK, color='#ff4444', linewidth=1.5, linestyle='-', alpha=0.8)
        ax7.axvline(datetime(2026, 3, 30, 8, 44, 13, tzinfo=timezone.utc),
                   color='#ffff00', linewidth=1.5, linestyle='-', alpha=0.8)
        ax7.text(FLARE_PEAK, ax7.get_ylim()[1] if ax7.get_ylim()[1] > 0 else 1e-3,
                ' X1.4', color='#ff4444', fontsize=7, va='top', rotation=90)
        ax7.text(datetime(2026, 3, 30, 8, 44, tzinfo=timezone.utc),
                1e-7, ' M7.3', color='#ffff00', fontsize=7, va='bottom', rotation=90)
        # CME arrival
        ax7.axvline(CME_ARRIVAL, color='#00ff00', linewidth=1, linestyle='--', alpha=0.6)
        ax7.set_ylabel('W/m2', color=text_color, fontsize=8)
        ax7.set_ylim(1e-8, 5e-4)
    else:
        ax7.text(0.5, 0.5, 'GOES XRS data unavailable',
                transform=ax7.transAxes, color='#666666', fontsize=10, ha='center')
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

    # === Panel 7b: df/dt Order Parameter ===
    ax7b = fig.add_subplot(gs[2, 2:4])
    style_ax(ax7b, 'dFlux/dt: KT Order Parameter (J crossing J_c)')
    if xrs_t and len(xrs_t) > 100:
        # Compute df/dt with 15-min smoothing
        xrs_arr = np.array(xrs_f)
        # Smooth first
        kernel = 15  # 15-minute window
        if len(xrs_arr) > kernel * 2:
            smoothed = np.convolve(xrs_arr, np.ones(kernel)/kernel, mode='same')
            dfdt = np.gradient(smoothed)
            # Normalize to fractional rate
            dfdt_frac = dfdt / np.maximum(smoothed, 1e-8)

            # Color: red when falling (J crossing J_c = earthquake risk)
            colors = ['#ff4444' if d < 0 else '#44ff44' for d in dfdt_frac]
            # Plot as filled
            xrs_t_arr = np.array(xrs_t)
            ax7b.fill_between(xrs_t, dfdt_frac, 0,
                            where=[d < 0 for d in dfdt_frac],
                            color='#ff4444', alpha=0.4, label='df/dt < 0 (RISK)')
            ax7b.fill_between(xrs_t, dfdt_frac, 0,
                            where=[d >= 0 for d in dfdt_frac],
                            color='#44ff44', alpha=0.3, label='df/dt > 0 (stable)')
            ax7b.plot(xrs_t, dfdt_frac, color='white', linewidth=0.4, alpha=0.5)
            ax7b.axhline(0, color='#888888', linewidth=0.5)

            # Mark flare and EQ
            ax7b.axvline(FLARE_PEAK, color='#ff4444', linewidth=1, alpha=0.7)
            ax7b.axvline(datetime(2026, 3, 30, 8, 44, tzinfo=timezone.utc),
                        color='#ffff00', linewidth=1.5)
            ax7b.axvline(CME_ARRIVAL, color='#00ff00', linewidth=1, linestyle='--', alpha=0.6)

            # Current state indicator
            if len(dfdt_frac) > 0:
                current_dfdt = dfdt_frac[-1]
                state = 'FALLING (J -> J_c)' if current_dfdt < -0.001 else \
                        'RISING (J > J_c)' if current_dfdt > 0.001 else 'STABLE'
                state_color = '#ff4444' if current_dfdt < -0.001 else \
                             '#44ff44' if current_dfdt > 0.001 else '#888888'
                ax7b.text(0.98, 0.95, state, transform=ax7b.transAxes,
                         color=state_color, fontsize=11, fontweight='bold',
                         ha='right', va='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a1a',
                                  edgecolor=state_color, alpha=0.9))

            ax7b.legend(fontsize=6, loc='upper left', facecolor='#0d0d2b',
                       edgecolor=grid_color, labelcolor=text_color)
            ax7b.set_ylabel('d(flux)/dt / flux', color=text_color, fontsize=7)

            # Clip y-axis to avoid noise spikes
            ylim = max(0.1, np.percentile(np.abs(dfdt_frac[~np.isnan(dfdt_frac)]), 99))
            ax7b.set_ylim(-ylim, ylim)
    else:
        ax7b.text(0.5, 0.5, 'Insufficient XRS data for derivative',
                 transform=ax7b.transAxes, color='#666666', fontsize=10, ha='center')
    ax7b.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

    # === Panel 8: Lunar / tidal ===
    ax8 = fig.add_subplot(gs[3, 0])
    style_ax(ax8, 'Lunar Phase & Tidal Stress')

    # Draw moon phase
    theta = np.linspace(0, 2*np.pi, 100)
    ax8.plot(np.cos(theta)*0.3 + 0.5, np.sin(theta)*0.3 + 0.65, color='#888888', linewidth=1)
    # Illuminated portion
    illum_phase = phase
    if illum_phase <= 0.5:
        # Waxing: right side lit
        x_term = np.cos(np.pi * (1 - 2*illum_phase))
        for t in theta:
            if np.cos(t) > 0:  # right side
                ax8.plot(np.cos(t)*0.3 + 0.5, np.sin(t)*0.3 + 0.65, '.', color='#ffffcc', markersize=1)
            elif np.cos(t) > x_term:
                ax8.plot(np.cos(t)*0.3 + 0.5, np.sin(t)*0.3 + 0.65, '.', color='#ffffcc', markersize=1)

    tidal_force = np.cos(2 * np.pi * phase)
    tidal_rate = -np.sin(2 * np.pi * phase)
    days_to_full = ((0.5 - phase) % 1.0) * SYNODIC

    ax8.text(0.5, 0.30, f'Phase: {phase:.0%}', color=text_color, fontsize=9,
            ha='center', transform=ax8.transAxes)
    ax8.text(0.5, 0.20, f'Full in {days_to_full:.1f}d', color='#ffff88', fontsize=8,
            ha='center', transform=ax8.transAxes)
    ax8.text(0.5, 0.10, f'Tidal force: {tidal_force:+.2f}', color=text_color, fontsize=8,
            ha='center', transform=ax8.transAxes)
    ax8.text(0.5, 0.02, f'dF/dt: {tidal_rate:+.2f}', color=text_color, fontsize=8,
            ha='center', transform=ax8.transAxes)
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.set_xticks([])
    ax8.set_yticks([])

    # === Panel 9: Prediction scorecard ===
    ax9 = fig.add_subplot(gs[3, 1])
    style_ax(ax9, 'Prediction Scorecard')
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.set_xticks([])
    ax9.set_yticks([])

    current_kp = kp_v[-1] if kp_v else 0
    current_v = speed[-1] if speed else 0
    current_bz = bz[-1] if bz else 0

    predictions = [
        ('Kp spike to 5+', current_kp >= 5, current_kp < 5 and hours_since_arrival < 7,
         f'Kp={current_kp:.0f}'),
        ('V_sw > 500 km/s', current_v > 500, current_v <= 500 and hours_since_arrival < 7,
         f'V={current_v:.0f}'),
        ('Bz < -10 nT', current_bz < -10, current_bz >= -10 and hours_since_arrival < 7,
         f'Bz={current_bz:.1f}'),
        ('Wavefront M5+', False, True, 'pending'),
        ('Eye suppression', False, True, 'pending'),
    ]

    for i, (label, confirmed, pending, detail) in enumerate(predictions):
        y = 0.88 - i * 0.18
        if confirmed:
            color = safe_color
            sym = '[OK]'
        elif pending:
            color = '#ffff44'
            sym = '[..]'
        else:
            color = warn_color
            sym = '[NO]'
        ax9.text(0.05, y, sym, color=color, fontsize=9, fontweight='bold',
                transform=ax9.transAxes, fontfamily='monospace')
        ax9.text(0.20, y, label, color=text_color, fontsize=8,
                transform=ax9.transAxes)
        ax9.text(0.95, y, detail, color=color, fontsize=7, ha='right',
                transform=ax9.transAxes, fontfamily='monospace')

    # === Panel 11: Coupling timescale reference ===
    ax11 = fig.add_subplot(gs[3, 2:4])
    style_ax(ax11, 'Coupling Timescales (from case studies)')
    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.set_xticks([])
    ax11.set_yticks([])

    timescales = [
        (0.90, '~3 min',   '10km',  'Internal EM pulse (Hubbard M7.0/M8.1 Dec 2025)', '#ff4444'),
        (0.72, '~30 min',  '18km',  'Ionospheric redistribution (Japan M6.8/X1.7 Nov 2025)', '#ffaa44'),
        (0.54, '~5 hours', '121km', 'EM diffusion to depth (Vanuatu M7.3/X1.5 Mar 2026)', '#44aaff'),
        (0.36, '~18 hours','any',   'Grade-4: iono relaxes through J_c (peak seismic risk)', '#ff44ff'),
        (0.18, '~36 hours','any',   'Grade-2: CME mechanical coupling (3.26x wavefront)', '#44ff44'),
    ]

    ax11.text(0.02, 0.98, 'Delay     Depth    Mechanism', color='#888888', fontsize=7,
             transform=ax11.transAxes, va='top', fontfamily='monospace')
    for y, delay, depth, desc, col in timescales:
        ax11.text(0.02, y, delay, color=col, fontsize=8, fontweight='bold',
                 transform=ax11.transAxes, fontfamily='monospace')
        ax11.text(0.15, y, depth, color=text_color, fontsize=7,
                 transform=ax11.transAxes, fontfamily='monospace')
        ax11.text(0.27, y, desc, color=text_color, fontsize=7,
                 transform=ax11.transAxes)

    # Highlight current phase
    if hours_since_flare < 0.1:
        current_ts = 'PRE-FLARE'
    elif hours_since_flare < 6:
        current_ts = 'Grade-0: INTERNAL EM + SID (0-6h)'
    elif hours_since_flare < 21:
        current_ts = 'Grade-4: IONOSPHERIC RELAXATION -- PEAK RISK'
    elif hours_since_flare < 48:
        current_ts = 'Grade-2: CME MECHANICAL -- WAVEFRONT ENHANCED'
    else:
        current_ts = 'POST-EVENT'
    ax11.text(0.5, 0.02, f'CURRENT: {current_ts}', color='#00ffff', fontsize=9,
             fontweight='bold', ha='center', transform=ax11.transAxes,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a1a',
                      edgecolor='#00ffff', alpha=0.9))

    # === Panel 10: Status text ===
    ax10 = fig.add_subplot(gs[4, :])
    style_ax(ax10, '')
    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, 1)
    ax10.set_xticks([])
    ax10.set_yticks([])

    # CME status
    if current_v > 600:
        cme_status = "CME ARRIVED -- STORM IN PROGRESS"
        cme_color = warn_color
    elif hours_since_arrival > 0 and current_v < 500:
        cme_status = f"CME OVERDUE by {hours_since_arrival:.0f}h (within +/-7h uncertainty) -- WATCH FOR SUDDEN ONSET"
        cme_color = '#ffff44'
    else:
        cme_status = f"CME arrival in ~{-hours_since_arrival:.0f}h"
        cme_color = '#44ffcc'

    status_lines = [
        (f"STATUS: {cme_status}", cme_color, 14),
        (f"X1.4 (AR 14405 S27E45) | 1689 km/s | 46deg half-angle | 92% Earth impact | Kp forecast 6-9",
         text_color, 9),
        (f"High-risk targets (60-100deg wavefront): Iceland, Central Italy, Istanbul, Cascadia, Iran, Alaska | "
         f"Suppression zone: Caribbean/equatorial Atlantic | "
         f"Full Moon in {days_to_full:.1f} days",
         text_color, 9),
    ]

    for i, (txt, col, size) in enumerate(status_lines):
        ax10.text(0.5, 0.75 - i * 0.30, txt, color=col, fontsize=size,
                 ha='center', va='center', transform=ax10.transAxes, fontweight='bold')

    # === Save ===
    outfile = OUT_DIR / 'dashboard.png'
    fig.savefig(outfile, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nDashboard saved to {outfile}")
    return str(outfile)


if __name__ == '__main__':
    build_dashboard()
